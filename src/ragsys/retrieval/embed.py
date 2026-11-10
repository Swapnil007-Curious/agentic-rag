"""Dense encoders (sentence-transformers, CPU) and the on-disk corpus embedding cache.

Cache layout: ``<emb_dir>/<model_slug>/<chunker>.npy`` (float32, one row per chunk in file
order) plus ``<chunker>.json`` with the model name, the chunk_id order and a SHA-256 of the
embedded texts. A cache whose sidecar does not match the current chunks is rebuilt.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ragsys.retrieval.config import DEFAULT_QUERY_INSTRUCTION
from ragsys.retrieval.corpus import ChunkCorpus

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class Encoder(Protocol):
    """Maps texts to L2-normalised float32 vectors."""

    name: str

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """bge-* (or any sentence-transformers) model on CPU with normalised embeddings.

    ``query_instruction`` is prepended to queries only (bge v1.5 retrieval convention).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        query_instruction: str | None = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            quiet_transformers()
            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension()), dtype=np.float32)
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        vecs = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())


def quiet_transformers() -> None:
    """Silence transformers' weight-loading progress bars and info logs in script output."""
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # type: ignore[no-untyped-call]
        hf_logging.disable_progress_bar()  # type: ignore[no-untyped-call]
    except Exception:  # pragma: no cover - cosmetic only
        pass


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model_name).strip("_")


@dataclass
class CacheInfo:
    npy_path: Path
    rebuilt: bool
    reason: str
    seconds: float
    n: int
    dim: int
    bytes: int


def _sidecar_matches(meta_path: Path, npy_path: Path, model: str, corpus: ChunkCorpus) -> str:
    """Empty string when the cache is valid, otherwise the reason it is stale."""
    if not npy_path.exists() or not meta_path.exists():
        return "missing"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "unreadable sidecar"
    if meta.get("model") != model:
        return "model changed"
    if meta.get("chunk_ids") != corpus.ids:
        return "chunk ids changed"
    if meta.get("texts_sha256") != corpus.texts_hash():
        return "chunk texts changed"
    return ""


def corpus_embeddings(
    encoder: Encoder, corpus: ChunkCorpus, emb_dir: Path, force: bool = False
) -> tuple[np.ndarray, CacheInfo]:
    """Load the cached embeddings of ``corpus`` or (re)compute and cache them."""
    out_dir = emb_dir / model_slug(encoder.name)
    npy_path = out_dir / f"{corpus.chunker}.npy"
    meta_path = out_dir / f"{corpus.chunker}.json"
    reason = "forced" if force else _sidecar_matches(meta_path, npy_path, encoder.name, corpus)
    if not reason:
        t0 = time.perf_counter()
        emb = np.load(npy_path, allow_pickle=False)
        if emb.shape[0] == len(corpus):
            info = CacheInfo(
                npy_path,
                False,
                "cached",
                time.perf_counter() - t0,
                emb.shape[0],
                emb.shape[1],
                npy_path.stat().st_size,
            )
            return emb.astype(np.float32, copy=False), info
        reason = "row count mismatch"
    t0 = time.perf_counter()
    emb = encoder.encode(corpus.texts, is_query=False).astype(np.float32)
    seconds = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = npy_path.with_name(npy_path.stem + ".tmp.npy")
    np.save(tmp, emb, allow_pickle=False)
    tmp.replace(npy_path)
    dim = int(emb.shape[1]) if emb.ndim == 2 else 0
    meta: dict[str, object] = {
        "model": encoder.name,
        "chunker": corpus.chunker,
        "n": int(emb.shape[0]),
        "dim": dim,
        "normalized": True,
        "texts_sha256": corpus.texts_hash(),
        "encode_seconds": round(seconds, 3),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunk_ids": corpus.ids,
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    info = CacheInfo(
        npy_path,
        True,
        reason,
        seconds,
        int(emb.shape[0]),
        dim,
        npy_path.stat().st_size,
    )
    return emb, info
