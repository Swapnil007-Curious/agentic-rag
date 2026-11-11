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
from collections.abc import Callable
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


def _encode_batched(
    encoder: Encoder,
    corpus: ChunkCorpus,
    out_dir: Path,
    progress: Callable[[str], None],
    batch_rows: int,
) -> np.ndarray:
    """Encode in batches of ``batch_rows`` chunks with a progress line (rate, ETA) after each;
    a partial result is saved after every batch (``<chunker>.partial.npy`` + ``.json``) and
    reused by the next run when the model and chunk texts are unchanged."""
    part_npy = out_dir / f"{corpus.chunker}.partial.npy"
    part_meta = out_dir / f"{corpus.chunker}.partial.json"
    texts = corpus.texts
    key = {"model": encoder.name, "texts_sha256": corpus.texts_hash(), "n": len(texts)}
    done: list[np.ndarray] = []
    n_done = 0
    if part_npy.exists() and part_meta.exists():
        try:
            meta = json.loads(part_meta.read_text(encoding="utf-8"))
            if {k: meta.get(k) for k in key} == key:
                prev = np.load(part_npy, allow_pickle=False)
                done, n_done = [prev], int(prev.shape[0])
                progress(
                    f"      resuming {encoder.name} / {corpus.chunker} at {n_done}/{len(texts)}"
                )
        except (json.JSONDecodeError, OSError, ValueError):
            done, n_done = [], 0
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    start_done = n_done
    while n_done < len(texts):
        batch = texts[n_done : n_done + batch_rows]
        done.append(encoder.encode(batch, is_query=False).astype(np.float32))
        n_done += len(batch)
        arr = np.concatenate(done) if len(done) > 1 else done[0]
        done = [arr]
        tmp = part_npy.with_name(part_npy.stem + ".tmp.npy")
        np.save(tmp, arr, allow_pickle=False)
        tmp.replace(part_npy)
        part_meta.write_text(json.dumps({**key, "n_done": n_done}), encoding="utf-8")
        rate = (n_done - start_done) / max(time.perf_counter() - t0, 1e-6)
        eta = (len(texts) - n_done) / rate if rate > 0 else 0.0
        progress(
            f"      {encoder.name} / {corpus.chunker}: {n_done}/{len(texts)} chunks "
            f"({rate:.2f} chunks/s, ETA {eta / 60:.1f} min)"
        )
    result = np.concatenate(done) if done else np.zeros((0, 0), dtype=np.float32)
    for f in (part_npy, part_meta):
        f.unlink(missing_ok=True)
    return result


def corpus_embeddings(
    encoder: Encoder,
    corpus: ChunkCorpus,
    emb_dir: Path,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
    batch_rows: int = 64,
) -> tuple[np.ndarray, CacheInfo]:
    """Load the cached embeddings of ``corpus`` or (re)compute and cache them. With
    ``progress``, encoding runs in resumable batches that report progress and an ETA."""
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
    if progress is not None and len(corpus) > 0:
        emb = _encode_batched(encoder, corpus, out_dir, progress, batch_rows)
    else:
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
