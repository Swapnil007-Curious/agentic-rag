"""In-memory dense indexes searched by the assistant: a collection's app index (structural chunks
and their bge-small embeddings, read from disk without writing anything) or the chunks of an
uploaded PDF (embedded in memory)."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ragsys.chunking.base import Chunk
from ragsys.retrieval.corpus import load_chunks, texts_hash
from ragsys.retrieval.embed import Encoder, model_slug


class IndexUnavailable(RuntimeError):
    """The app index of a collection is missing or does not match its chunks."""


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str  # collection name, or "upload"


class LockedEncoder:
    """An encoder shared by threads (query embedding, upload embedding): one call at a time."""

    def __init__(self, encoder: Encoder) -> None:
        self.inner = encoder
        self.name = encoder.name
        self._lock = threading.Lock()

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        with self._lock:
            return np.asarray(self.inner.encode(texts, is_query=is_query), dtype=np.float32)


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(norms > 0, x / np.where(norms > 0, norms, 1.0), x).astype(np.float32)


class DenseIndex:
    """Chunks and their L2-normalised embeddings (one row per chunk)."""

    def __init__(self, chunks: Sequence[Chunk], matrix: np.ndarray, source: str) -> None:
        if len(chunks) != matrix.shape[0]:
            raise ValueError("chunks and embeddings differ in length")
        self.chunks = list(chunks)
        self.matrix = _normalise(matrix) if len(chunks) else np.zeros((0, 1), dtype=np.float32)
        self.source = source
        self.by_id = {c.chunk_id: c for c in self.chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def n_tokens(self) -> int:
        return sum(c.n_tokens for c in self.chunks)

    @classmethod
    def load(
        cls, processed_dir: Path, emb_dir: Path, model_name: str, chunker: str, source: str
    ) -> DenseIndex:
        """A built app index; raises :class:`IndexUnavailable` when the embeddings are missing
        or were computed for other chunks (nothing is recomputed or written)."""
        try:
            corpus = load_chunks(processed_dir, chunker)
        except FileNotFoundError as exc:
            raise IndexUnavailable(f"{source}: no {chunker} chunks in {processed_dir}") from exc
        npy = emb_dir / model_slug(model_name) / f"{chunker}.npy"
        meta_path = npy.with_suffix(".json")
        if not npy.exists() or not meta_path.exists():
            raise IndexUnavailable(f"{source}: the {model_name} embeddings are not built ({npy})")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("chunk_ids") != corpus.ids or meta.get("texts_sha256") != corpus.texts_hash():
            raise IndexUnavailable(f"{source}: {npy.name} does not match the current chunks")
        matrix = np.load(npy, allow_pickle=False)
        return cls(corpus.chunks, matrix, source)

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        encoder: Encoder | LockedEncoder,
        source: str,
        *,
        batch_size: int = 16,
        progress: Callable[[int, int], None] | None = None,
    ) -> DenseIndex:
        """Embed ``chunks`` in memory (batches, so the encoder is shared fairly)."""
        rows: list[np.ndarray] = []
        texts = [c.text for c in chunks]
        for i in range(0, len(texts), batch_size):
            rows.append(encoder.encode(texts[i : i + batch_size], is_query=False))
            if progress is not None:
                progress(min(i + batch_size, len(texts)), len(texts))
        matrix = np.concatenate(rows) if rows else np.zeros((0, 1), dtype=np.float32)
        return cls(chunks, matrix, source)

    def search(self, queries: np.ndarray, k: int) -> list[Hit]:
        """Top ``k`` chunks by the best cosine over the query rows (ties: earlier chunk)."""
        if not self.chunks or k <= 0:
            return []
        q = _normalise(np.atleast_2d(np.asarray(queries, dtype=np.float32)))
        scores = (self.matrix @ q.T).max(axis=1)
        k = min(k, len(self.chunks))
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [Hit(self.chunks[i], float(scores[i]), self.source) for i in order]

    def texts_digest(self) -> str:
        return texts_hash([c.text for c in self.chunks])
