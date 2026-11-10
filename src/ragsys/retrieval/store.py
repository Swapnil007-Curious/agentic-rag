"""Vector stores over precomputed, normalised embeddings.

* :class:`NumpyStore`: exact cosine search with one matrix product (default; used for every
  experiment).
* :class:`ChromaStore`: chromadb ``PersistentClient`` collection in cosine space with the same
  precomputed embeddings (no Chroma embedding function). Kept for parity checks and as the
  path to larger corpora.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class VectorStore(Protocol):
    def __len__(self) -> int: ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top-``k`` (id, cosine similarity), best first."""
        ...


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(norms > 0, x / np.where(norms > 0, norms, 1.0), x)


class NumpyStore:
    def __init__(self, ids: list[str], embeddings: np.ndarray) -> None:
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ids and embeddings differ in length")
        self.ids = list(ids)
        self.matrix = _normalise(embeddings)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self.ids or k <= 0:
            return []
        q = _normalise(np.asarray(query, dtype=np.float32).reshape(-1))
        scores = self.matrix @ q
        k = min(k, len(self.ids))
        # stable: higher score first, earlier row on ties
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [(self.ids[i], float(scores[i])) for i in order]


def chroma_collection_name(chunker: str, model_name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{chunker}__{model_name}")
    name = name.strip("._-")[:512]
    return name if len(name) >= 3 else f"col_{name}"


class ChromaStore:
    """chromadb collection holding precomputed embeddings (cosine distance)."""

    def __init__(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        collection: str,
        persist_dir: Path | None = None,
        fingerprint: str = "",
        rebuild: bool = False,
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        settings = Settings(anonymized_telemetry=False, allow_reset=False)
        if persist_dir is None:
            self._client: Any = chromadb.EphemeralClient(settings=settings)
        else:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
        self.ids = list(ids)
        self.rebuilt = False
        existing = None
        if not rebuild:
            try:
                existing = self._client.get_collection(collection, embedding_function=None)
            except Exception:  # chromadb raises NotFoundError / ValueError by version
                existing = None
        if (
            existing is not None
            and existing.count() == len(ids)
            and (existing.metadata or {}).get("fingerprint") == fingerprint
        ):
            self._col: Any = existing
            return
        if existing is not None or rebuild:
            with contextlib.suppress(Exception):
                self._client.delete_collection(collection)
        self._col = self._client.create_collection(
            collection,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 200,
                "fingerprint": fingerprint,
            },
        )
        vecs = _normalise(embeddings)
        step = max(1, int(self._client.get_max_batch_size()))
        for i in range(0, len(ids), step):
            self._col.add(ids=self.ids[i : i + step], embeddings=vecs[i : i + step].tolist())
        self.rebuilt = True

    def __len__(self) -> int:
        return int(self._col.count())

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        n = len(self.ids)
        if n == 0 or k <= 0:
            return []
        q = _normalise(np.asarray(query, dtype=np.float32).reshape(-1))
        res = self._col.query(
            query_embeddings=[q.tolist()], n_results=min(k, n), include=["distances"]
        )
        ids = res["ids"][0]
        dists = res["distances"][0]
        return [(str(i), 1.0 - float(d)) for i, d in zip(ids, dists, strict=True)]

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
