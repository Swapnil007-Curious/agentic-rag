"""Cross-encoder reranking (default ``cross-encoder/ms-marco-MiniLM-L-6-v2`` on CPU)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from ragsys.retrieval.embed import quiet_transformers

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class PairScorer(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Any: ...


class Reranker:
    """Scores (query, passage) pairs; higher is more relevant.

    ``model`` may be any object with ``predict(pairs)`` (a sentence-transformers
    ``CrossEncoder``, or a fake in tests); when omitted the model is loaded lazily.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        model: PairScorer | None = None,
    ) -> None:
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = model

    def _load(self) -> Any:
        if self._model is None:
            self._model = _STCrossEncoder(self.name, self.device, self.batch_size)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        raw = self._load().predict([(query, t) for t in texts])
        return [float(x) for x in np.asarray(raw, dtype=np.float64).reshape(-1)]

    def rerank(
        self, query: str, items: Sequence[tuple[str, str]], top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """Rerank ``(id, text)`` items; returns ``(id, score)`` best first (stable on ties)."""
        scores = self.score(query, [t for _i, t in items])
        order = sorted(range(len(items)), key=lambda i: (-scores[i], i))
        out = [(items[i][0], scores[i]) for i in order]
        return out[:top_k] if top_k is not None else out


class _STCrossEncoder:
    """sentence-transformers ``CrossEncoder`` behind the :class:`PairScorer` interface.

    Scores are the model's raw outputs (logits for the ms-marco cross-encoders)."""

    def __init__(self, name: str, device: str, batch_size: int) -> None:
        from sentence_transformers import CrossEncoder

        quiet_transformers()
        self.batch_size = batch_size
        self.model = CrossEncoder(name, device=device)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Any:
        return self.model.predict(list(pairs), batch_size=self.batch_size, show_progress_bar=False)
