"""Cross-encoder reranking (default ``cross-encoder/ms-marco-MiniLM-L-6-v2`` on CPU), with an
optional persistent score cache.

Score cache: ``<cache_dir>/<model_slug>.sqlite``, one row per (model, sha256(query), chunk_id,
sha256(chunk text)) holding the score and the per-pair latency measured when the pair was first
scored (batch time / pairs in the batch). On a cache hit the stored latency is reported instead
of the (near-zero) lookup time, so latency results do not depend on whether the cache was warm.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ragsys.retrieval.embed import model_slug, quiet_transformers

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class PairScorer(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Any: ...


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RerankCache:
    """Persistent (query, chunk) -> (score, per-pair latency) store for one cross-encoder."""

    def __init__(self, path: Path, model: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.model = model
        self.hits = 0
        self.misses = 0
        self._con: sqlite3.Connection | None = sqlite3.connect(str(path))
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS scores ("
            " model TEXT NOT NULL, query_sha256 TEXT NOT NULL, chunk_id TEXT NOT NULL,"
            " text_sha256 TEXT NOT NULL, score REAL NOT NULL, pair_ms REAL NOT NULL,"
            " created_utc TEXT NOT NULL,"
            " PRIMARY KEY (model, query_sha256, chunk_id, text_sha256))"
        )
        self._con.commit()

    @classmethod
    def for_model(cls, cache_dir: Path, model: str) -> RerankCache:
        return cls(cache_dir / f"{model_slug(model)}.sqlite", model)

    def _db(self) -> sqlite3.Connection:
        if self._con is None:
            raise RuntimeError(f"rerank cache {self.path} is closed")
        return self._con

    def get_many(
        self, query_sha: str, keys: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], tuple[float, float]]:
        """Stored (score, pair_ms) for those (chunk_id, text_sha256) keys that are cached."""
        if not keys:
            return {}
        ids = sorted({k[0] for k in keys})
        marks = ",".join("?" * len(ids))
        rows = self._db().execute(
            "SELECT chunk_id, text_sha256, score, pair_ms FROM scores "
            f"WHERE model = ? AND query_sha256 = ? AND chunk_id IN ({marks})",
            [self.model, query_sha, *ids],
        )
        wanted = set(keys)
        found = {(c, t): (float(s), float(ms)) for c, t, s, ms in rows if (c, t) in wanted}
        self.hits += sum(1 for k in keys if k in found)
        self.misses += sum(1 for k in keys if k not in found)
        return found

    def put_many(self, query_sha: str, rows: Sequence[tuple[str, str, float, float]]) -> None:
        """Store (chunk_id, text_sha256, score, pair_ms) rows."""
        now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        db = self._db()
        db.executemany(
            "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(self.model, query_sha, c, t, s, ms, now) for c, t, s, ms in rows],
        )
        db.commit()

    def __len__(self) -> int:
        row = self._db().execute("SELECT COUNT(*) FROM scores WHERE model = ?", [self.model])
        return int(row.fetchone()[0])

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None


class Reranker:
    """Scores (query, passage) pairs; higher is more relevant.

    ``model`` may be any object with ``predict(pairs)`` (a sentence-transformers
    ``CrossEncoder``, or a fake in tests); when omitted the model is loaded lazily. With a
    ``cache``, :meth:`score_items` reuses stored scores (see the module docstring).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        model: PairScorer | None = None,
        cache: RerankCache | None = None,
    ) -> None:
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = model
        self.cache = cache
        self._warmed = False

    def _load(self) -> Any:
        if self._model is None:
            self._model = _STCrossEncoder(self.name, self.device, self.batch_size)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        raw = self._load().predict([(query, t) for t in texts])
        return [float(x) for x in np.asarray(raw, dtype=np.float64).reshape(-1)]

    def score_items(
        self, query: str, items: Sequence[tuple[str, str]]
    ) -> tuple[list[float], float | None]:
        """Scores of ``(chunk_id, text)`` items and, when a cache is attached, the latency in ms
        to report for them: freshly measured for pairs scored now, the originally measured
        per-pair latency for cached pairs. Without a cache the latency is None (callers time the
        call themselves, as for :meth:`score`)."""
        if self.cache is None:
            return self.score(query, [t for _c, t in items]), None
        if not items:
            return [], 0.0
        qsha = sha256_text(query)
        keys = [(cid, sha256_text(text)) for cid, text in items]
        found = self.cache.get_many(qsha, keys)
        scores = [0.0] * len(items)
        reported = 0.0
        misses: list[int] = []
        for i, k in enumerate(keys):
            if k in found:
                scores[i], pair_ms = found[k]
                reported += pair_ms
            else:
                misses.append(i)
        if misses:
            model = self._load()
            if not self._warmed:  # the first call after loading is slow; keep it untimed
                model.predict([(query, items[misses[0]][1])])
                self._warmed = True
            t0 = time.perf_counter()
            raw = model.predict([(query, items[i][1]) for i in misses])
            elapsed = (time.perf_counter() - t0) * 1000.0
            vals = [float(x) for x in np.asarray(raw, dtype=np.float64).reshape(-1)]
            per_pair = elapsed / len(misses)
            rows = []
            for i, v in zip(misses, vals, strict=True):
                scores[i] = v
                rows.append((keys[i][0], keys[i][1], v, per_pair))
            self.cache.put_many(qsha, rows)
            reported += elapsed
        return scores, reported

    def rerank(
        self, query: str, items: Sequence[tuple[str, str]], top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """Rerank ``(id, text)`` items; returns ``(id, score)`` best first (stable on ties)."""
        scores = self.score(query, [t for _i, t in items])
        order = sorted(range(len(items)), key=lambda i: (-scores[i], i))
        out = [(items[i][0], scores[i]) for i in order]
        return out[:top_k] if top_k is not None else out

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()


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
