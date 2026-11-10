"""Deterministic, dependency-free stand-ins for the neural models.

They let the whole retrieval stack (and the scripts) run in tests and CI without downloading
any model. Select them with model names starting with ``fake:``:

* ``fake:hash-<dim>``: hashed bag-of-words encoder (e.g. ``fake:hash-64``);
* ``fake:overlap``: cross-encoder that scores query/passage word overlap.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class HashingEncoder:
    """Signed feature hashing of lower-cased words, L2-normalised."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.name = f"fake:hash-{dim}"
        self.calls = 0  # number of encode() calls (used by cache tests)

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in _words(text):
            h = int.from_bytes(hashlib.md5(w.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        self.calls += 1
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)


class OverlapCrossEncoder:
    """Scores a (query, passage) pair by the share of query words found in the passage."""

    name = "fake:overlap"

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        out = []
        for q, p in pairs:
            qw = set(_words(q))
            pw = set(_words(p))
            out.append(len(qw & pw) / math.sqrt(len(qw) or 1))
        return np.asarray(out, dtype=np.float32)


def fake_encoder(name: str) -> HashingEncoder:
    m = re.fullmatch(r"fake:hash-(\d+)", name)
    if not m:
        raise ValueError(f"unknown fake encoder {name!r} (use fake:hash-<dim>)")
    return HashingEncoder(int(m.group(1)))
