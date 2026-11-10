"""Retrieval metrics (per question; averaged over answerable questions by the harness).

Metric functions take a *relevance profile*: for each ranked position, the set of gold nodes
that chunk is relevant to (see :mod:`ragsys.eval.relevance`). This keeps them pure and easy to
check by hand.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Profile = Sequence[set[str]]


def recall_at_k(profile: Profile, golds: Sequence[str], k: int) -> float:
    """Fraction of gold nodes covered by at least one of the top-k chunks."""
    if not golds:
        return 0.0
    covered: set[str] = set().union(*profile[:k]) if profile[:k] else set()
    return len(covered & set(golds)) / len(set(golds))


def hit_at_k(profile: Profile, k: int) -> float:
    return 1.0 if any(profile[:k]) else 0.0


def mrr_at_k(profile: Profile, k: int = 10) -> float:
    for i, rel in enumerate(profile[:k]):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(profile: Profile, k: int, n_relevant_total: int) -> float:
    """Binary gain per chunk (relevant to any gold); ideal DCG uses min(R, k) relevant chunks,
    R = number of relevant chunks in the corpus."""
    dcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(profile[:k]) if rel)
    ideal_n = min(max(n_relevant_total, sum(1 for rel in profile[:k] if rel)), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def budget_prefix(n_tokens: Sequence[int], budget: int) -> int:
    """Number of top results that fit in ``budget`` tokens (stops at the first that does not)."""
    used = 0
    for i, n in enumerate(n_tokens):
        if used + n > budget:
            return i
        used += n
    return len(n_tokens)


def budget_coverage(
    profile: Profile, n_tokens: Sequence[int], golds: Sequence[str], budget: int
) -> float:
    """Fraction of gold nodes covered by the top results that fit in a context budget."""
    return recall_at_k(profile, golds, budget_prefix(n_tokens, budget))


def percentile(values: Sequence[float], p: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p)) if values else None


def mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """P(score of a positive > score of a negative), ties counted 1/2 (Mann-Whitney U)."""
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(positives) * len(negatives))


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 1000, seed: int = 42
) -> dict[str, float | int | None]:
    """Mean of ``a - b`` over paired questions with a percentile 95% bootstrap interval."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    n = len(a)
    if n == 0:
        return {"n": 0, "mean_diff": None, "ci_low": None, "ci_high": None}
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diffs[idx].mean(axis=1)
    return {
        "n": n,
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }
