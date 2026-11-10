"""Rank fusion of several first-stage result lists.

Every fused hit keeps the rank (1-based) and raw score it had in each component list, so a
result can be inspected afterwards (``search.py`` prints them).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

Ranking = Sequence[tuple[str, float]]  # (chunk_id, score), best first


@dataclass(frozen=True)
class ComponentHit:
    rank: int  # 1-based rank in that component's list
    score: float


@dataclass
class FusedHit:
    chunk_id: str
    score: float
    components: dict[str, ComponentHit] = field(default_factory=dict)


def _collect(
    rankings: Mapping[str, Ranking],
) -> tuple[dict[str, dict[str, ComponentHit]], list[str]]:
    comps: dict[str, dict[str, ComponentHit]] = {}
    first_seen: list[str] = []
    for name, ranking in rankings.items():
        for rank, (cid, score) in enumerate(ranking, start=1):
            if cid not in comps:
                comps[cid] = {}
                first_seen.append(cid)
            comps[cid].setdefault(name, ComponentHit(rank, float(score)))
    return comps, first_seen


def _sorted(
    scores: dict[str, float], comps: dict[str, dict[str, ComponentHit]], order: list[str]
) -> list[FusedHit]:
    pos = {cid: i for i, cid in enumerate(order)}
    ids = sorted(
        scores,
        key=lambda c: (-scores[c], min(h.rank for h in comps[c].values()), pos[c]),
    )
    return [FusedHit(c, scores[c], dict(comps[c])) for c in ids]


def single(name: str, ranking: Ranking) -> list[FusedHit]:
    """Wrap one component list in the fused format (score = the component's score)."""
    return [
        FusedHit(cid, float(s), {name: ComponentHit(r, float(s))})
        for r, (cid, s) in enumerate(ranking, start=1)
    ]


def rrf(rankings: Mapping[str, Ranking], k: int = 60) -> list[FusedHit]:
    """Reciprocal Rank Fusion: ``score(d) = sum_c 1 / (k + rank_c(d))``.

    Ties are broken by the best component rank, then by first appearance.
    """
    comps, order = _collect(rankings)
    scores = {cid: sum(1.0 / (k + h.rank) for h in hits.values()) for cid, hits in comps.items()}
    return _sorted(scores, comps, order)


def minmax(values: Sequence[float]) -> list[float]:
    """Min-max normalise to [0, 1]; a constant list maps to 1.0."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def linear(rankings: Mapping[str, Ranking], weights: Mapping[str, float]) -> list[FusedHit]:
    """Weighted sum of min-max-normalised component scores (missing from a list -> 0)."""
    comps, order = _collect(rankings)
    norm: dict[str, dict[str, float]] = {}
    for name, ranking in rankings.items():
        vals = minmax([float(s) for _cid, s in ranking])
        norm[name] = {}
        for (cid, _s), v in zip(ranking, vals, strict=True):
            norm[name].setdefault(cid, v)
    scores = {
        cid: sum(weights.get(name, 0.0) * norm[name].get(cid, 0.0) for name in rankings)
        for cid in comps
    }
    return _sorted(scores, comps, order)


def hybrid_linear(bm25: Ranking, dense: Ranking, alpha: float) -> list[FusedHit]:
    """``alpha * dense_norm + (1 - alpha) * bm25_norm``."""
    return linear({"bm25": bm25, "dense": dense}, {"dense": alpha, "bm25": 1.0 - alpha})
