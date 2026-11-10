"""Fusion arithmetic on hand-computed examples and the reranker interface."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from ragsys.retrieval.fakes import OverlapCrossEncoder
from ragsys.retrieval.fusion import hybrid_linear, linear, minmax, rrf, single
from ragsys.retrieval.rerank import Reranker

BM25 = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
DENSE = [("b", 0.9), ("d", 0.8), ("a", 0.1)]


def test_rrf_scores_and_components() -> None:
    fused = rrf({"bm25": BM25, "dense": DENSE}, k=60)
    assert [h.chunk_id for h in fused] == ["b", "a", "d", "c"]
    by = {h.chunk_id: h for h in fused}
    assert by["a"].score == pytest.approx(1 / 61 + 1 / 63)
    assert by["b"].score == pytest.approx(1 / 62 + 1 / 61)
    assert by["c"].score == pytest.approx(1 / 63)
    assert by["d"].score == pytest.approx(1 / 62)
    assert (by["a"].components["bm25"].rank, by["a"].components["bm25"].score) == (1, 10.0)
    assert by["a"].components["dense"].rank == 3
    assert "dense" not in by["c"].components


def test_rrf_tie_breaks_on_best_rank_then_first_seen() -> None:
    fused = rrf({"x": [("p", 1.0), ("q", 0.5)], "y": [("q", 1.0), ("p", 0.5)]}, k=60)
    assert [h.chunk_id for h in fused] == ["p", "q"]  # equal scores, equal best rank, p first
    assert fused[0].score == pytest.approx(fused[1].score)


def test_linear_fusion_minmax() -> None:
    fused = hybrid_linear(BM25, DENSE, alpha=0.5)
    by = {h.chunk_id: h.score for h in fused}
    assert by["a"] == pytest.approx(0.5)
    assert by["b"] == pytest.approx(0.5 + 0.5 * 4 / 9)
    assert by["c"] == pytest.approx(0.0)
    assert by["d"] == pytest.approx(0.5 * 0.7 / 0.8)
    assert [h.chunk_id for h in fused] == ["b", "a", "d", "c"]
    dense_only = hybrid_linear(BM25, DENSE, alpha=1.0)
    assert dense_only[0].chunk_id == "b" and {h.chunk_id: h.score for h in dense_only}["a"] == 0
    assert minmax([2.0, 2.0]) == [1.0, 1.0] and minmax([]) == []
    assert linear({"x": [("p", 3.0)]}, {"x": 2.0})[0].score == pytest.approx(2.0)


def test_single_keeps_component() -> None:
    hits = single("bm25", BM25)
    assert [h.chunk_id for h in hits] == ["a", "b", "c"]
    assert hits[1].components["bm25"].rank == 2 and hits[1].score == 5.0


class RecordingScorer:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        self.pairs.extend(pairs)
        return np.array([float(len(p)) for _q, p in pairs])


def test_reranker_interface_with_fake_cross_encoder() -> None:
    rr = Reranker("fake:overlap", model=OverlapCrossEncoder())
    out = rr.rerank(
        "consumer complaint",
        [("x", "nothing relevant"), ("y", "a consumer complaint is filed"), ("z", "complaint")],
    )
    assert [i for i, _s in out] == ["y", "z", "x"]
    assert out[0][1] == pytest.approx(2 / np.sqrt(2))
    assert rr.rerank("q", [("x", "a"), ("y", "b")], top_k=1) == [("x", 0.0)]  # stable ties
    rec = RecordingScorer()
    scores = Reranker("fake", model=rec).score("query", ["aa", "b"])
    assert scores == [2.0, 1.0] and rec.pairs == [("query", "aa"), ("query", "b")]
    assert Reranker("fake", model=rec).score("q", []) == []
