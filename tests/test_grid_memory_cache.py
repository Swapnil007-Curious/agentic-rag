"""Memory-safe grid execution and caches: configs grouped by dense model with one encoder alive
at a time, the rerank score cache (identical scores, stored latencies reported on hits) and the
query-embedding cache; grid results are identical with and without the cache."""

from __future__ import annotations

import gc
import time
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragsys.eval import harness
from ragsys.eval.harness import Evaluator, execution_groups, run_configs
from ragsys.eval.validate import load_questions
from ragsys.retrieval import pipeline
from ragsys.retrieval.config import RetrievalConfig, RetrievalSettings
from ragsys.retrieval.embed import SentenceTransformerEncoder
from ragsys.retrieval.fakes import HashingEncoder, OverlapCrossEncoder
from ragsys.retrieval.pipeline import RetrievalResources, Retriever
from ragsys.retrieval.rerank import RerankCache, Reranker
from retrieval_helpers import fake_settings, write_questions


@pytest.fixture()
def settings(retrieval_dir: Path, tmp_path: Path) -> RetrievalSettings:
    q = tmp_path / "questions.jsonl"
    write_questions(q)
    s = fake_settings(retrieval_dir, tmp_path / "results", q)
    s.paths.emb_dir = tmp_path / "emb"
    s.dense_models.update({"enc-a": "test/enc-a", "enc-b": "test/enc-b"})
    return s


def cfg(chunker: str, retriever: str, model: str = "fake-a", **kw: Any) -> RetrievalConfig:
    return RetrievalConfig(
        chunker=chunker, retriever=retriever, dense_model=model, rerank_model="fake", **kw
    )


def test_execution_groups_put_bm25_first_then_one_group_per_model(
    settings: RetrievalSettings,
) -> None:
    configs = [
        cfg("structural", "dense", "enc-a"),
        cfg("structural", "bm25"),
        cfg("fixed_64", "hybrid_rrf", "enc-b"),
        cfg("fixed_64", "dense", "enc-a", rerank=True),
        cfg("fixed_64", "bm25", rerank=True),
    ]
    assert execution_groups(configs, settings) == [
        (None, [1, 4]),
        ("test/enc-a", [0, 3]),
        ("test/enc-b", [2]),
    ]


def test_grid_keeps_at_most_one_encoder_alive(
    settings: RetrievalSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live: weakref.WeakSet[Any] = weakref.WeakSet()
    events: list[str] = []
    peak = [0]

    class FakeSTModel:
        def __init__(self, name: str) -> None:
            gc.collect()
            live.add(self)
            peak[0] = max(peak[0], len(live))
            events.append(f"load {name}")
            self._enc = HashingEncoder(32)

        def encode(self, texts: Sequence[str], **_kw: Any) -> np.ndarray:
            return self._enc.encode(list(texts), is_query=False)

        def get_sentence_embedding_dimension(self) -> int:
            return 32

    class TrackedEncoder(SentenceTransformerEncoder):
        def _load(self) -> Any:
            if self._model is None:
                self._model = FakeSTModel(self.name)
            return self._model

    monkeypatch.setattr(pipeline, "SentenceTransformerEncoder", TrackedEncoder)
    configs = [
        cfg("structural", "dense", "enc-a"),
        cfg("structural", "bm25"),
        cfg("structural", "hybrid_rrf", "enc-b", rerank=True),
        cfg("fixed_64", "dense", "enc-a", xref_expand=True),
        cfg("fixed_64", "bm25", rerank=True),
        cfg("fixed_64", "dense", "enc-b"),
    ]
    questions, _ = load_questions(settings.paths.questions)
    res = RetrievalResources(settings)
    summaries, records = run_configs(
        Evaluator(settings, res), configs, questions, "dev", tmp_path / "out", log=events.append
    )
    assert peak[0] == 1
    loads = [e for e in events if e.startswith("load ")]
    assert loads == ["load test/enc-a", "load test/enc-b"]  # each model loaded once
    first_load = events.index("load test/enc-a")
    bm25_lines = [i for i, e in enumerate(events) if "|bm25" in e and e.startswith("[")]
    assert len(bm25_lines) == 2 and max(bm25_lines) < first_load
    assert any("released test/enc-a" in e for e in events) and res.loaded_encoders() == []
    assert all("RSS" in e for e in events if e.startswith("=="))
    assert all("ETA" in e for e in events if e.startswith("[") and "skip" not in e)
    # results come back in the order of the configs, whatever the execution order
    assert [s["config_id"] for s in summaries] == [c.config_id() for c in configs]
    assert list(records) == [c.config_id() for c in configs]


class CountingScorer:
    def __init__(self) -> None:
        self.calls = 0
        self.inner = OverlapCrossEncoder()

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Any:
        self.calls += 1
        time.sleep(0.001 * len(pairs))  # a measurable latency
        return self.inner.predict(pairs)


def test_rerank_cache_hits_give_identical_scores_and_reported_latency(tmp_path: Path) -> None:
    query = "who can file a complaint about goods"
    items = [(f"c{i}", f"a complaint about goods number {i} may be filed") for i in range(6)]
    scorer = CountingScorer()
    r = Reranker(
        "fake:counting", model=scorer, cache=RerankCache.for_model(tmp_path, "fake:counting")
    )
    s1, ms1 = r.score_items(query, items)
    assert scorer.calls == 2  # untimed warm-up + one batch for the misses
    assert ms1 is not None and ms1 > 0
    s2, ms2 = r.score_items(query, items)
    assert scorer.calls == 2  # all hits: the model is not called
    assert s2 == s1 and ms2 == pytest.approx(ms1)
    plain = Reranker("x", model=OverlapCrossEncoder()).score(query, [t for _c, t in items])
    assert s1 == plain  # cached scores equal uncached ones
    # persisted: a new process (new objects on the same file) hits too
    scorer2 = CountingScorer()
    r2 = Reranker(
        "fake:counting", model=scorer2, cache=RerankCache.for_model(tmp_path, "fake:counting")
    )
    s3, ms3 = r2.score_items(query, items)
    assert scorer2.calls == 0 and s3 == s1 and ms3 == pytest.approx(ms1)
    # partial hit: only the new pair is scored; changed chunk text is a miss
    more = [*items, ("c9", "an appeal lies within thirty days")]
    changed = [("c0", "completely different text now"), *items[1:]]
    _s, ms4 = r2.score_items(query, more)
    assert scorer2.calls == 2 and ms4 is not None and ms4 > ms1
    r2.score_items(query, changed)
    assert scorer2.calls == 3 and r2.cache is not None and len(r2.cache) == 8
    # without a cache the caller measures the latency itself
    scores, none = Reranker("x", model=OverlapCrossEncoder()).score_items(query, items)
    assert scores == s1 and none is None
    r.close()
    r2.close()


def test_query_embedding_cache_reports_stored_latency(settings: RetrievalSettings) -> None:
    res = RetrievalResources(settings)
    c = cfg("structural", "dense")
    retriever = Retriever(c, res)
    enc = retriever.encoder
    assert isinstance(enc, HashingEncoder)
    first = retriever.search("Who can file a complaint?")
    calls = enc.calls
    second = Retriever(c, res).search("Who can file a complaint?")  # another config, same run
    assert enc.calls == calls  # served from the in-memory cache
    assert second.timings_ms["dense_encode"] == first.timings_ms["dense_encode"]
    assert second.timings_ms["total"] >= second.timings_ms["dense_encode"]
    assert [r.chunk_id for r in second.ranked] == [r.chunk_id for r in first.ranked]


def _strip(records: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        cid: [{k: v for k, v in r.items() if k != "timings_ms"} for r in recs]
        for cid, recs in records.items()
    }


def test_grid_results_identical_with_and_without_rerank_cache(
    settings: RetrievalSettings, tmp_path: Path
) -> None:
    grid = settings.grid_configs()
    configs = [*[c for c in grid if c.rerank][:6], grid[0]]
    questions, _ = load_questions(settings.paths.questions)
    runs = {}
    for name, cache in (("off", False), ("cold", True), ("warm", True)):
        res = RetrievalResources(settings, rerank_cache=cache, rerank_cache_dir=tmp_path / "rc")
        _s, records = run_configs(
            Evaluator(settings, res), configs, questions, "dev", tmp_path / name, overwrite=True,
            log=lambda _m: None,
        )  # fmt: skip
        runs[name] = records
        res.close()
    assert _strip(runs["off"]) == _strip(runs["cold"]) == _strip(runs["warm"])
    for cid, recs in runs["cold"].items():
        for cold, warm in zip(recs, runs["warm"][cid], strict=True):
            if "rerank" in cold["timings_ms"]:
                assert warm["timings_ms"]["rerank"] == pytest.approx(
                    cold["timings_ms"]["rerank"], abs=2e-3
                )
                assert warm["timings_ms"]["total"] >= warm["timings_ms"]["rerank"]
    assert (tmp_path / "rc" / "fake_overlap.sqlite").exists()
    assert harness.questions_fingerprint(questions)  # sanity: fixture questions loaded
