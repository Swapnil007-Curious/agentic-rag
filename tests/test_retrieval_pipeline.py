"""RetrievalConfig ids and the Retriever end to end on the synthetic corpus (fake models)."""

from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from ragsys.config import REPO_ROOT
from ragsys.retrieval.config import RetrievalConfig, load_retrieval_settings, parse_config_id
from ragsys.retrieval.pipeline import RetrievalResources, Retriever
from retrieval_helpers import fake_settings


def test_config_ids_are_short_deterministic_and_parseable() -> None:
    assert RetrievalConfig().config_id() == "structural|hybrid_rrf|bge-base"
    cfg = RetrievalConfig(xref_expand=True, rerank=True)
    assert cfg.config_id() == "structural|hybrid_rrf|bge-base|xref|rerank-minilm"
    assert RetrievalConfig(chunker="fixed_512", retriever="bm25").config_id() == "fixed_512|bm25"
    lin = RetrievalConfig(retriever="hybrid_linear", alpha=0.5)
    assert lin.config_id() == "structural|hybrid_linear|bge-base|a0.5"
    # parameters that are not spelled out produce a stable hash suffix
    odd = RetrievalConfig(first_stage_k=30)
    assert odd.config_id().startswith("structural|hybrid_rrf|bge-base|cfg-")
    assert odd.config_id() == RetrievalConfig(first_stage_k=30).config_id()
    # irrelevant parameters (alpha for BM25) do not change the id
    assert RetrievalConfig(retriever="bm25", alpha=0.9).config_id() == "structural|bm25"
    assert cfg.dir_name == "structural__hybrid_rrf__bge-base__xref__rerank-minilm"
    for c in (cfg, lin, RetrievalConfig(chunker="semantic", retriever="dense")):
        assert parse_config_id(c.config_id()).config_id() == c.config_id()
    assert parse_config_id(cfg.dir_name).config_id() == cfg.config_id()
    with pytest.raises(ValueError):
        parse_config_id(odd.config_id())
    with pytest.raises(ValueError):
        parse_config_id("structural|nope")
    with pytest.raises(pydantic.ValidationError):
        RetrievalConfig.model_validate({"chunker": "x", "unknown": 1})


def test_repository_grid_has_expected_configs() -> None:
    s = load_retrieval_settings(REPO_ROOT / "configs" / "retrieval.yaml")
    ids = [c.config_id() for c in s.grid_configs()]
    assert len(ids) == len(set(ids)) == 4 * 6 * 2 * 2 + 4 * 2
    for b in s.selection.baselines.values():
        assert b in ids
    assert "structural|hybrid_rrf|bge-large|rerank-minilm" in ids
    assert "fixed_256|hybrid_linear|bge-base|a0.5|xref|rerank-minilm" in ids


@pytest.fixture(scope="module")
def resources(retrieval_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> RetrievalResources:
    tmp = tmp_path_factory.mktemp("res")
    s = fake_settings(retrieval_dir, tmp / "results", tmp / "q.jsonl")
    return RetrievalResources(s, emb_dir=tmp / "emb")


def test_bm25_and_dense_first_stage(resources: RetrievalResources) -> None:
    r = Retriever(RetrievalConfig(chunker="structural", retriever="bm25"), resources)
    res = r.search("appeal within thirty days")
    assert res.results and set(res.timings_ms) == {"bm25", "fusion", "total"}
    top = res.results[0]
    assert "testact:s4" in top.chunk.section_ids
    assert top.provenance == "first_stage" and top.components["bm25"].rank == 1
    assert res.top1_first_stage_score == top.first_stage_score
    d = Retriever(
        RetrievalConfig(chunker="fixed_64", retriever="dense", dense_model="fake-a"), resources
    ).search("appeal within thirty days")
    assert {"dense_encode", "dense_search"} <= set(d.timings_ms)
    assert [x.rank for x in d.ranked] == list(range(1, len(d.ranked) + 1))
    assert len(d.results) == 10 and len(d.ranked) >= len(d.results)


def test_hybrid_xref_rerank_provenance(resources: RetrievalResources) -> None:
    cfg = RetrievalConfig(
        chunker="structural",
        retriever="hybrid_rrf",
        dense_model="fake-a",
        xref_expand=True,
        rerank=True,
        rerank_model="fake",
        final_k=5,
    )
    r = Retriever(cfg, resources)
    res = r.search("complaint under the rules in writing")
    # final_k=5 is not spelled out in the id, so a parameter hash is appended
    assert res.config_id.startswith("structural|hybrid_rrf|fake-a|xref|rerank-fake|cfg-")
    assert {"bm25", "dense_encode", "fusion", "xref", "rerank", "total"} <= set(res.timings_ms)
    assert len(res.results) == 5
    head = res.ranked[: len([x for x in res.ranked if x.rerank_score is not None])]
    scores = [x.rerank_score for x in head]
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]
    for x in res.ranked:
        if x.provenance != "first_stage":
            assert x.provenance.startswith("xref_from:") and x.xref_target
    one = res.ranked[0]
    assert set(one.components) <= {"bm25", "dense"} and one.to_dict()["chunk_id"] == one.chunk_id


def test_expanded_chunks_follow_their_source_without_rerank(
    resources: RetrievalResources,
) -> None:
    cfg = RetrievalConfig(
        chunker="fixed_64", retriever="bm25", xref_expand=True, xref_top_m=10, xref_max_added=10
    )
    found = False
    for q in ("complaint under sub-section (1) of section 3 of the Act", "appeal section 3"):
        res = Retriever(cfg, resources).search(q)
        ids = [x.chunk_id for x in res.ranked]
        for i, x in enumerate(res.ranked):
            if x.provenance.startswith("xref_from:"):
                src = x.provenance.split(":", 1)[1]
                j = ids.index(src)
                assert j < i and all(
                    y.provenance == x.provenance or y.provenance.startswith("xref_from:")
                    for y in res.ranked[j + 1 : i]
                )
                found = True
    assert found
