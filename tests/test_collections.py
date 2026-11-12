"""Collections: the path resolver, backward compatibility of consumer_law (paths, prompts,
fingerprints, untouched result tables), per-collection isolation of every derived location,
domain profiles and the "selected:" retrieval of a system."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from ragsys.agents.config import SelectionMissing, load_generation_settings
from ragsys.collections import (
    Collection,
    add_collection_arguments,
    all_collections,
    collection_from_args,
    consistency_errors,
    get_collection,
)
from ragsys.config import REPO_ROOT
from ragsys.eval.generation import prompt_digest
from ragsys.profiles import CONSUMER_LAW, CS_AI, get_profile
from snapshot_helpers import consumer_law_snapshot

DATA = Path(__file__).parent / "data"

# The locations every consumer_law command used before collections existed.
CONSUMER_LAW_PATHS = {
    "ingest_config": "configs/ingest.yaml",
    "chunking_config": "configs/chunking.yaml",
    "retrieval_config": "configs/retrieval.yaml",
    "generation_config": "configs/generation.yaml",
    "llm_config": "configs/llm.yaml",
    "raw_dir": "data/raw",
    "processed_dir": "data/processed",
    "eval_dir": "data/eval",
    "results_dir": "results",
    "sources_csv": "data/raw/SOURCES.csv",
    "clean_dir": "data/processed/clean",
    "structure_dir": "data/processed/structure",
    "nodes_file": "data/processed/structure/nodes.jsonl",
    "chunks_dir": "data/processed/chunks",
    "emb_dir": "data/processed/emb",
    "chroma_dir": "data/processed/chroma",
    "semantic_cache_dir": "data/processed/cache",
    "llm_cache_dir": "data/processed/llm_cache",
    "rerank_cache_dir": "data/processed/rerank_cache",
    "nli_cache_dir": "data/processed/nli_cache",
    "questions_file": "data/eval/questions.jsonl",
    "drafts_file": "data/eval/drafts.jsonl",
    "tables_dir": "results/tables",
    "retrieval_results_dir": "results/retrieval",
    "generation_results_dir": "results/generation",
    "test_access_log": "results/test_access_log.json",
    "threshold_file": "results/tables/abstention_threshold.json",
    "selected_configs_file": "results/tables/selected_configs.json",
    "eval_construction_file": "results/tables/eval_construction.json",
    "feedback_file": "data/feedback.jsonl",
}


def test_consumer_law_paths_are_unchanged() -> None:
    paths = get_collection("consumer_law").paths()
    assert {k: paths[k] for k in CONSUMER_LAW_PATHS} == CONSUMER_LAW_PATHS
    assert get_collection(None).name == "consumer_law"  # the default collection


def test_default_collection_from_cli() -> None:
    ap = argparse.ArgumentParser()
    add_collection_arguments(ap)
    assert collection_from_args(ap.parse_args([])).name == "consumer_law"
    assert collection_from_args(ap.parse_args(["--collection", "cs_ai"])).name == "cs_ai"
    with pytest.raises(SystemExit):
        collection_from_args(ap.parse_args(["--collection", "nope"]))


@pytest.mark.parametrize("coll", all_collections(), ids=lambda c: c.name)
def test_config_files_agree_with_collection_roots(coll: Collection) -> None:
    assert consistency_errors(coll) == []


def test_collections_share_no_output_or_cache() -> None:
    colls = all_collections()
    assert {c.name for c in colls} >= {"consumer_law", "cs_ai"}
    shared_ok = {"llm_config"}  # the LLM endpoint config is shared; its cache is not
    seen: dict[str, str] = {}
    for c in colls:
        for key, path in c.paths().items():
            if key in shared_ok:
                continue
            assert path not in seen, f"{c.name}.{key} == {seen[path]}"
            seen[path] = f"{c.name}.{key}"
    cs = get_collection("cs_ai").paths()
    assert all(
        v.startswith(("collections/cs_ai/", "configs/collections/cs_ai/"))
        for k, v in cs.items()
        if k not in shared_ok
    )


def test_consumer_law_tables_are_untouched() -> None:
    """The files results/tables and the test-access log held before collections were introduced
    are still there with the same bytes (line endings normalised). Since the generator-scale
    ablation, the generation_test tables may only gain appended rows and the log appended
    entries; tests/test_frozen_results.py checks every frozen file of both evaluated collections
    (and that no other file appears in their tables directories)."""
    expected = json.loads((DATA / "consumer_law_tables.sha256.json").read_text(encoding="utf-8"))
    frozen = json.loads((DATA / "frozen_results.json").read_text(encoding="utf-8"))["files"]
    for rel, digest in expected.items():
        mode = frozen[rel]["mode"]
        assert mode in ("exact", "prefix", "json_prefix"), rel
        data = (REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        if mode == "exact":
            assert frozen[rel]["sha256"] == digest == hashlib.sha256(data).hexdigest(), rel
        elif mode == "prefix":  # the same pin, extended: existing bytes are a fixed prefix
            n = frozen[rel]["bytes"]
            assert frozen[rel]["sha256"] == digest == hashlib.sha256(data[:n]).hexdigest(), rel
        else:  # the test-access log: the entries logged so far are unchanged
            entries = json.loads(data.decode("utf-8"))
            assert len(entries) >= frozen[rel]["entries"], rel


def test_consumer_law_prompts_and_fingerprints_are_unchanged() -> None:
    expected = json.loads((DATA / "consumer_law_prompt_snapshot.json").read_text(encoding="utf-8"))
    assert consumer_law_snapshot() == expected


def test_profiles() -> None:
    assert get_profile(None) is CONSUMER_LAW and get_profile("cs_ai") is CS_AI
    with pytest.raises(KeyError):
        get_profile("nope")
    assert sum(CS_AI.type_mix.values()) == 60 and "cross_reference" not in CS_AI.type_mix
    assert CS_AI.type_mix == {
        "definition": 15,
        "factual": 15,
        "procedural": 12,
        "numeric": 8,
        "multi_section": 10,
    }
    assert CS_AI.targets == (60, 20) and CONSUMER_LAW.targets == (90, 30)
    assert "legal" not in CS_AI.generation.synth_system.lower()
    assert prompt_digest(CS_AI.generation) != prompt_digest(CONSUMER_LAW.generation)
    assert any("pretraining" in n for n in CS_AI.notes)
    topics = " ".join(d for _i, d, _s in CS_AI.oos_topics).lower()
    assert all(t in topics for t in ("kubernetes", "quantum", "reinforcement learning"))


def test_selected_retrieval_resolution(tmp_path: Path) -> None:
    gen = load_generation_settings(REPO_ROOT / "configs/collections/cs_ai/generation.yaml")
    gen.results_dir = tmp_path
    assert gen.retrieval_cfg("G1").config_id() == "fixed_512|dense|bge-base"
    with pytest.raises(SelectionMissing):
        gen.retrieval_cfg("G2")
    tables = tmp_path / "tables"
    tables.mkdir()
    cfg = {
        "chunker": "structural",
        "retriever": "hybrid_rrf",
        "dense_model": "bge-base",
        "rerank": True,
    }
    (tables / "selected_configs.json").write_text(
        json.dumps({"selected": [{"config_id": "x", "roles": ["best_overall"], "config": cfg}]}),
        encoding="utf-8",
    )
    assert gen.retrieval_cfg("G2").config_id() == "structural|hybrid_rrf|bge-base|rerank-minilm"
    assert gen.retrieval_cfg("G3") == gen.retrieval_cfg("G2")  # derived from G2
    assert gen.retrieval_cfg("G4").config_id() == "structural|hybrid_rrf|bge-base|rerank-minilm"
    assert set(gen.skipped_systems) == {"G5", "G6"} and gen.abstention.tuned_system is None
