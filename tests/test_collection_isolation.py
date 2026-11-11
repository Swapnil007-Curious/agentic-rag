"""Two collections built from the synthetic corpus with fake models: the scripts write every
embedding, rerank-score cache, NLI cache, result table and test-access log entry under the
collection's own directories, and the test split of one collection is opened (and logged)
independently of the other."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from agent_helpers import generation_settings_dict
from ragsys.collections import get_collection
from ragsys.eval.split import read_access_log
from retrieval_helpers import fake_settings, write_questions, write_settings

UI = {
    "label": "L",
    "eyebrow": "E",
    "heading": "H",
    "lede": "Lede",
    "placeholder": "P",
    "disclaimer": "D",
}


def _make_collection(root: Path, name: str, corpus: Path) -> dict[str, object]:
    base = root / name
    processed = base / "processed"
    shutil.copytree(corpus, processed)
    (base / "eval").mkdir(parents=True)
    write_questions(base / "eval" / "questions.jsonl")
    rs = fake_settings(processed, base / "results", base / "eval" / "questions.jsonl")
    retrieval_yaml = write_settings(base / "retrieval.yaml", rs)
    gen = generation_settings_dict(base, retrieval_yaml)
    gen["results_dir"] = str(base / "results")
    gen["nli"]["cache_dir"] = str(processed / "nli_cache")
    gen["llm_overrides"]["cache_dir"] = str(processed / "llm_cache")
    (base / "generation.yaml").write_text(yaml.safe_dump(gen), encoding="utf-8")
    return {
        "title": name,
        "configs": {
            "ingest": "configs/ingest.yaml",
            "chunking": "configs/chunking.yaml",
            "retrieval": str(retrieval_yaml),
            "generation": str(base / "generation.yaml"),
        },
        "roots": {
            "raw_dir": str(base / "raw"),
            "processed_dir": str(processed),
            "eval_dir": str(base / "eval"),
            "results_dir": str(base / "results"),
        },
        "ui": UI,
        "pipeline": {"index_models": ["fake-a"], "dev_systems": ["G2"], "test_systems": ["G1"]},
    }


@pytest.fixture()
def registry(retrieval_dir: Path, tmp_path: Path) -> Path:
    reg = {
        "default": "alpha",
        "collections": {
            "alpha": _make_collection(tmp_path, "alpha", retrieval_dir),
            "beta": _make_collection(tmp_path, "beta", retrieval_dir),
        },
    }
    path = tmp_path / "collections.yaml"
    path.write_text(yaml.safe_dump(reg), encoding="utf-8")
    return path


def test_caches_results_and_test_logs_are_per_collection(
    registry: Path, script: Callable[[str], ModuleType], capsys: pytest.CaptureFixture[str]
) -> None:
    alpha = get_collection("alpha", registry)
    beta = get_collection("beta", registry)

    def c(name: str) -> list[str]:
        return ["--collection", name, "--collections-file", str(registry)]

    build, grid, select, ev, gen = (
        script(n)
        for n in (
            "build_index",
            "run_retrieval_grid",
            "select_configs",
            "eval_retrieval",
            "run_generation",
        )
    )
    for name in ("alpha", "beta"):
        assert (
            build.main(
                [
                    "--chunker",
                    "structural",
                    "fixed_64",
                    "--models",
                    "fake-a",
                    "--no-chroma",
                    *c(name),
                ]
            )
            == 0
        )
        assert grid.main(["--split", "dev", "--only", "bm25", *c(name)]) == 0
        assert select.main(c(name)) == 0
    for coll in (alpha, beta):
        assert (coll.emb_dir / "fake_hash-64" / "structural.npy").exists()
        assert (coll.rerank_cache_dir / "fake_overlap.sqlite").exists()
        assert (coll.tables_dir / "retrieval_dev.csv").exists()
        assert (coll.tables_dir / "selected_configs.json").exists()
        assert not coll.test_access_log.exists()

    # opening alpha's test split logs to alpha only
    assert ev.main(["--split", "test", *c("alpha")]) == 0
    assert len(read_access_log(alpha.test_access_log)) == 1
    assert not beta.test_access_log.exists()
    assert ev.main(["--split", "test", *c("alpha")]) == 3  # repeat refused for alpha ...
    assert ev.main(["--split", "test", *c("beta")]) == 0  # ... but beta is independent
    assert len(read_access_log(beta.test_access_log)) == 1
    assert len(read_access_log(alpha.test_access_log)) == 1
    capsys.readouterr()

    # generation dev run (fake LLM, fake NLI): records and the NLI cache stay in alpha
    assert gen.main(["--split", "dev", "--systems", "G2", "--fake-llm", *c("alpha")]) == 0
    assert (alpha.generation_results_dir / "dev" / "G2" / "per_question.jsonl").exists()
    assert (alpha.nli_cache_dir / "nli_scores.sqlite").exists()
    assert not beta.generation_results_dir.exists() and not beta.nli_cache_dir.exists()
    assert (alpha.tables_dir / "generation_dev.csv").exists()
    assert not (beta.tables_dir / "generation_dev.csv").exists()
    # each collection's LLM responses are cached in its own processed folder
    assert alpha.llm().cache_dir != beta.llm().cache_dir
    assert json.loads((alpha.tables_dir / "selected_configs.json").read_text(encoding="utf-8"))


def test_eval_generation_needs_threshold_only_for_threshold_systems(
    registry: Path, script: Callable[[str], ModuleType], capsys: pytest.CaptureFixture[str]
) -> None:
    ev = script("eval_generation")
    c = ["--collections-file", str(registry), "--collection", "alpha"]
    assert ev.main(["--split", "test", "--systems", "G5", "--fake-llm", *c]) == 2
    assert "abstention_threshold.json is missing" in capsys.readouterr().out
    alpha = get_collection("alpha", registry)
    assert not alpha.test_access_log.exists()  # refused before opening the test split
    assert ev.main(["--split", "test", "--systems", "G1", "G2", "--fake-llm", *c]) == 0
    entries = read_access_log(alpha.test_access_log)
    assert len(entries) == 1 and "no abstention threshold" in entries[0]["note"]
