"""Generation scripts end to end on the synthetic corpus with fake models and the offline fake
LLM: dev runs (resumable), threshold tuning (dev only), derived systems, and the test-split guard
of eval_generation.py (refuses without the threshold, logs every access, refuses a repeat)."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from agent_helpers import write_generation_settings
from retrieval_helpers import fake_settings, write_questions, write_settings


@pytest.fixture()
def genv(retrieval_dir: Path, tmp_path: Path) -> dict[str, Path]:
    questions = tmp_path / "questions.jsonl"
    write_questions(questions)
    s = fake_settings(retrieval_dir, tmp_path / "results", questions)
    s.paths.emb_dir = tmp_path / "emb"
    rcfg = write_settings(tmp_path / "retrieval.yaml", s)
    gcfg, _settings = write_generation_settings(tmp_path, rcfg)
    return {"cfg": gcfg, "results": tmp_path / "results", "tmp": tmp_path}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_dev_runs_tuning_derivation_and_test_guard(
    genv: dict[str, Path],
    script: Callable[[str], ModuleType],
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = ["--config", str(genv["cfg"])]
    run = script("run_generation")
    with pytest.raises(SystemExit):
        run.main([*base, "--split", "test"])  # dev only
    capsys.readouterr()
    systems = ["--systems", "G1", "G2", "G4", "G6"]
    assert run.main([*base, "--split", "dev", *systems, "--fake-llm"]) == 0
    out = capsys.readouterr().out
    assert "dev split: 2 answerable + 1 unanswerable" in out and "ETA" in out and "peak RSS" in out
    gen = genv["results"] / "generation" / "dev"
    for s in ("G1", "G2", "G4", "G6"):
        recs = _jsonl(gen / s / "per_question.jsonl")
        assert len(recs) == 3 and {r["split"] for r in recs} == {"dev"}
        assert (gen / s / "eval.jsonl").exists()
    tables = genv["results"] / "tables"
    assert [r["system"] for r in _rows(tables / "generation_dev.csv")] == ["G1", "G2", "G4", "G6"]
    assert {r["metric"] for r in _rows(tables / "generation_dev_bootstrap.csv")} == {
        "overall_accuracy",
        "faithfulness",
        "correct_refusal_rate",
    }
    assert _rows(tables / "generation_dev_by_type.csv")[0]["type"]
    # resumable: nothing left to run
    assert run.main([*base, "--split", "dev", *systems, "--fake-llm", "--no-eval"]) == 0
    out = capsys.readouterr().out
    assert out.count("3 done, 0 to run") == 4
    assert run.main([*base, "--systems", "G3", "--fake-llm"]) == 2  # derived: not run here
    capsys.readouterr()

    derive = script("derive_systems")
    with pytest.raises(SystemExit):
        derive.main([*base, "--split", "test"])
    capsys.readouterr()
    assert derive.main([*base]) == 1  # G5 needs the threshold
    assert "tune_abstention" in capsys.readouterr().out

    tune = script("tune_abstention")
    assert tune.main(base) == 0
    thr = json.loads((tables / "abstention_threshold.json").read_text(encoding="utf-8"))
    assert thr["split"] == "dev" and thr["n"] == 3 and thr["n_unanswerable"] == 1
    assert thr["retrieval_config_id"] == "structural|dense|fake-a" and thr["curve"]
    capsys.readouterr()

    assert derive.main(base) == 0
    out = capsys.readouterr().out
    assert "G3 <- G2" in out and "G5 <- G4 (threshold" in out
    rows = _rows(tables / "generation_dev.csv")
    assert [r["system"] for r in rows] == ["G1", "G2", "G3", "G4", "G5", "G6"]
    g2 = {r["qid"]: r for r in _jsonl(gen / "G2" / "per_question.jsonl")}
    g3 = _jsonl(gen / "G3" / "per_question.jsonl")
    for r in g3:
        assert r["cost"] == g2[r["qid"]]["cost"]
        if r["outcome"] == "answered":
            assert r["answer"] == g2[r["qid"]]["answer"]
    g4 = {r["qid"]: r for r in _jsonl(gen / "G4" / "per_question.jsonl")}
    for r in _jsonl(gen / "G5" / "per_question.jsonl"):
        score = r["retrieval"]["abstention_score"]
        if score < thr["threshold"]:
            assert r["reason"] == "low_retrieval_score" and r["cost"]["llm_calls"] == 0
        else:
            assert r["outcome"] == g4[r["qid"]]["outcome"] and r["cost"] == g4[r["qid"]]["cost"]

    # ---------------------------------------------------------------- test split guard
    ev = script("eval_generation")
    log = genv["results"] / "test_access_log.json"
    with pytest.raises(SystemExit):
        ev.main([*base, "--split", "dev"])  # only --split test
    capsys.readouterr()
    thr_path = tables / "abstention_threshold.json"
    saved = thr_path.read_text(encoding="utf-8")
    thr_path.unlink()
    assert ev.main([*base, "--split", "test", "--fake-llm"]) == 2
    assert "missing" in capsys.readouterr().out and not log.exists()  # test never opened
    thr_path.write_text(saved, encoding="utf-8")
    assert ev.main([*base, "--split", "test", "--fake-llm"]) == 0
    out = capsys.readouterr().out
    assert "TEST split opened (3 questions)" in out
    entries = json.loads(log.read_text(encoding="utf-8"))
    assert len(entries) == 1 and entries[0]["note"].startswith("generation")
    assert entries[0]["configs"] == ["G1", "G2", "G3", "G4", "G5", "G6"]
    assert [r["system"] for r in _rows(tables / "generation_test.csv")] == [
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
    ]
    assert (tables / "generation_test_by_type.csv").exists()
    assert {r["system"] for r in _rows(tables / "generation_test_bootstrap.csv")} == {
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
    }
    assert all(
        r["split"] == "test"
        for r in _jsonl(genv["results"] / "generation" / "test" / "G4" / "per_question.jsonl")
    )
    assert ev.main([*base, "--split", "test", "--fake-llm"]) == 3  # completed once already
    assert len(json.loads(log.read_text(encoding="utf-8"))) == 1
    capsys.readouterr()
    assert (
        ev.main([*base, "--split", "test", "--fake-llm", "--allow-repeat", "--systems", "G1", "G5"])
        == 0
    )
    entries = json.loads(log.read_text(encoding="utf-8"))
    assert (
        len(entries) == 2 and "(repeat)" in entries[1]["note"] and "(resume)" in entries[1]["note"]
    )


def test_ask_script_with_fake_llm(
    genv: dict[str, Path], script: Callable[[str], ModuleType], capsys: pytest.CaptureFixture[str]
) -> None:
    ask = script("ask")
    q = "Who may file a complaint and how can it be filed?"
    assert ask.main([q, "--config", str(genv["cfg"]), "--system", "G4", "--fake-llm"]) == 0
    out = capsys.readouterr().out
    assert "system:    G4" in out and "outcome:" in out and "step trace:" in out
    assert "retrieval " in out and "synthesis" in out and "timings: end-to-end" in out
    out.encode("ascii")
    assert ask.main([q, "--config", str(genv["cfg"]), "--system", "G5", "--fake-llm"]) == 0
    assert "threshold not tuned" in capsys.readouterr().out
    assert ask.main([q, "--config", str(genv["cfg"]), "--system", "G9", "--fake-llm"]) == 2
