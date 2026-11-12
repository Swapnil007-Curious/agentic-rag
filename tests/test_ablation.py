"""Generator-scale ablation end to end with fake models: after the standard test evaluation,
G2H / G4H (G2 / G4 on the "hosted" backend with its own context budgets) append rows to the
generation_test tables without touching a byte of the existing ones, bootstrap against G1, G2
and the mirrored system, log the test access, record the model, tokens and latency, refuse a
repeat and, with --allow-repeat, replace only their own rows."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from agent_helpers import generation_settings_dict
from ragsys.eval.ablation import AblationError, append_rows
from ragsys.pipeline_plan import ablation_section
from retrieval_helpers import fake_settings, write_questions, write_settings


@pytest.fixture()
def aenv(retrieval_dir: Path, tmp_path: Path) -> dict[str, Path]:
    questions = tmp_path / "questions.jsonl"
    write_questions(questions)
    s = fake_settings(retrieval_dir, tmp_path / "results", questions)
    s.paths.emb_dir = tmp_path / "emb"
    rcfg = write_settings(tmp_path / "retrieval.yaml", s)
    data = generation_settings_dict(tmp_path, rcfg)
    llm_yaml = tmp_path / "llm.yaml"
    llm_yaml.write_text(
        yaml.safe_dump(
            {
                "default_backend": "local",
                "backends": {
                    "local": {"base_url": "http://local.invalid/v1", "model": "fake"},
                    "hosted": {
                        "base_url": "http://hosted.invalid/v1",
                        "model": "big-model",
                        "requires_key": True,
                        "label": "Hosted: big-model",
                        "provider": "Test provider",
                        "model_selected_on": "2026-11-12",
                        "model_selection": "chosen before any result",
                        "context_budget_tokens": 400,
                        "revision_budget_tokens": 800,
                        "generation_overrides": {"max_tokens": 1024},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    data["llm_config"] = str(llm_yaml)
    data["llm_overrides"] = {"temperature": 0.0, "cache_dir": str(tmp_path / "llm_cache")}
    data["systems"]["G2H"] = {
        "description": "G2 hosted",
        "retrieval": "structural|dense|fake-a",
        "backend": "hosted",
        "ablation_of": "G2",
    }
    data["systems"]["G4H"] = {
        "description": "G4 hosted",
        "retrieval": "structural|dense|fake-a",
        "guards": True,
        "critic": True,
        "backend": "hosted",
        "ablation_of": "G4",
    }
    gcfg = tmp_path / "generation.yaml"
    gcfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    return {"cfg": gcfg, "results": tmp_path / "results"}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


TABLES = ("generation_test.csv", "generation_test_by_type.csv", "generation_test_bootstrap.csv")


def test_ablation_appends_rows_and_keeps_existing_bytes(
    aenv: dict[str, Path], script: Callable[[str], ModuleType], capsys: pytest.CaptureFixture[str]
) -> None:
    ev = script("eval_generation")
    base = ["--config", str(aenv["cfg"]), "--split", "test", "--fake-llm"]
    abl = [*base, "--systems", "G2H", "G4H"]
    assert ev.main(abl) == 2  # the standard evaluation must be complete first
    assert "not complete" in capsys.readouterr().out
    assert ev.main([*base, "--systems", "G1", "G2", "G3", "G4"]) == 0
    capsys.readouterr()
    tables = aenv["results"] / "tables"
    frozen = {t: (tables / t).read_bytes() for t in TABLES}
    meta_before = (tables / "generation_test.meta.json").read_bytes()
    gen = aenv["results"] / "generation" / "test"
    per_q = {s: (gen / s / "per_question.jsonl").read_bytes() for s in ("G1", "G2", "G4")}
    evals = {s: (gen / s / "eval.jsonl").read_bytes() for s in ("G1", "G2", "G4")}

    assert ev.main(abl) == 0
    out = capsys.readouterr().out
    assert "TEST split opened" in out and "Hosted: big-model" in out
    for t in TABLES:
        assert (tables / t).read_bytes().startswith(frozen[t]), t  # existing rows untouched
    assert (tables / "generation_test.meta.json").read_bytes() == meta_before
    for s in ("G1", "G2", "G4"):
        assert (gen / s / "per_question.jsonl").read_bytes() == per_q[s]
        assert (gen / s / "eval.jsonl").read_bytes() == evals[s]
    rows = _rows(tables / "generation_test.csv")
    assert [r["system"] for r in rows] == ["G1", "G2", "G3", "G4", "G2H", "G4H"]
    label = rows[-1]["label"]
    assert label == "ablation: hosted fake-hosted-llm (chosen 2026-11-12)"
    boot = [(r["system"], r["reference"]) for r in _rows(tables / "generation_test_bootstrap.csv")]
    for pair in (("G2H", "G1"), ("G2H", "G2"), ("G4H", "G1"), ("G4H", "G2"), ("G4H", "G4")):
        assert pair in boot
    meta = json.loads((tables / "generation_test_ablation.meta.json").read_text(encoding="utf-8"))
    assert meta["backend"]["model"] == "fake-hosted-llm" and meta["backend"]["provider"]
    assert meta["backend"]["model_selected_on"] == "2026-11-12"
    assert meta["context"]["G2H"]["budget_tokens"] == 400
    assert meta["comparisons"]["G4H"] == ["G1", "G2", "G4"]
    assert {"prompt_tokens", "completion_tokens", "api_latency_ms_p50"} <= set(meta["cost"]["G4H"])
    recs = [json.loads(x) for x in (gen / "G2H" / "per_question.jsonl").read_text().splitlines()]
    assert recs and all(
        r["generator"] == {"backend": "hosted", "model": "fake-hosted-llm"} for r in recs
    )
    assert all(r["rounds"][0]["context"]["budget_tokens"] == 400 for r in recs)
    assert {r["fingerprint"] for r in recs}.isdisjoint(
        {json.loads(x)["fingerprint"] for x in per_q["G2"].decode().splitlines()}
    )
    log = json.loads((aenv["results"] / "test_access_log.json").read_text(encoding="utf-8"))
    assert len(log) == 2 and log[1]["note"].startswith("generation ablation")
    assert "fake-hosted-llm" in log[1]["note"] and log[1]["configs"] == ["G2H", "G4H"]
    assert (gen / "_complete_ablation.json").exists() and (gen / "_complete.json").exists()

    # the summary section
    section = "\n".join(ablation_section(tables, [("overall_accuracy", "accuracy")]))
    assert "## Generator-scale ablation (test)" in section and "| G4H |" in section
    assert "chosen 2026-11-12" in section and "| G4H | G4 |" in section

    # a repeat is refused; with --allow-repeat only the ablation rows are replaced
    assert ev.main(abl) == 3
    capsys.readouterr()
    assert ev.main([*abl, "--allow-repeat"]) == 0
    rows = _rows(tables / "generation_test.csv")
    assert [r["system"] for r in rows] == ["G1", "G2", "G3", "G4", "G2H", "G4H"]
    for t in TABLES:
        assert (tables / t).read_bytes().startswith(frozen[t]), t
    log = json.loads((aenv["results"] / "test_access_log.json").read_text(encoding="utf-8"))
    assert len(log) == 3 and "(repeat)" in log[2]["note"]


def test_append_rows_checks_columns(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_bytes(b"system,label,x\r\nG1,,1.0000\r\n")
    append_rows(p, [{"system": "G2H", "label": "ablation: h", "x": 0.5}], ["G2H"])
    assert p.read_bytes() == b"system,label,x\r\nG1,,1.0000\r\nG2H,ablation: h,0.5000\r\n"
    append_rows(p, [{"system": "G2H", "label": "ablation: h", "x": 0.25}], ["G2H"])
    assert p.read_bytes().count(b"G2H") == 1 and p.read_bytes().startswith(
        b"system,label,x\r\nG1,,1.0000\r\n"
    )
    with pytest.raises(AblationError):
        append_rows(p, [{"system": "G4H", "label": "", "y": 1}], ["G4H"])
    with pytest.raises(AblationError):
        append_rows(tmp_path / "missing.csv", [], [])
