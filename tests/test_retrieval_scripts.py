"""build_index / search / grid (resume) / select_configs / eval_retrieval (test guard) scripts on
the synthetic corpus with fake models."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from retrieval_helpers import fake_settings, write_questions, write_settings


@pytest.fixture()
def env(retrieval_dir: Path, tmp_path: Path) -> dict[str, Path]:
    questions = tmp_path / "questions.jsonl"
    write_questions(questions)
    results = tmp_path / "results"
    s = fake_settings(retrieval_dir, results, questions)
    s.paths.emb_dir = tmp_path / "emb"
    s.paths.chroma_dir = tmp_path / "chroma"
    cfg = write_settings(tmp_path / "retrieval.yaml", s)
    return {"cfg": cfg, "results": results, "questions": questions, "tmp": tmp_path}


def test_build_index_and_search(
    env: dict[str, Path],
    script: Callable[[str], ModuleType],
    capsys: pytest.CaptureFixture[str],
) -> None:
    build = script("build_index")
    args = ["--config", str(env["cfg"]), "--chunker", "structural", "fixed_64"]
    assert build.main([*args, "--models", "fake-a"]) == 0
    out = capsys.readouterr().out
    assert "computed (missing)" in out and "identical top-10" in out
    report = json.loads((env["tmp"] / "emb" / "index_report.json").read_text(encoding="utf-8"))
    par = report["chunkers"]["structural"]["embeddings"]["fake-a"]["chroma"]
    assert par["identical_top10"] == par["queries"] and report["run_info"]["python"]
    assert build.main([*args, "--models", "fake-a", "--no-chroma"]) == 0
    assert "cached" in capsys.readouterr().out

    search = script("search")
    cid = "structural|hybrid_rrf|fake-a|xref|rerank-fake"
    q = "Who can file a complaint about defective goods?"
    assert search.main([q, "--config", str(env["cfg"]), "--config-id", cid]) == 0
    out = capsys.readouterr().out
    assert f"config: {cid}" in out and "provenance:" in out and "rerank=" in out
    assert "#1 " in out and "path:" in out
    out.encode("ascii")  # console-safe
    flags = ["--chunker", "fixed_64", "--retriever", "dense", "--model", "fake-a"]
    assert search.main([q, "--config", str(env["cfg"]), *flags, "--no-xref", "--no-rerank"]) == 0
    out = capsys.readouterr().out
    assert "config: fixed_64|dense|fake-a\n" in out and "rerank=" not in out


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_grid_resume_select_and_test_guard(
    env: dict[str, Path],
    script: Callable[[str], ModuleType],
    capsys: pytest.CaptureFixture[str],
) -> None:
    grid = script("run_retrieval_grid")
    base = ["--config", str(env["cfg"])]
    with pytest.raises(SystemExit):
        grid.main([*base, "--split", "test"])  # the grid cannot touch test
    capsys.readouterr()
    assert grid.main([*base, "--split", "dev"]) == 0
    out = capsys.readouterr().out
    assert "24 config(s)" in out and "skip" not in out and "total wall time" in out
    tables = env["results"] / "tables"
    rows = _rows(tables / "retrieval_dev.csv")
    assert len(rows) == 24 and {r["n_answerable"] for r in rows} == {"2"}
    assert {r["n_unanswerable"] for r in rows} == {"1"}
    for col in ("recall@1", "recall@10", "hit@5", "mrr@10", "ndcg@10", "coverage@1024"):
        assert all(r[col] != "" for r in rows)
    assert all(r["latency_total_p50_ms"] for r in rows)
    assert any(r["auroc_top1_rerank"] for r in rows if r["rerank"])
    by_type = _rows(tables / "retrieval_dev_by_type.csv")
    assert {r["type"] for r in by_type} == {"procedural", "numeric", "unanswerable"}
    assert (tables / "retrieval_dev_bootstrap.csv").exists()
    meta = json.loads((tables / "retrieval_dev.meta.json").read_text(encoding="utf-8"))
    assert meta["run_info"]["git"] is not None and meta["settings"]["paths"]
    one = env["results"] / "retrieval" / "dev" / "fixed_64__bm25"
    summary = json.loads((one / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["chunker"] == "fixed_64" and summary["run_info"]["cpu_model"]
    recs = [
        json.loads(x) for x in (one / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {r["split"] for r in recs} == {"dev"} and len(recs) == 3

    # resume: everything is skipped; --overwrite recomputes
    assert grid.main([*base, "--split", "dev"]) == 0
    assert capsys.readouterr().out.count("skip (exists)") == 24
    assert grid.main([*base, "--only", "fixed_64|bm25", "--overwrite"]) == 0
    out = capsys.readouterr().out
    assert "skip" not in out and "fixed_64|bm25" in out

    select = script("select_configs")
    assert select.main(base) == 0
    sel = json.loads((tables / "selected_configs.json").read_text(encoding="utf-8"))
    roles = {r for s in sel["selected"] for r in s["roles"]}
    assert {"best_overall", "best_fixed_64", "best_structural"} <= roles
    assert {"baseline_naive_rag", "baseline_bm25"} <= roles
    assert (
        "coverage@1024" in sel["rule"] and sel["reference_baseline_id"] == "fixed_64|dense|fake-a"
    )
    capsys.readouterr()

    ev = script("eval_retrieval")
    sel_path = str(tables / "selected_configs.json")
    assert (
        ev.main([*base, "--split", "test", "--configs-from", sel_path, "--min-answerable", "99"])
        == 2
    )
    assert not (env["results"] / "test_access_log.json").exists()  # refused before opening
    assert ev.main([*base, "--split", "test", "--configs-from", sel_path]) == 0
    log = json.loads((env["results"] / "test_access_log.json").read_text(encoding="utf-8"))
    assert len(log) == 1 and log[0]["n_questions"] == 3 and "git_commit" in log[0]
    assert "git_dirty" in log[0] and log[0]["configs"]
    test_rows = _rows(tables / "retrieval_test.csv")
    assert {r["n_answerable"] for r in test_rows} == {"3"}
    boot = _rows(tables / "retrieval_test_bootstrap.csv")
    assert {r["metric"] for r in boot} == {"recall@5", "coverage@1024"}
    assert {r["baseline_id"] for r in boot} == {"fixed_64|dense|fake-a"}
    assert (tables / "retrieval_test_by_type.csv").exists()
    # a second evaluation of test needs an explicit flag and is logged too
    assert ev.main([*base, "--split", "test", "--configs-from", sel_path]) == 3
    assert ev.main([*base, "--split", "test", "--configs-from", sel_path, "--allow-repeat"]) == 0
    log = json.loads((env["results"] / "test_access_log.json").read_text(encoding="utf-8"))
    assert len(log) == 2


def test_smoke_grid_is_labelled(
    env: dict[str, Path],
    script: Callable[[str], ModuleType],
    capsys: pytest.CaptureFixture[str],
) -> None:
    grid = script("run_retrieval_grid")
    assert grid.main(["--config", str(env["cfg"]), "--smoke", "--only", "structural|bm25"]) == 0
    assert "SMOKE - NOT RESULTS" in capsys.readouterr().out
    rows = _rows(env["results"] / "smoke" / "tables" / "retrieval_dev.csv")
    assert rows and {r["label"] for r in rows} == {"smoke - not results"}
    assert not (env["results"] / "tables").exists()
