"""The command-line scripts run end to end on the synthetic corpus (in-process)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from ragsys.ingest.pipeline import IngestResult


def test_browse_commands(
    ingested: IngestResult,
    processed_dir: Path,
    script: Callable[[str], ModuleType],
    capsys: pytest.CaptureFixture[str],
) -> None:
    browse = script("browse")
    pd = ["--processed-dir", str(processed_dir)]

    assert browse.main([*pd, "list", "--doc", "testact"]) == 0
    out = capsys.readouterr().out
    assert "testact:s3" in out and "Section 3 - Filing of complaint" in out and "CHAPTER II" in out

    assert browse.main([*pd, "show", "testact:s3"]) == 0
    out = capsys.readouterr().out
    assert "[testact:s3(1)(a)]" in out and "[testact:s3(1)-prov1] Provided that" in out
    assert "pages 3-3" in out and "referred to by:" in out
    out.encode("ascii")  # console-safe: em dashes are folded to "--"

    # PowerShell splits an unquoted testact:s3(1) into two arguments; browse rejoins them
    assert browse.main([*pd, "show", "testact:s3", "1"]) == 0
    assert capsys.readouterr().out.startswith("testact:s3(1)")

    assert browse.main([*pd, "show", "testact:s9"]) == 1
    assert "Did you mean" in capsys.readouterr().out

    assert browse.main([*pd, "search", "complaint"]) == 0
    out = capsys.readouterr().out
    assert "match 'complaint'" in out and "testact:s3(1)" in out

    assert browse.main([*pd, "search", "consumer"]) == 0
    assert capsys.readouterr().out.splitlines()[1].startswith("testact:s2(1)")  # definition first

    assert browse.main([*pd, "defs"]) == 0
    out = capsys.readouterr().out
    assert "6 defined term(s)" in out and "testact:s2(1)" in out and "consumer" in out


def test_validate_eval_script(
    ingested: IngestResult,
    processed_dir: Path,
    tmp_path: Path,
    script: Callable[[str], ModuleType],
    capsys: pytest.CaptureFixture[str],
) -> None:
    validate = script("validate_eval")
    nodes = str(processed_dir / "structure" / "nodes.jsonl")
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps(
            {
                "id": "q001",
                "question": "What is a trader?",
                "type": "definition",
                "answerable": True,
                "gold_node_ids": ["testact:s2(3)"],
                "gold_answer": "A person who sells goods.",
                "difficulty": "easy",
                "author": "Swapnil",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert validate.main(["--questions", str(good), "--nodes", nodes]) == 0
    out = capsys.readouterr().out
    assert "progress: answerable 1/90, unanswerable 0/30" in out and "RESULT: OK" in out
    assert "dev answerable" in out and "WARNING: dev split has only 0 answerable" in out

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        good.read_text(encoding="utf-8").replace("testact:s2(3)", "testact:s2(9)"), encoding="utf-8"
    )
    assert validate.main(["--questions", str(bad), "--nodes", nodes]) == 1
    out = capsys.readouterr().out
    assert "did you mean: testact:s2" in out and "RESULT: FAILED" in out


def test_ingest_and_chunk_scripts(
    pdf_dir: Path,
    tmp_path: Path,
    script: Callable[[str], ModuleType],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from conftest import fixture_config

    cfg_path = tmp_path / "ingest.yaml"
    cfg_path.write_text(yaml.safe_dump(fixture_config().model_dump(mode="json")), encoding="utf-8")
    out_dir = tmp_path / "processed"
    ingest = script("ingest")
    assert (
        ingest.main(
            ["--config", str(cfg_path), "--raw-dir", str(pdf_dir), "--processed-dir", str(out_dir)]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "testact" in printed and "sections=  5" in printed
    report = json.loads(
        (out_dir / "structure" / "structure_report.json").read_text(encoding="utf-8")
    )
    assert report["documents"]["testact"]["sections_or_rules"] == 5
    assert report["run_info"]["cpu_model"] and report["config"]["documents"]

    chunk_cfg = tmp_path / "chunking.yaml"
    chunk_cfg.write_text(
        yaml.safe_dump(
            {"paths": {"processed_dir": str(out_dir), "cache_dir": str(tmp_path / "cache")}}
        ),
        encoding="utf-8",
    )
    chunk = script("chunk")
    assert chunk.main(["--chunker", "structural", "--config", str(chunk_cfg)]) == 0
    assert chunk.main(["--chunker", "fixed_256", "--config", str(chunk_cfg)]) == 0
    rep = json.loads((out_dir / "chunks" / "chunk_report.json").read_text(encoding="utf-8"))
    assert set(rep["chunkers"]) == {"structural", "fixed_256"}  # merged across runs
    assert rep["chunkers"]["structural"]["chunks_crossing_section_boundaries"] == 0
    assert (out_dir / "chunks" / "structural.jsonl").exists()
    with pytest.raises(SystemExit):
        chunk.main(["--chunker", "nope", "--config", str(chunk_cfg)])
