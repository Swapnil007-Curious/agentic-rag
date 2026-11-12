"""Existing results stay byte-identical: every result table, per-question / per-query file and
test-access log of consumer_law and cs_ai (tests/data/frozen_results.json, see
tests/frozen_results.py). The generator-scale ablation may only append rows to the
generation_test tables and entries to the log, and add its own meta file and SUMMARY.md.
Regenerating SUMMARY.md keeps every existing section."""

from __future__ import annotations

import hashlib
import json

import pytest

from frozen_results import MANIFEST, REPO, check, normalized, summary_core
from ragsys.collections import get_collection
from ragsys.pipeline_plan import render_summary

FROZEN = json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_every_existing_result_file_is_unchanged() -> None:
    assert check(FROZEN) == []


def test_manifest_covers_both_collections_tables_and_logs() -> None:
    files = FROZEN["files"]
    for root in ("results", "collections/cs_ai/results"):
        for name in (
            "generation_test.csv",
            "generation_test_by_type.csv",
            "generation_test_bootstrap.csv",
            "generation_dev.csv",
            "retrieval_test.csv",
            "selected_configs.json",
        ):
            assert f"{root}/tables/{name}" in files
        assert files[f"{root}/test_access_log.json"]["mode"] == "json_prefix"
        assert files[f"{root}/tables/generation_test.csv"]["mode"] == "prefix"
    assert files["collections/cs_ai/results/tables/SUMMARY.md"]["mode"] == "summary"
    assert {e["mode"] for e in files.values()} == {"exact", "prefix", "json_prefix", "summary"}


def test_check_detects_edits() -> None:
    rel = "collections/cs_ai/results/tables/generation_dev.csv"
    entry = FROZEN["files"][rel]
    data = normalized(REPO / rel)
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    fake = {"files": {rel: {**entry, "sha256": "0" * 64}}, "allowed_new": []}
    problems = check(fake)
    assert any(p.startswith(rel) for p in problems)
    # a prefix entry accepts appended rows but not edited ones
    prel = "collections/cs_ai/results/tables/generation_test.csv"
    pe = FROZEN["files"][prel]
    assert pe["bytes"] == len(normalized(REPO / prel))


def test_summary_core_ignores_the_timestamp_and_the_ablation_section() -> None:
    text = "# T\n\nCollection `x` (profile `x`), written 2026 by y.\n\n## A\n\na\n"
    extra = (
        text.replace("written 2026", "written 2027")
        + "## Generator-scale ablation (test)\n\nrows\n"
    )
    assert summary_core(text) == summary_core(extra.rstrip("\n") + "\n").rstrip("\n") + "\n" or (
        summary_core(text).strip() == summary_core(extra).strip()
    )
    assert summary_core(text.replace("## A\n\na", "## A\n\nb")) != summary_core(text)


def test_regenerated_cs_ai_summary_keeps_every_existing_section() -> None:
    coll = get_collection("cs_ai")
    if not (REPO / coll.pipeline_state).exists():
        pytest.skip("the pipeline state (git-ignored) is needed to render the steps section")
    digest = FROZEN["files"]["collections/cs_ai/results/tables/SUMMARY.md"]["sha256"]
    rendered = summary_core(render_summary(coll))
    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == digest
