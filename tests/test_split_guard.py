"""Deterministic dev/test split and the test-split guard (runtime + static source check)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ragsys.eval.split import (
    TestAccess,
    TestSplitLocked,
    grant_test_access,
    questions_for_split,
    split_of,
)

REPO = Path(__file__).resolve().parents[1]


def test_split_is_deterministic_and_about_30_percent() -> None:
    assert split_of("q001") == "test" and split_of("q003") == "dev"
    ids = [f"syn{i:05d}" for i in range(1000)]
    first = [split_of(i) for i in ids]
    assert first == [split_of(i) for i in ids]
    share = first.count("dev") / len(ids)
    assert 0.25 <= share <= 0.35


def test_dev_questions_need_no_grant_test_questions_do() -> None:
    ids = ["q001", "q002", "q003"]
    assert questions_for_split(ids, "dev") == ["q003"]
    with pytest.raises(TestSplitLocked):
        questions_for_split(ids, "test")
    with pytest.raises(TestSplitLocked):
        TestAccess(object(), "forged")


def test_grant_refused_outside_eval_script(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    with pytest.raises(TestSplitLocked):
        grant_test_access(log, ["x"], 3)
    assert not log.exists()


def test_static_only_eval_retrieval_opens_test() -> None:
    allowed = {
        REPO / "src" / "ragsys" / "eval" / "split.py",
        REPO / "scripts" / "eval_retrieval.py",
    }
    offenders = []
    for base in (REPO / "src", REPO / "scripts"):
        for py in base.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if re.search(r"\bgrant_test_access\b", text) and py not in allowed:
                offenders.append(str(py.relative_to(REPO)))
            if (
                re.search(r"""choices=\[[^\]]*["']test["']""", text)
                and py.name != "eval_retrieval.py"
            ):
                offenders.append(f"{py.relative_to(REPO)} (--split test)")
    assert offenders == []
    grid = (REPO / "scripts" / "run_retrieval_grid.py").read_text(encoding="utf-8")
    assert 'choices=["dev"]' in grid
