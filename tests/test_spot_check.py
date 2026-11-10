"""Spot check of automatic items: stratified sampling, y/n/s session, counts and Wilson
interval, and removals that only ever touch llm_auto_filtered items."""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ragsys.eval.schema import EvalQuestion
from ragsys.eval.spot_check import remove_auto_items, run_session, sample_items, summarise
from ragsys.eval.validate import load_questions
from ragsys.retrieval.corpus import load_chunks
from retrieval_helpers import fake_settings, write_settings


def q(
    i: int, answerable: bool = True, origin: str = "llm_auto_filtered", **kw: Any
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": f"q{i:03d}",
        "question": f"Question number {i} about complaints?",
        "type": "factual" if answerable else "unanswerable",
        "answerable": answerable,
        "gold_node_ids": ["testact:s3(1)"] if answerable else [],
        "gold_answer": "In writing or electronically." if answerable else "Not in the corpus.",
        "difficulty": "medium",
        "author": "auto (fake) - filtered" if origin == "llm_auto_filtered" else "Swapnil",
        "origin": origin,
    }
    if origin == "llm_auto_filtered":
        base["gen_meta"] = {"method": "draft_from_source" if answerable else "out_of_scope"}
    base.update(kw)
    return base


def models(rows: list[dict[str, Any]]) -> list[EvalQuestion]:
    return [EvalQuestion.model_validate(r) for r in rows]


def scripted(answers: list[str]) -> Callable[[str], str]:
    it: Iterator[str] = iter(answers)

    def _input(_prompt: str) -> str:
        try:
            return next(it)
        except StopIteration as exc:
            raise EOFError from exc

    return _input


def test_sample_is_stratified_seeded_and_ignores_human_items() -> None:
    rows = [q(i) for i in range(1, 10)] + [q(i, False) for i in range(10, 13)]
    rows += [q(20, origin="human"), q(21, False, origin="human")]
    qs = models(rows)
    s = sample_items(qs, 8, seed=0)
    assert len(s) == 8 and all(x.origin == "llm_auto_filtered" for x in s)
    assert sum(1 for x in s if x.answerable) == 6 and sum(1 for x in s if not x.answerable) == 2
    assert [x.id for x in s] == [x.id for x in sample_items(qs, 8, seed=0)]
    assert [x.id for x in s] != [x.id for x in sample_items(qs, 8, seed=1)]
    # a small sample still gets one item of each stratum
    s2 = sample_items(qs, 2, seed=0)
    assert sorted(x.answerable for x in s2) == [False, True]
    assert len(sample_items(qs, 100, seed=0)) == 12


def test_session_counts_error_rate_and_wilson() -> None:
    qs = models([q(i) for i in range(1, 6)] + [q(6, False)])
    lines: list[str] = []
    judgments = run_session(
        qs, lambda x: [x.question], scripted(["y", "n", "x", "s", "y", "n"]), lines.append
    )
    assert judgments == {
        "q001": "correct",
        "q002": "wrong",
        "q003": "skipped",
        "q004": "correct",
        "q005": "wrong",
    }
    assert any("please type y, n, s or q" in x for x in lines) and "quit" in lines
    res = summarise(qs, judgments, seed=0)
    assert res["counts"] == {"correct": 2, "wrong": 2, "skipped": 1, "not_judged": 1}
    assert res["judged"] == 4 and res["error_rate"] == 0.5
    lo, hi = res["wilson_95"]
    assert lo == pytest.approx(0.15, abs=0.01) and hi == pytest.approx(0.85, abs=0.01)
    assert res["ids_wrong"] == ["q002", "q005"] and res["ids_skipped"] == ["q003"]
    assert res["by_stratum"]["unanswerable"]["not_judged"] == 1
    assert res["wrong_items"][0] == {"id": "q002", "question": qs[1].question}
    # quitting right away keeps nothing
    assert run_session(qs, lambda x: [], scripted(["q"]), lines.append) == {}


def test_remove_only_automatic_items_with_matching_text(tmp_path: Path) -> None:
    p = tmp_path / "questions.jsonl"
    rows = [q(1, origin="human"), q(2), q(3), q(4, origin="llm_draft_human_reviewed")]
    human_line = json.dumps(rows[0], ensure_ascii=False)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    removed = remove_auto_items(
        p,
        {"q001": None, "q002": rows[1]["question"], "q003": "an older question text", "q004": None},
    )
    assert removed == ["q002"]
    left, _ = load_questions(p)
    assert [x.id for x in left] == ["q001", "q003", "q004"]
    assert p.read_text(encoding="utf-8").splitlines()[0] == human_line


def test_spot_check_script_session_and_apply_removals(
    retrieval_dir: Path,
    tmp_path: Path,
    script: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    questions = tmp_path / "questions.jsonl"
    chunk_ids = load_chunks(retrieval_dir, "structural").ids[:5]
    rows = [q(1, origin="human"), q(2), q(3)]
    rows.append(
        q(
            4,
            False,
            gen_meta={
                "method": "perturbation",
                "target": "the RTI Act",
                "retrieved_chunk_ids": chunk_ids,
                "retrieval_config": "structural|hybrid_rrf|bge-small",
            },
        )
    )
    rows.append(q(5, False))  # no stored retrieval: BM25 fallback
    questions.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg = write_settings(
        tmp_path / "r.yaml", fake_settings(retrieval_dir, tmp_path / "results", questions)
    )
    spot = script("spot_check")
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\nn\ny\ns\n"))
    assert spot.main(["--config", str(cfg), "--n", "4", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "source text:" in out and "[testact:s3(1)]" in out
    assert "top-5 passages retrieved at construction" in out and "BM25 top-5 passages" in out
    assert "error rate" in out and "Wilson" in out
    res_path = tmp_path / "results" / "tables" / "eval_spot_check.json"
    res = json.loads(res_path.read_text(encoding="utf-8"))
    assert res["n_sampled"] == 4 and res["judged"] == 3 and len(res["ids_wrong"]) == 1
    assert res["removals_applied"] is False and len(res["wilson_95"]) == 2
    before = questions.read_text(encoding="utf-8")
    assert before == "".join(json.dumps(r) + "\n" for r in rows)  # the session never edits
    assert spot.main(["--config", str(cfg), "--apply-removals"]) == 0
    left, _ = load_questions(questions)
    assert res["ids_wrong"][0] not in {x.id for x in left} and len(left) == 4
    assert left[0].id == "q001"
    res = json.loads(res_path.read_text(encoding="utf-8"))
    assert res["removals_applied"] is True and res["removed_ids"] == res["ids_wrong"]
