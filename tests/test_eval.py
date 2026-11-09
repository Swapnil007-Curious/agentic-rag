"""Evaluation schema and validator."""

from __future__ import annotations

import json
from pathlib import Path

import pydantic
import pytest

from ragsys.eval.schema import EvalQuestion
from ragsys.eval.validate import load_questions, suggest_ids, validate_file, validate_questions

NODE_IDS = [
    "cpa2019:s2",
    "cpa2019:s2(7)",
    "cpa2019:s35",
    "cpa2019:s35(1)",
    "cpa2019:s35(2)",
    "ecom2020:r4",
]


def q(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "q001",
        "question": "Who may file a complaint?",
        "type": "factual",
        "answerable": True,
        "gold_node_ids": ["cpa2019:s35(1)"],
        "gold_answer": "A consumer, among others.",
        "difficulty": "easy",
        "author": "Swapnil",
    }
    base.update(kw)
    return base


def test_schema_accepts_valid_and_rejects_bad() -> None:
    assert EvalQuestion.model_validate(q()).id == "q001"
    with pytest.raises(pydantic.ValidationError):
        EvalQuestion.model_validate(q(id="Q1"))
    with pytest.raises(pydantic.ValidationError):
        EvalQuestion.model_validate(q(type="opinion"))
    with pytest.raises(pydantic.ValidationError):
        EvalQuestion.model_validate(q(extra_field=1))


def test_consistency_rules() -> None:
    assert EvalQuestion.model_validate(
        q(answerable=False, type="unanswerable")
    ).consistency_errors() == ["unanswerable question must have empty gold_node_ids"]
    assert EvalQuestion.model_validate(q(gold_node_ids=[])).consistency_errors() == [
        "answerable question needs at least one gold_node_id"
    ]
    assert (
        "answerable=false requires"
        in EvalQuestion.model_validate(q(answerable=False, gold_node_ids=[])).consistency_errors()[
            0
        ]
    )


def test_suggestions_for_bad_ids() -> None:
    assert suggest_ids("cpa2019:s35(9)", NODE_IDS)[:3] == [
        "cpa2019:s35",
        "cpa2019:s35(1)",
        "cpa2019:s35(2)",
    ]
    assert "cpa2019:s35" in suggest_ids("s35", NODE_IDS)
    assert "cpa2019:s2(7)" in suggest_ids("cpa2019:s2(7", NODE_IDS)


def test_validator_counts_and_errors() -> None:
    qs = [
        EvalQuestion.model_validate(q()),
        EvalQuestion.model_validate(
            q(id="q002", type="definition", gold_node_ids=["cpa2019:s2(7)"], difficulty="medium")
        ),
        EvalQuestion.model_validate(
            q(id="q003", type="cross_reference", gold_node_ids=["ecom2020:r4", "cpa2019:s2"])
        ),
        EvalQuestion.model_validate(
            q(id="q004", type="unanswerable", answerable=False, gold_node_ids=[], difficulty="hard")
        ),
        EvalQuestion.model_validate(q(id="q004", gold_node_ids=["cpa2019:s35(9)"])),
    ]
    rep = validate_questions(qs, NODE_IDS)
    assert not rep.ok
    assert any("duplicate question id q004" in e for e in rep.errors)
    bad = next(e for e in rep.errors if "cpa2019:s35(9)" in e)
    assert "did you mean: cpa2019:s35" in bad
    c = rep.counts
    assert (c["total"], c["answerable"], c["unanswerable"]) == (5, 4, 1)
    assert c["by_type"] == {"cross_reference": 1, "definition": 1, "factual": 2, "unanswerable": 1}
    assert c["by_difficulty"] == {"easy": 3, "hard": 1, "medium": 1}
    assert c["by_document"] == {"(none)": 1, "cpa2019": 4, "ecom2020": 1}
    assert c["progress"] == {"answerable": "4/90", "unanswerable": "1/30"}


def test_load_questions_reports_line_errors(tmp_path: Path) -> None:
    p = tmp_path / "questions.jsonl"
    p.write_text(
        json.dumps(q()) + "\n{not json}\n" + json.dumps(q(id="q002", difficulty="trivial")) + "\n",
        encoding="utf-8",
    )
    qs, errors = load_questions(p)
    assert len(qs) == 1
    assert errors[0].startswith("line 2: invalid JSON")
    assert errors[1].startswith("line 3 (q002): difficulty")


def test_repository_examples_are_valid(tmp_path: Path) -> None:
    repo_questions = Path(__file__).resolve().parents[1] / "data" / "eval" / "questions.jsonl"
    qs, errors = load_questions(repo_questions)
    assert errors == [] and len(qs) == 3
    assert sorted(x.type for x in qs) == ["cross_reference", "definition", "unanswerable"]
    nodes = tmp_path / "nodes.jsonl"
    ids = sorted({g for x in qs for g in x.gold_node_ids})
    nodes.write_text("".join(json.dumps({"node_id": i}) + "\n" for i in ids), encoding="utf-8")
    assert validate_file(repo_questions, nodes).ok
