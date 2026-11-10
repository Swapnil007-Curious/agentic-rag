"""Question drafting (FakeLLM) and the review / add-unanswerable flows with scripted input."""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

from ragsys.eval.drafting import (
    TYPE_MIX,
    allocate,
    build_messages,
    build_pools,
    draft_flags,
    lexical_overlap,
    load_drafts,
    stratified_sample,
)
from ragsys.eval.review import add_unanswerable, next_question_id, review_drafts
from ragsys.eval.validate import load_questions
from ragsys.ingest.pipeline import load_nodes, load_xrefs
from ragsys.retrieval.corpus import NodeGraph

EXISTING = {
    "id": "q007",
    "question": "What is a trader?",
    "type": "definition",
    "answerable": True,
    "gold_node_ids": ["testact:s2(3)"],
    "gold_answer": "A person who sells goods.",
    "difficulty": "easy",
    "author": "Swapnil",
}


def scripted(answers: list[str]) -> Callable[[str], str]:
    it: Iterator[str] = iter(answers)

    def _input(prompt: str) -> str:
        try:
            return next(it)
        except StopIteration as exc:
            raise EOFError from exc

    return _input


def test_allocation_matches_type_mix() -> None:
    assert allocate(90, TYPE_MIX) == TYPE_MIX
    a150 = allocate(150, TYPE_MIX)
    assert sum(a150.values()) == 150 and a150["factual"] == 33 and a150["multi_section"] == 17
    assert allocate(6, TYPE_MIX) == dict.fromkeys(TYPE_MIX, 1)
    assert allocate(0, TYPE_MIX) == dict.fromkeys(TYPE_MIX, 0)


def test_pools_and_stratified_sampling(retrieval_dir: Path) -> None:
    nodes = load_nodes(retrieval_dir / "structure" / "nodes.jsonl")
    xrefs = load_xrefs(retrieval_dir / "structure" / "xrefs.jsonl")
    pools = build_pools(nodes, xrefs, NodeGraph(nodes))
    assert {c.node_ids[0] for c in pools["definition"]} >= {"testact:s2(1)", "testact:s2(2)"}
    assert any(
        c.node_ids == ["testrules2024:r3(2)", "testact:s3(1)"] for c in pools["multi_section"]
    )
    assert all(len(c.nodes) == 2 for c in pools["cross_reference"])
    assert any("testact:s4(1)" in c.node_ids for c in pools["numeric"])  # "thirty days"
    alloc = allocate(8, TYPE_MIX)
    s1, _short = stratified_sample(pools, alloc, seed=0)
    s2, _ = stratified_sample(pools, alloc, seed=0)
    assert [c.node_ids for c in s1] == [c.node_ids for c in s2]  # deterministic
    used = [n for c in s1 for n in c.node_ids]
    assert len(used) == len(set(used))  # no node reused across drafts
    excl = build_pools(nodes, xrefs, NodeGraph(nodes), exclude={"testact:s2(1)"})
    assert "testact:s2(1)" not in {c.node_ids[0] for c in excl["definition"]}


def test_prompt_and_flags(retrieval_dir: Path) -> None:
    nodes = load_nodes(retrieval_dir / "structure" / "nodes.jsonl")
    xrefs = load_xrefs(retrieval_dir / "structure" / "xrefs.jsonl")
    pools = build_pools(nodes, xrefs, NodeGraph(nodes))
    msg = build_messages(pools["definition"][0], {})[1]["content"]
    assert "Do not mention section" in msg and "40 words" in msg and '"type": "definition"' in msg
    xr = build_messages(pools["cross_reference"][0], {})[1]["content"]
    assert "You may mention" in xr and "SECOND PROVISION" in xr
    assert lexical_overlap("What does goods mean?", "goods means movable property") == 0.5
    assert lexical_overlap("", "x") == 0.0
    assert draft_flags("factual", "What does section 3 say?", "short", 0.7) == [
        "copy_like",
        "mentions_provision_number",
    ]
    assert draft_flags("cross_reference", "What does section 3 say?", "x " * 41, 0.1) == [
        "answer_too_long"
    ]


def test_draft_script_with_fake_llm(
    retrieval_dir: Path,
    tmp_path: Path,
    script: Callable[[str], ModuleType],
    capsys: pytest.CaptureFixture[str],
) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps(EXISTING) + "\n", encoding="utf-8")
    before = questions.read_text(encoding="utf-8")
    drafts = tmp_path / "drafts.jsonl"
    draft = script("draft_questions")
    args = ["--fake", "--processed-dir", str(retrieval_dir), "--questions", str(questions)]
    assert draft.main([*args, "--n", "6", "--seed", "0", "--out", str(drafts)]) == 0
    out = capsys.readouterr().out
    assert "FakeLLM" in out and "review_drafts.py" in out
    rows = load_drafts(drafts)
    assert rows and len(rows) <= 6
    assert all(r["status"] == "pending" and r["llm_model"] == "fake-llm" for r in rows)
    assert all(r["gold_node_ids"] and "lexical_overlap" in r and "overlap_flag" in r for r in rows)
    assert all(r["type"] != "unanswerable" for r in rows)
    assert all("testact:s2(3)" not in r["gold_node_ids"] for r in rows)  # already a question
    assert [r["draft_id"] for r in rows] == [f"d{i:04d}" for i in range(1, len(rows) + 1)]
    assert questions.read_text(encoding="utf-8") == before  # never touched
    # a second run appends new drafts for other nodes
    assert draft.main([*args, "--n", "2", "--types", "definition", "--out", str(drafts)]) == 0
    rows2 = load_drafts(drafts)
    assert len(rows2) > len(rows) and rows2[-1]["type"] == "definition"
    firsts = [tuple(r["gold_node_ids"]) for r in rows2]
    assert len(firsts) == len(set(firsts))


def _draft(i: int, gold: list[str], overlap: float = 0.2) -> dict[str, object]:
    return {
        "draft_id": f"d{i:04d}",
        "status": "pending",
        "type": "factual",
        "difficulty": "medium",
        "question": f"Question number {i}?",
        "gold_answer": f"Answer {i}.",
        "gold_node_ids": gold,
        "source_text": "[testact:s3(1)] A complaint may be filed by a consumer.",
        "llm_model": "fake-llm",
        "lexical_overlap": overlap,
        "overlap_flag": overlap > 0.6,
        "flags": [],
    }


def test_review_loop_accept_edit_reject_skip(retrieval_dir: Path, tmp_path: Path) -> None:
    node_ids = [n.node_id for n in load_nodes(retrieval_dir / "structure" / "nodes.jsonl")]
    drafts = tmp_path / "drafts.jsonl"
    drafts.write_text(
        "".join(
            json.dumps(d) + "\n"
            for d in (
                _draft(1, ["testact:s3(1)"], 0.8),
                _draft(2, ["testact:s4(1)"]),
                _draft(3, ["testact:s5"]),
                _draft(4, ["testact:s1(1)"]),
            )
        ),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps(EXISTING) + "\n", encoding="utf-8")
    lines: list[str] = []
    answers = [
        "a",  # d0001 accept
        "e",  # d0002 edit
        "Edited question?",
        "",  # keep answer
        "testact:s9 testact:s4(2)",  # unknown id -> asked again
        "testact:s4(1), testact:s4(2)",
        "procedural",
        "hard",
        "a",
        "r",  # d0003 reject
        "x",  # invalid key
        "s",  # d0004 skip -> then input ends (quit)
    ]
    stats = review_drafts(drafts, questions, node_ids, scripted(answers), lines.append)
    assert stats == {"accepted": 2, "rejected": 1, "skipped": 1, "edited": 1}
    text = "\n".join(lines)
    assert "COPY-LIKE" in text and "unknown node id 'testact:s9'" in text
    assert "please type a, e, r, s or q" in text
    qs, errors = load_questions(questions)
    assert errors == [] and [q.id for q in qs] == ["q007", "q008", "q009"]
    q8, q9 = qs[1], qs[2]
    assert q8.origin == "llm_draft_human_reviewed" and q8.author == "Swapnil"
    assert q8.gold_node_ids == ["testact:s3(1)"] and "d0001" in (q8.notes or "")
    assert (q9.question, q9.type, q9.difficulty) == ("Edited question?", "procedural", "hard")
    assert q9.gold_node_ids == ["testact:s4(1)", "testact:s4(2)"] and "edited" in (q9.notes or "")
    status = [d["status"] for d in load_drafts(drafts)]
    assert status == ["accepted", "accepted", "rejected", "pending"]
    assert load_drafts(drafts)[0]["question_id"] == "q008"
    # quitting immediately keeps everything as it is
    assert (
        review_drafts(drafts, questions, node_ids, scripted(["q"]), lines.append)["accepted"] == 0
    )
    assert [d["status"] for d in load_drafts(drafts)][-1] == "pending"


def test_add_unanswerable_and_review_script(
    retrieval_dir: Path,
    tmp_path: Path,
    script: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps(EXISTING) + "\n", encoding="utf-8")
    seen: list[str] = []

    def search(q: str) -> list[tuple[str, str, str]]:
        seen.append(q)
        return [("c1", "Act > Section 3", "A complaint may be filed ...")]

    lines: list[str] = []
    answers = ["What is the GST rate on tea?", "Tax rates are not in the corpus.", "y", "easy"]
    answers += ["Who wrote the Act?", "Not stated.", "n", ""]
    assert add_unanswerable(questions, search, scripted(answers), lines.append) == 1
    assert seen == ["What is the GST rate on tea?", "Who wrote the Act?"]
    assert any("Top 5 BM25 hits" in x for x in lines) and any("not saved" in x for x in lines)
    qs, _ = load_questions(questions)
    new = qs[-1]
    assert (new.id, new.answerable, new.type, new.origin) == (
        "q008",
        False,
        "unanswerable",
        "human",
    )
    assert new.gold_node_ids == [] and new.gold_answer == "Tax rates are not in the corpus."
    assert next_question_id(questions) == "q009"

    review = script("review_drafts")
    # scripted stdin: input() reads these lines
    lines_in = ["Is there a fee for mediation in space?", "Not covered.", "y", "hard", "", ""]
    stdin = "\n".join(lines_in)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    args = ["--processed-dir", str(retrieval_dir), "--questions", str(questions)]
    assert review.main([*args, "--add-unanswerable", "--chunker", "structural"]) == 0
    out = capsys.readouterr().out
    assert "structural:" in out and "saved as q009" in out
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert review.main([*args, "--drafts", str(tmp_path / "none.jsonl")]) == 0
    assert "No pending drafts" in capsys.readouterr().out
