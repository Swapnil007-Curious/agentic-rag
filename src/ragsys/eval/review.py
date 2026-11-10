"""Human review of drafted questions, and manual entry of unanswerable questions.

Both loops take ``input_fn``/``out`` callables so they can be driven by a terminal or by a
script of answers in tests. Progress is saved after every action.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ragsys.eval.drafting import OVERLAP_FLAG_THRESHOLD, load_drafts, save_drafts
from ragsys.eval.schema import Difficulty, EvalQuestion
from ragsys.eval.validate import load_questions, suggest_ids

AUTHOR = "Swapnil"
ANSWERABLE_TYPES = (
    "definition",
    "factual",
    "procedural",
    "numeric",
    "cross_reference",
    "multi_section",
)
DIFFICULTIES = ("easy", "medium", "hard")
_DIFFICULTY: dict[str, Difficulty] = {"easy": "easy", "medium": "medium", "hard": "hard"}

InputFn = Callable[[str], str]
OutFn = Callable[[str], None]


class QuitReview(Exception):
    pass


def _ask(input_fn: InputFn, prompt: str) -> str:
    try:
        return input_fn(prompt)
    except EOFError as exc:  # end of scripted input / Ctrl-Z
        raise QuitReview from exc


def next_question_id(path: Path) -> str:
    ids: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    ids.append(str(json.loads(line).get("id", "")))
                except json.JSONDecodeError:
                    continue
    nums = [int(m.group(1)) for i in ids if (m := re.fullmatch(r"q(\d+)", i))]
    return f"q{max(nums, default=0) + 1:03d}"


def append_question(path: Path, q: EvalQuestion) -> None:
    errs = q.consistency_errors()
    if errs:
        raise ValueError(f"{q.id}: {'; '.join(errs)}")
    existing, _errors = load_questions(path) if path.exists() else ([], [])
    if any(e.id == q.id for e in existing):
        raise ValueError(f"question id {q.id} already exists")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    line = json.dumps(q.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        fh.write(line + "\n")


def _wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(
        textwrap.fill(p, width=100, initial_indent=indent, subsequent_indent=indent)
        for p in text.splitlines()
        if p.strip()
    )


def show_draft(d: dict[str, Any], pos: int, total: int, out: OutFn) -> None:
    overlap = float(d.get("lexical_overlap", 0.0))
    flag = (
        f"  << COPY-LIKE (> {OVERLAP_FLAG_THRESHOLD:.1f})"
        if overlap > OVERLAP_FLAG_THRESHOLD
        else ""
    )
    out("=" * 100)
    out(f"draft {d['draft_id']}  ({pos}/{total} pending)   model: {d.get('llm_model')}")
    out(f"type: {d['type']}   difficulty: {d['difficulty']}")
    out(f"question:    {d['question']}")
    out(f"gold_answer: {d['gold_answer']}")
    out(f"gold_node_ids: {', '.join(d['gold_node_ids'])}")
    out(f"lexical overlap with source: {overlap:.2f}{flag}")
    if d.get("flags"):
        out(f"flags: {', '.join(d['flags'])}")
    out("source text:")
    out(_wrap(d.get("source_text", "")))
    out("-" * 100)


def _edit(d: dict[str, Any], node_ids: set[str], input_fn: InputFn, out: OutFn) -> None:
    out("Edit (press Enter to keep the current value).")
    q = _ask(input_fn, "question: ").strip()
    if q:
        d["question"] = q
    a = _ask(input_fn, "gold_answer: ").strip()
    if a:
        d["gold_answer"] = a
    while True:
        g = _ask(input_fn, "gold_node_ids (comma or space separated): ").strip()
        if not g:
            break
        ids = [x for x in re.split(r"[,\s]+", g) if x]
        bad = [x for x in ids if x not in node_ids]
        if not bad:
            d["gold_node_ids"] = list(dict.fromkeys(ids))
            break
        for b in bad:
            sug = suggest_ids(b, sorted(node_ids))
            out(f"  unknown node id {b!r}" + (f"; did you mean: {', '.join(sug)}" if sug else ""))
    while True:
        t = _ask(input_fn, f"type {ANSWERABLE_TYPES}: ").strip()
        if not t or t in ANSWERABLE_TYPES:
            d["type"] = t or d["type"]
            break
        out(f"  type must be one of {ANSWERABLE_TYPES}")
    while True:
        df = _ask(input_fn, "difficulty (easy/medium/hard): ").strip().lower()
        if not df or df in DIFFICULTIES:
            d["difficulty"] = df or d["difficulty"]
            break
        out("  difficulty must be easy, medium or hard")
    d["edited"] = True


def _accept(d: dict[str, Any], questions_path: Path, node_ids: set[str]) -> str:
    missing = [g for g in d["gold_node_ids"] if g not in node_ids]
    if missing:
        raise ValueError(f"gold node(s) not in nodes.jsonl: {missing}; edit the draft first")
    qid = next_question_id(questions_path)
    note = f"from draft {d['draft_id']} ({d.get('llm_model')})" + (
        ", edited" if d.get("edited") else ""
    )
    q = EvalQuestion(
        id=qid,
        question=d["question"],
        type=d["type"],
        answerable=True,
        gold_node_ids=d["gold_node_ids"],
        gold_answer=d["gold_answer"],
        difficulty=d["difficulty"],
        author=AUTHOR,
        notes=note,
        origin="llm_draft_human_reviewed",
    )
    append_question(questions_path, q)
    return qid


def review_drafts(
    drafts_path: Path,
    questions_path: Path,
    node_ids: Sequence[str],
    input_fn: InputFn = input,
    out: OutFn = print,
) -> dict[str, int]:
    """Interactive review loop over pending drafts. Returns counts of actions taken."""
    drafts = load_drafts(drafts_path)
    ids = set(node_ids)
    stats = {"accepted": 0, "rejected": 0, "skipped": 0, "edited": 0}
    pending = [d for d in drafts if d.get("status") == "pending"]
    if not pending:
        out(f"No pending drafts in {drafts_path}. Create some with scripts/draft_questions.py.")
        return stats
    out(f"{len(pending)} pending draft(s). Keys: a=accept  e=edit  r=reject  s=skip  q=quit")
    try:
        for pos, d in enumerate(pending, start=1):
            while True:
                show_draft(d, pos, len(pending), out)
                key = _ask(input_fn, "[a]ccept [e]dit [r]eject [s]kip [q]uit > ").strip().lower()
                now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
                if key == "a":
                    try:
                        qid = _accept(d, questions_path, ids)
                    except (ValueError, ValidationError) as exc:
                        out(f"cannot accept: {exc}")
                        continue
                    d.update(status="accepted", question_id=qid, reviewed_utc=now)
                    stats["accepted"] += 1
                    out(f"accepted as {qid} -> {questions_path}")
                elif key == "e":
                    _edit(d, ids, input_fn, out)
                    stats["edited"] += 1
                    save_drafts(drafts_path, drafts)
                    continue  # show the edited draft again
                elif key == "r":
                    d.update(status="rejected", reviewed_utc=now)
                    stats["rejected"] += 1
                elif key == "s":
                    stats["skipped"] += 1
                elif key == "q":
                    raise QuitReview
                else:
                    out("please type a, e, r, s or q")
                    continue
                save_drafts(drafts_path, drafts)
                break
    except QuitReview:
        out("quit")
    save_drafts(drafts_path, drafts)
    left = sum(1 for d in drafts if d.get("status") == "pending")
    out(
        f"accepted {stats['accepted']}, rejected {stats['rejected']}, skipped {stats['skipped']}; "
        f"{left} draft(s) still pending. Run scripts/validate_eval.py to check the set."
    )
    return stats


def add_unanswerable(
    questions_path: Path,
    search: Callable[[str], list[tuple[str, str, str]]],
    input_fn: InputFn = input,
    out: OutFn = print,
) -> int:
    """Type unanswerable questions; each is checked against the top BM25 hits before saving.

    ``search(query)`` returns (chunk_id, heading_path, snippet) tuples. Returns the number of
    questions added."""
    added = 0
    out("Add unanswerable questions (questions the corpus does not answer). Blank question = stop.")
    try:
        while True:
            q = _ask(input_fn, "unanswerable question: ").strip()
            if not q:
                break
            reason = _ask(input_fn, "why the corpus cannot answer it (one line): ").strip()
            if not reason:
                out("a reason is required; question discarded")
                continue
            out("Top 5 BM25 hits -- check that none of them answers the question:")
            for i, (cid, path, snip) in enumerate(search(q)[:5], start=1):
                out(f"  {i}. {cid}  {path}")
                out(_wrap(snip, "       "))
            ok = _ask(input_fn, "Confirm the corpus does NOT answer this question [y/N]: ")
            if ok.strip().lower() != "y":
                out("not saved")
                continue
            df = _ask(input_fn, "difficulty (easy/medium/hard) [medium]: ").strip().lower()
            difficulty = _DIFFICULTY.get(df, "medium")
            qid = next_question_id(questions_path)
            append_question(
                questions_path,
                EvalQuestion(
                    id=qid,
                    question=q,
                    type="unanswerable",
                    answerable=False,
                    gold_node_ids=[],
                    gold_answer=reason,
                    difficulty=difficulty,
                    author=AUTHOR,
                    origin="human",
                ),
            )
            added += 1
            out(f"saved as {qid}")
    except QuitReview:
        pass
    out(f"added {added} unanswerable question(s) to {questions_path}")
    return added
