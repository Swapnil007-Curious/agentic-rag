"""Validation of the evaluation set against the parsed corpus."""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ragsys.eval.schema import TARGET_ANSWERABLE, TARGET_UNANSWERABLE, EvalQuestion


@dataclass
class ValidationReport:
    questions: list[EvalQuestion] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_questions(path: Path) -> tuple[list[EvalQuestion], list[str]]:
    """Parse a JSONL file; returns valid questions and per-line error messages."""
    questions: list[EvalQuestion] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {lineno}: invalid JSON ({exc.msg})")
                continue
            try:
                questions.append(EvalQuestion.model_validate(data))
            except ValidationError as exc:
                qid = data.get("id", "?") if isinstance(data, dict) else "?"
                for err in exc.errors():
                    loc = ".".join(str(p) for p in err["loc"]) or "(root)"
                    errors.append(f"line {lineno} ({qid}): {loc}: {err['msg']}")
    return questions, errors


def suggest_ids(bad: str, node_ids: list[str], n: int = 5) -> list[str]:
    """Closest existing node IDs for a mistyped/unknown one."""
    out: list[str] = []
    id_set = set(node_ids)
    if ":" not in bad:
        out.extend(i for i in sorted(id_set) if i.split(":", 1)[1] == bad)
    # nearest existing ancestor, e.g. cpa2019:s35(9) -> cpa2019:s35 and its children
    cur = bad
    while True:
        m = re.match(r"^(.*?)(\([^()]*\)|-prov\d+|-expl\d+|~\d+)$", cur)
        if not m:
            break
        cur = m.group(1)
        if cur in id_set:
            out.append(cur)
            out.extend(
                sorted(
                    i
                    for i in id_set
                    if i.startswith(cur + "(") and i.count("(") == cur.count("(") + 1
                )[:3]
            )
            break
    out.extend(difflib.get_close_matches(bad, node_ids, n=n, cutoff=0.6))
    seen: list[str] = []
    for s in out:
        if s not in seen and s != bad:
            seen.append(s)
    return seen[:n]


def validate_questions(questions: list[EvalQuestion], node_ids: list[str]) -> ValidationReport:
    rep = ValidationReport(questions=questions)
    id_set = set(node_ids)
    dup = sorted(k for k, v in Counter(q.id for q in questions).items() if v > 1)
    for d in dup:
        rep.errors.append(f"duplicate question id {d}")
    for q in questions:
        for msg in q.consistency_errors():
            rep.errors.append(f"{q.id}: {msg}")
        for gid in q.gold_node_ids:
            if gid not in id_set:
                sug = suggest_ids(gid, node_ids)
                hint = f"; did you mean: {', '.join(sug)}" if sug else ""
                rep.errors.append(f"{q.id}: gold_node_id {gid!r} not found in nodes.jsonl{hint}")
        if q.author != "Swapnil":
            rep.warnings.append(f"{q.id}: author is {q.author!r}")
        if q.notes and "example" in q.notes.lower():
            rep.warnings.append(f"{q.id}: still marked as an example in notes")

    answerable = sum(1 for q in questions if q.answerable)
    unanswerable = sum(1 for q in questions if not q.answerable)
    docs: Counter[str] = Counter()
    for q in questions:
        qdocs = sorted({g.split(":", 1)[0] for g in q.gold_node_ids}) or ["(none)"]
        docs.update(qdocs)
    rep.counts = {
        "total": len(questions),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "by_type": dict(sorted(Counter(q.type for q in questions).items())),
        "by_difficulty": dict(sorted(Counter(q.difficulty for q in questions).items())),
        "by_document": dict(sorted(docs.items())),
        "target": {"answerable": TARGET_ANSWERABLE, "unanswerable": TARGET_UNANSWERABLE},
        "progress": {
            "answerable": f"{answerable}/{TARGET_ANSWERABLE}",
            "unanswerable": f"{unanswerable}/{TARGET_UNANSWERABLE}",
        },
    }
    return rep


def validate_file(questions_path: Path, nodes_path: Path) -> ValidationReport:
    questions, parse_errors = load_questions(questions_path)
    node_ids: list[str] = []
    with nodes_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                node_ids.append(json.loads(line)["node_id"])
    rep = validate_questions(questions, node_ids)
    rep.errors = parse_errors + rep.errors
    return rep


def format_report(rep: ValidationReport) -> str:
    c = rep.counts
    lines = [
        f"questions: {c.get('total', 0)}  (answerable {c.get('answerable', 0)}, "
        f"unanswerable {c.get('unanswerable', 0)})"
    ]
    for key in ("by_type", "by_difficulty", "by_document"):
        items = ", ".join(f"{k}={v}" for k, v in c.get(key, {}).items()) or "-"
        lines.append(f"{key.replace('_', ' ')}: {items}")
    prog = c.get("progress", {})
    lines.append(
        f"progress: answerable {prog.get('answerable')}, unanswerable {prog.get('unanswerable')}"
    )
    for w in rep.warnings:
        lines.append(f"WARNING: {w}")
    for e in rep.errors:
        lines.append(f"ERROR: {e}")
    lines.append("RESULT: " + ("OK" if rep.ok else f"FAILED ({len(rep.errors)} error(s))"))
    return "\n".join(lines)
