"""Optional manual spot check of automatically generated evaluation items.

A seeded random sample of ``llm_auto_filtered`` items, stratified into answerable and
unanswerable items (proportional allocation, at least one per non-empty stratum), is shown one
by one; the person answers y (correct), n (wrong) or s (skip). The result records the counts,
the error rate among judged items with a 95% Wilson score interval, and the ids marked wrong.
Removing the wrong items from questions.jsonl is a separate, explicit step.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ragsys.eval.drafting import allocate
from ragsys.eval.schema import AUTO_ORIGIN, EvalQuestion

Z95 = 1.959963984540054
STRATA = ("answerable", "unanswerable")
InputFn = Callable[[str], str]
OutFn = Callable[[str], None]
KEYS = {"y": "correct", "n": "wrong", "s": "skipped"}


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion k/n (None when n == 0)."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def stratum(q: EvalQuestion) -> str:
    return "answerable" if q.answerable else "unanswerable"


def sample_items(questions: Sequence[EvalQuestion], n: int, seed: int) -> list[EvalQuestion]:
    """Seeded sample of automatic items, stratified answerable/unanswerable."""
    auto = [q for q in questions if q.origin == AUTO_ORIGIN]
    pools = {s: sorted((q for q in auto if stratum(q) == s), key=lambda q: q.id) for s in STRATA}
    n = min(n, len(auto))
    sizes = {s: len(p) for s, p in pools.items()}
    alloc = allocate(n, sizes) if n else dict.fromkeys(STRATA, 0)
    nonempty = [s for s in STRATA if sizes[s] > 0]
    if n >= len(nonempty):  # every non-empty stratum gets at least one item
        for s in nonempty:
            if alloc[s] == 0:
                donor = max(STRATA, key=lambda x: alloc[x])
                alloc[donor] -= 1
                alloc[s] = 1
    rng = random.Random(f"spot-check-{seed}")
    picked: list[EvalQuestion] = []
    for s in STRATA:
        picked += rng.sample(pools[s], min(alloc[s], sizes[s]))
    rng.shuffle(picked)
    return picked


class QuitSpotCheck(Exception):
    pass


def run_session(
    items: Sequence[EvalQuestion],
    render: Callable[[EvalQuestion], list[str]],
    input_fn: InputFn = input,
    out: OutFn = print,
) -> dict[str, str]:
    """Ask y/n/s (q quits) for each item; returns {question id: correct|wrong|skipped}."""
    judgments: dict[str, str] = {}
    out(f"{len(items)} item(s). Keys: y = correct, n = wrong, s = skip, q = quit")
    try:
        for pos, q in enumerate(items, start=1):
            out("=" * 100)
            out(f"[{pos}/{len(items)}] {q.id}  {stratum(q)}  type: {q.type}")
            for line in render(q):
                out(line)
            while True:
                try:
                    key = input_fn("[y] correct  [n] wrong  [s] skip  [q] quit > ").strip().lower()
                except EOFError as exc:
                    raise QuitSpotCheck from exc
                if key == "q":
                    raise QuitSpotCheck
                if key in KEYS:
                    judgments[q.id] = KEYS[key]
                    break
                out("please type y, n, s or q")
    except QuitSpotCheck:
        out("quit")
    return judgments


def _counts(ids: Sequence[str], judgments: dict[str, str]) -> dict[str, Any]:
    c = {v: sum(1 for i in ids if judgments.get(i) == v) for v in ("correct", "wrong", "skipped")}
    judged = c["correct"] + c["wrong"]
    ci = wilson_interval(c["wrong"], judged)
    return {
        **c,
        "not_judged": sum(1 for i in ids if i not in judgments),
        "judged": judged,
        "error_rate": round(c["wrong"] / judged, 4) if judged else None,
        "wilson_95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
    }


def summarise(
    items: Sequence[EvalQuestion], judgments: dict[str, str], **meta: Any
) -> dict[str, Any]:
    ids = [q.id for q in items]
    overall = _counts(ids, judgments)
    return {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        **meta,
        "n_sampled": len(items),
        "sample": [
            {"id": q.id, "stratum": stratum(q), "type": q.type, "judgment": judgments.get(q.id)}
            for q in items
        ],
        "counts": {k: overall[k] for k in ("correct", "wrong", "skipped", "not_judged")},
        "judged": overall["judged"],
        "error_rate": overall["error_rate"],
        "wilson_95": overall["wilson_95"],
        "by_stratum": {
            s: _counts([q.id for q in items if stratum(q) == s], judgments) for s in STRATA
        },
        "ids_wrong": [i for i in ids if judgments.get(i) == "wrong"],
        "wrong_items": [
            {"id": q.id, "question": q.question} for q in items if judgments.get(q.id) == "wrong"
        ],
        "ids_correct": [i for i in ids if judgments.get(i) == "correct"],
        "ids_skipped": [i for i in ids if judgments.get(i) == "skipped"],
        "removals_applied": False,
    }


def remove_auto_items(path: Path, wanted: dict[str, str | None]) -> list[str]:
    """Remove items from a questions file: only llm_auto_filtered items whose id is in
    ``wanted`` and whose question text equals the recorded one (None: any text), so ids that
    now belong to other items (e.g. after a rebuild) are left alone. Every other line is kept
    byte-for-byte. Returns the removed ids."""
    keep: list[str] = []
    removed: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            keep.append(line)
            continue
        qid = data.get("id") if isinstance(data, dict) else None
        if (
            isinstance(data, dict)
            and qid in wanted
            and data.get("origin") == AUTO_ORIGIN
            and wanted[str(qid)] in (None, data.get("question"))
        ):
            removed.append(str(data["id"]))
            continue
        keep.append(line)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(x + "\n" for x in keep), encoding="utf-8", newline="\n")
    tmp.replace(path)
    return removed
