"""Retrieval-score abstention: threshold tuning on dev and the threshold file.

A question is abstained when its abstention score (see
:func:`ragsys.agents.orchestrator.abstention_score`) is strictly below the threshold. Candidate
thresholds are the distinct dev scores; the chosen one maximises abstention F1 with
"unanswerable" as the positive class, ties broken towards the lower threshold (fewer
abstentions). Because candidates are observed scores, the question with the highest score is
never abstained, so "abstain on everything" is not a candidate.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall and F1 (0 when undefined)."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate_threshold(
    scores: Sequence[float], unanswerable: Sequence[bool], threshold: float
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for s, u in zip(scores, unanswerable, strict=True):
        abstain = s < threshold
        if abstain and u:
            tp += 1
        elif abstain:
            fp += 1
        elif u:
            fn += 1
        else:
            tn += 1
    p, r, f = prf(tp, fp, fn)
    return {
        "threshold": threshold,
        "precision": p,
        "recall": r,
        "f1": f,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_abstained": tp + fp,
    }


def tune_threshold(scores: Sequence[float], unanswerable: Sequence[bool]) -> dict[str, Any]:
    """Best threshold by abstention F1 (ties -> lower threshold) plus the full curve."""
    if len(scores) != len(unanswerable):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("no scores to tune on")
    curve = [evaluate_threshold(scores, unanswerable, t) for t in sorted(set(scores))]
    best = curve[0]
    for row in curve[1:]:
        if row["f1"] > best["f1"]:  # strictly better only: ties keep the lower threshold
            best = row
    return {
        "threshold": best["threshold"],
        "dev": {k: v for k, v in best.items() if k != "threshold"},
        "n": len(scores),
        "n_unanswerable": sum(1 for u in unanswerable if u),
        "curve": curve,
    }


def load_threshold(path: Path) -> dict[str, Any] | None:
    """The threshold file written by scripts/tune_abstention.py, or None when missing."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "threshold" not in data:
        raise ValueError(f"{path} is not an abstention threshold file")
    if data.get("split") != "dev":
        raise ValueError(f"{path}: the threshold must be tuned on dev, found {data.get('split')!r}")
    return data
