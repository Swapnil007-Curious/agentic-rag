"""Derived systems, computed from a base system's per-question records without LLM calls.

* guards (G3 = G2 + guards): the base answer is kept when the guards (recorded by the base run
  on that same answer) passed, and becomes an abstention (``guard_fail``) otherwise. Cost and
  latency are those of the base run (the guards take well under a millisecond).
* threshold (G5 = G4 + threshold): questions whose abstention score is below the threshold
  become abstentions (``low_retrieval_score``) whose cost is the retrieval only: LLM calls and
  tokens are removed and the latency is the pre-LLM part of the base run. Other questions keep
  the base record.
"""

from __future__ import annotations

import copy
from typing import Any

from ragsys.agents.config import GenerationSettings

PRE_LLM_STEPS = ("retrieval", "context")


def derivation_kind(settings: GenerationSettings, system: str) -> str:
    spec = settings.system(system)
    if spec.derived_from is None:
        raise ValueError(f"{system} is not a derived system")
    base = settings.system(spec.derived_from)
    return "threshold" if spec.threshold and not base.threshold else "guards"


def _relabel(rec: dict[str, Any], system: str, base: str) -> dict[str, Any]:
    out = copy.deepcopy(rec)
    out["system"] = system
    out["derived_from"] = base
    return out


def derive_guarded(rec: dict[str, Any], system: str, base: str) -> dict[str, Any]:
    out = _relabel(rec, system, base)
    out["derivation"] = {"kind": "guards"}
    if rec["outcome"] != "answered":
        return out
    guards = rec.get("guards") or {}
    if guards.get("passed", False):
        return out
    out.update(
        outcome="abstained",
        reason="guard_fail",
        detail="; ".join(guards.get("reasons", [])) or "guards failed",
        answer="",
        citations=[],
        sentences=[],
        cited_chunks=[],
        withheld_answer=rec.get("answer", ""),
    )
    return out


def derive_thresholded(
    rec: dict[str, Any], system: str, base: str, threshold: float
) -> dict[str, Any]:
    out = _relabel(rec, system, base)
    score = rec["retrieval"].get("abstention_score")
    below = score is None or float(score) < threshold
    out["derivation"] = {"kind": "threshold", "threshold": threshold, "below": below}
    out["retrieval"] = {**rec["retrieval"], "threshold": threshold}
    if not below:
        return out
    pre = [s for s in rec["steps"] if s["name"] in PRE_LLM_STEPS]
    pre_ms = sum(float(s["ms"]) for s in pre)
    pre.append(
        {
            "name": "threshold",
            "round": 0,
            "ms": 0.0,
            "status": "below",
            "detail": {"score": score, "threshold": threshold},
            "llm_calls": [],
        }
    )
    out.update(
        outcome="abstained",
        reason="low_retrieval_score",
        detail=f"top retrieval score {score} is below the abstention threshold {threshold:.4f}",
        answer="",
        citations=[],
        sentences=[],
        cited_chunks=[],
        withheld_answer="",
        rounds=[],
        guards=None,
        steps=pre,
        cost={
            "llm_calls": 0,
            "synth_calls": 0,
            "critic_calls": 0,
            "repair_calls": 0,
            "cached_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_ms": 0.0,
            "pre_llm_ms": round(pre_ms, 1),
            "e2e_ms": round(pre_ms, 1),
            "measured_ms": round(pre_ms, 1),
        },
    )
    # only the first-round context had been built at that point
    out["context"] = [c for c in rec["context"] if c.get("round_added", 1) == 1]
    return out


def derive_records(
    settings: GenerationSettings,
    system: str,
    base_records: list[dict[str, Any]],
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    base = settings.runnable_base(system)
    kind = derivation_kind(settings, system)
    if kind == "threshold":
        if threshold is None:
            raise ValueError(f"{system} needs the abstention threshold (run tune_abstention.py)")
        return [derive_thresholded(r, system, base, threshold) for r in base_records]
    return [derive_guarded(r, system, base) for r in base_records]
