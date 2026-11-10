"""Threshold tuning on synthetic scores and the derived systems G3 / G5 (consistent with the
base records produced by the orchestrator)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_helpers import ScriptedLLM, StubRetriever, critic_reply, synth_reply
from ragsys.agents.abstention import evaluate_threshold, load_threshold, prf, tune_threshold
from ragsys.agents.config import GenerationSettings
from ragsys.agents.context import ContextBuilder
from ragsys.agents.derive import derive_records
from ragsys.agents.orchestrator import Orchestrator, PipelineSpec
from ragsys.retrieval.config import parse_config_id


def test_threshold_tuning_by_hand() -> None:
    # unanswerable questions (True) tend to have lower scores
    scores = [0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    unans = [True, True, False, True, False, False, False]
    res = tune_threshold(scores, unans)
    # t=0.50 abstains {0.30, 0.40, 0.45}: tp=2 fp=1 fn=1 -> P=R=F1=2/3
    # t=0.60 abstains 4: tp=3 fp=1 fn=0 -> P=.75 R=1 F1=6/7 (best)
    assert res["threshold"] == 0.60
    assert res["dev"]["f1"] == pytest.approx(6 / 7) and res["dev"]["recall"] == 1.0
    assert (res["n"], res["n_unanswerable"], len(res["curve"])) == (7, 3, 7)
    assert res["curve"][0]["n_abstained"] == 0 and res["curve"][0]["f1"] == 0.0
    assert evaluate_threshold(scores, unans, 0.5)["tp"] == 2
    assert prf(0, 0, 3) == (0.0, 0.0, 0.0)


def test_threshold_ties_go_to_the_lower_threshold() -> None:
    scores = [0.1, 0.2, 0.3, 0.4]
    unans = [True, False, True, False]
    # t=0.2: tp1 fp0 fn1 -> F1 2/3 ; t=0.4: tp2 fp1 fn0 -> F1 0.8 ; t=0.3: tp1 fp1 fn1 -> 0.5
    assert tune_threshold(scores, unans)["threshold"] == 0.4
    tie = tune_threshold([0.1, 0.2, 0.3, 0.4, 0.5], [True, False, False, True, False])
    # t=0.2: P1 R.5 -> F1 2/3 ; t=0.5: P.5 R1 -> F1 2/3 (tie) ; the lower threshold wins
    f1 = {row["threshold"]: row["f1"] for row in tie["curve"]}
    assert f1[0.2] == f1[0.5] == max(f1.values())
    assert tie["threshold"] == 0.2
    flat = tune_threshold([0.5, 0.5], [True, False])
    assert flat["threshold"] == 0.5 and flat["dev"]["n_abstained"] == 0
    with pytest.raises(ValueError):
        tune_threshold([], [])


def test_load_threshold_requires_dev(tmp_path: Path) -> None:
    p = tmp_path / "t.json"
    assert load_threshold(p) is None
    p.write_text(json.dumps({"threshold": 0.5, "split": "test"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_threshold(p)
    p.write_text(json.dumps({"threshold": 0.5, "split": "dev"}), encoding="utf-8")
    assert load_threshold(p) == {"threshold": 0.5, "split": "dev"}


SETTINGS = GenerationSettings.model_validate(
    {
        "systems": {
            "G1": {"description": "naive", "retrieval": "fixed_512|dense|bge-base"},
            "G2": {"description": "s", "retrieval": "structural|dense|bge-base"},
            "G3": {"description": "g", "derived_from": "G2", "guards": True},
            "G4": {
                "description": "c",
                "retrieval": "structural|dense|bge-base",
                "guards": True,
                "critic": True,
            },
            "G5": {
                "description": "t",
                "derived_from": "G4",
                "guards": True,
                "critic": True,
                "threshold": True,
            },
        },
        "ui": {"systems": ["G2", "G4", "G5"], "default_system": "G5"},
    }
)

GOOD = synth_reply("A complaint is decided within three months [C1].", ["C1"])
BAD = synth_reply("A complaint is decided within ninety days [C1].", ["C1"])


def _records(system: str, replies: dict[str, list[str]], scores: dict[str, float]) -> list[dict]:
    spec = PipelineSpec.from_settings(SETTINGS, system)
    out = []
    for q, synth in replies.items():
        llm = ScriptedLLM(synth=synth, critic=[critic_reply("accept")])
        o = Orchestrator(spec, StubRetriever(scores=scores), llm, ContextBuilder(250, 450))  # type: ignore[arg-type]
        rec = o.answer(q).to_record()
        rec["qid"] = q
        out.append(rec)
    return out


def test_settings_validate_derivations() -> None:
    with pytest.raises(ValueError):
        GenerationSettings.model_validate(
            {
                "systems": {
                    "A": {"description": "x", "retrieval": "structural|dense|bge-base"},
                    "B": {"description": "y", "derived_from": "A", "critic": True, "guards": True},
                },
                "ui": {"systems": ["A"], "default_system": "A"},
                "eval": {"reference_system": "A"},
                "abstention": {"tuned_system": "A"},
            }
        )
    assert (
        SETTINGS.runnable_base("G5") == "G4"
        and SETTINGS.retrieval_id("G3") == "structural|dense|bge-base"
    )


def test_g3_is_g2_plus_guards() -> None:
    g2 = _records("G2", {"qa": [GOOD], "qb": [BAD], "qc": [synth_reply(abstain=True)]}, {})
    assert [r["outcome"] for r in g2] == ["answered", "answered", "abstained"]
    g3 = derive_records(SETTINGS, "G3", g2)
    assert [r["outcome"] for r in g3] == ["answered", "abstained", "abstained"]
    assert [r["reason"] for r in g3] == [None, "guard_fail", "synth_abstain"]
    assert g3[0]["answer"] == g2[0]["answer"] and g3[1]["withheld_answer"] == g2[1]["answer"]
    assert all(a["cost"] == b["cost"] for a, b in zip(g2, g3, strict=True))  # no new LLM calls
    assert all(r["system"] == "G3" and r["derived_from"] == "G2" for r in g3)
    answered_g2 = {r["qid"] for r in g2 if r["outcome"] == "answered"}
    assert {r["qid"] for r in g3 if r["outcome"] == "answered"} <= answered_g2
    # a live G3 (guards only, no critic) reaches the same outcomes
    live = _records("G3", {"qa": [GOOD], "qb": [BAD]}, {})
    assert [(r["outcome"], r["reason"]) for r in live] == [
        ("answered", None),
        ("abstained", "guard_fail"),
    ]


def test_g5_is_g4_plus_threshold() -> None:
    scores = {"qa": 0.82, "qb": 0.41}
    g4 = _records("G4", {"qa": [GOOD], "qb": [GOOD]}, scores)
    assert all(r["outcome"] == "answered" and r["cost"]["llm_calls"] == 2 for r in g4)
    g5 = derive_records(SETTINGS, "G5", g4, threshold=0.5)
    above, below = g5
    assert above["outcome"] == "answered" and above["answer"] == g4[0]["answer"]
    assert above["cost"] == g4[0]["cost"] and above["retrieval"]["threshold"] == 0.5
    assert (below["outcome"], below["reason"]) == ("abstained", "low_retrieval_score")
    assert below["cost"]["llm_calls"] == 0 and below["cost"]["prompt_tokens"] == 0
    assert below["cost"]["e2e_ms"] == pytest.approx(g4[1]["cost"]["pre_llm_ms"], abs=0.11)
    assert [s["name"] for s in below["steps"]] == ["retrieval", "context", "threshold"]
    with pytest.raises(ValueError):
        derive_records(SETTINGS, "G5", g4)  # threshold required
    # a live G5 with the same threshold gives the same outcomes and costs no LLM call below it
    live_spec = PipelineSpec.from_settings(SETTINGS, "G5", threshold=0.5)
    llm = ScriptedLLM(synth=[GOOD], critic=[critic_reply()])
    live = Orchestrator(
        live_spec, StubRetriever(scores=scores), llm, ContextBuilder(250, 450)
    ).answer("qb")  # type: ignore[arg-type]
    assert (live.outcome, live.reason, len(llm.calls)) == ("abstained", "low_retrieval_score", 0)
    assert PipelineSpec.from_settings(SETTINGS, "G4", threshold=0.5).threshold is None
    assert PipelineSpec.from_settings(SETTINGS, "G5", threshold=0.5).retrieval == parse_config_id(
        "structural|dense|bge-base"
    )
