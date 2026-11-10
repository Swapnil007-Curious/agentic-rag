"""Orchestrator: loop bounds, every abstain path, revision with extended context, the threshold
before any LLM call, cached-latency reuse (client and orchestrator)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_helpers import ScriptedLLM, StubRetriever, critic_reply, synth_reply
from ragsys.agents.config import LoopSettings
from ragsys.agents.context import ContextBuilder
from ragsys.agents.orchestrator import Orchestrator, PipelineSpec, abstention_score
from ragsys.retrieval.config import parse_config_id
from test_llm_client import MSGS, make_client

GOOD = synth_reply("A complaint is decided within three months [C1].", ["C1"])
GOOD2 = synth_reply("Decided within five months when testing is needed [C2].", ["C2"])
NO_CITE = synth_reply("A complaint is decided within three months.", [])
BAD_NUMBER = synth_reply("A complaint is decided within ninety days [C1].", ["C1"])


def orch(
    llm: ScriptedLLM,
    *,
    guards: bool = True,
    critic: bool = True,
    threshold: float | None = None,
    loop: LoopSettings | None = None,
    retriever: StubRetriever | None = None,
) -> Orchestrator:
    spec = PipelineSpec(
        "GX", parse_config_id("structural|dense|bge-base"), guards, critic, threshold
    )
    return Orchestrator(
        spec,
        retriever or StubRetriever(),  # type: ignore[arg-type]
        llm,
        ContextBuilder(250, 450),  # 2 passages in round 1, 4 after the extension
        loop,
    )


def test_synthesizer_only_records_guards_without_acting() -> None:
    llm = ScriptedLLM(synth=[NO_CITE])
    r = orch(llm, guards=False, critic=False).answer("q")
    assert r.outcome == "answered" and r.reason is None and r.answer.startswith("A complaint")
    assert r.final_guards is not None and not r.final_guards.passed
    assert (r.synth_calls, r.critic_calls, len(r.llm_calls)) == (1, 0, 1)
    rec = r.to_record()
    assert rec["guards"]["status"] == "unsupported" and rec["cost"]["llm_calls"] == 1


def test_guards_only_abstains_on_failure() -> None:
    r = orch(ScriptedLLM(synth=[BAD_NUMBER]), critic=False).answer("q")
    assert (r.outcome, r.reason) == ("abstained", "guard_fail")
    assert r.answer == "" and "ninety" in r.withheld_answer and "number" in r.detail
    r2 = orch(ScriptedLLM(synth=[GOOD]), critic=False).answer("q")
    assert r2.outcome == "answered" and r2.synth_calls == 1


def test_accept_path_two_calls() -> None:
    llm = ScriptedLLM(synth=[GOOD], critic=[critic_reply("accept")])
    r = orch(llm).answer("q")
    assert r.outcome == "answered" and (r.synth_calls, r.critic_calls) == (1, 1)
    critic_user = llm.calls[1][1][1]["content"]
    assert "<<<PASSAGE C1>>>" in critic_user and "<<<PASSAGE C2>>>" not in critic_user  # cited only
    assert [s["name"] for s in r.to_record()["steps"]] == [
        "retrieval",
        "context",
        "synthesis",
        "guards",
        "critic",
    ]


def test_guard_failure_triggers_one_revision_with_extended_context() -> None:
    llm = ScriptedLLM(synth=[NO_CITE, GOOD2], critic=[critic_reply("accept")])
    r = orch(llm).answer("q")
    assert r.outcome == "answered" and r.answer.startswith("Decided within five")
    # round 1 guards failed -> critic skipped, revision, guards + critic again
    assert (r.synth_calls, r.critic_calls) == (2, 1)
    assert [x.round for x in r.rounds] == [1, 2]
    assert r.rounds[0].context["labels"] == ["C1", "C2"]
    assert r.context.labels == ["C1", "C2", "C3", "C4"]
    revision_user = llm.calls[1][1][1]["content"]
    assert "Your previous answer was" in revision_user and "no citation" in revision_user
    assert "<<<PASSAGE C4>>>" in revision_user and "(C3-C4)" in revision_user
    names = [s.name for s in r.steps]
    assert names.index("revision_context") < names.index("synthesis", names.index("guards"))


def test_critic_revise_then_accept_uses_feedback() -> None:
    llm = ScriptedLLM(
        synth=[GOOD, GOOD2],
        critic=[
            critic_reply(
                "revise", ["A complaint is decided within three months."], "mention testing"
            ),
            critic_reply("accept"),
        ],
    )
    r = orch(llm).answer("q")
    assert r.outcome == "answered" and (r.synth_calls, r.critic_calls, len(r.llm_calls)) == (
        2,
        2,
        4,
    )
    revision_user = llm.calls[2][1][1]["content"]
    assert "reviewer feedback: mention testing" in revision_user
    assert "unsupported sentence: A complaint is decided" in revision_user


def test_loop_bounds_and_final_revise_abstains() -> None:
    llm = ScriptedLLM(synth=[GOOD], critic=[critic_reply("revise", [], "still wrong")])
    r = orch(llm).answer("q")
    assert (r.outcome, r.reason) == ("abstained", "critic_abstain")
    assert (r.synth_calls, r.critic_calls) == (2, 2) and len(llm.calls) == 4
    assert "after the revision" in r.detail
    released = orch(
        ScriptedLLM(synth=[GOOD], critic=[critic_reply("revise")]),
        loop=LoopSettings(on_final_revise="answer"),
    ).answer("q")
    assert released.outcome == "answered" and released.warnings
    single = orch(
        ScriptedLLM(synth=[GOOD], critic=[critic_reply("revise")]),
        loop=LoopSettings(max_synth_calls=1),
    ).answer("q")
    assert single.reason == "critic_abstain" and single.synth_calls == 1


def test_every_abstain_path() -> None:
    r = orch(ScriptedLLM(synth=[synth_reply(abstain=True)])).answer("q")
    assert (r.reason, len(r.llm_calls)) == ("synth_abstain", 1)
    r = orch(ScriptedLLM(synth=["garbage", "still garbage"])).answer("q")
    assert (
        r.reason == "synth_abstain" and "unreadable" in r.detail and r.cost()["repair_calls"] == 1
    )
    r = orch(ScriptedLLM(synth=[GOOD], critic=[critic_reply("abstain", [], "not covered")])).answer(
        "q"
    )
    assert (r.reason, r.detail, r.critic_calls) == ("critic_abstain", "not covered", 1)
    r = orch(ScriptedLLM(synth=[NO_CITE, BAD_NUMBER], critic=[critic_reply("accept")])).answer("q")
    assert (r.reason, r.synth_calls, r.critic_calls) == ("guard_fail", 2, 0)
    r = orch(ScriptedLLM(synth=[NO_CITE, synth_reply(abstain=True)])).answer("q")
    assert r.reason == "synth_abstain" and r.synth_calls == 2
    llm = ScriptedLLM(synth=[GOOD])
    retr = StubRetriever(scores={"q": 0.42})
    r = orch(llm, threshold=0.5, retriever=retr).answer("q")
    assert (r.reason, len(llm.calls), r.abstention_score) == ("low_retrieval_score", 0, 0.42)
    assert [s.name for s in r.steps] == ["retrieval", "context", "threshold"]
    assert r.cost()["llm_calls"] == 0 and r.e2e_ms == pytest.approx(r.pre_llm_ms, abs=50)
    above = orch(ScriptedLLM(synth=[GOOD], critic=[critic_reply()]), threshold=0.5).answer("q")
    assert above.outcome == "answered" and above.steps[2].status == "pass"


def test_unreadable_critic_counts_as_no_objection() -> None:
    llm = ScriptedLLM(synth=[GOOD], critic=["nope", "nope again"])
    r = orch(llm).answer("q")
    assert r.outcome == "answered" and r.critic_calls == 1 and len(r.llm_calls) == 3
    assert r.warnings and "unreadable" in r.warnings[0]


def test_cached_llm_latency_is_reported_not_zero() -> None:
    llm = ScriptedLLM(synth=[GOOD], critic=[critic_reply()], cached_latency_ms=4000.0)
    r = orch(llm).answer("q")
    assert all(c.cached and c.ms == 4000.0 and c.measured_ms < 1000 for c in r.llm_calls)
    assert r.e2e_ms >= 8000.0 and r.measured_ms < 2000
    synth_step = next(s for s in r.steps if s.name == "synthesis")
    assert synth_step.ms >= 4000.0
    # retrieval reports the retriever's (cache-adjusted) total
    assert r.steps[0].ms == 7.5
    assert r.cost()["cached_calls"] == 2 and r.cost()["prompt_tokens"] == 200


def test_client_cache_hit_carries_original_latency(tmp_path: Path) -> None:
    client, transport, _ = make_client(tmp_path, ['{"a": 1}'])
    first = client.chat(MSGS)
    assert not first.cached and first.reported_latency_ms == first.latency_ms
    second = client.chat(MSGS)
    assert second.cached and second.latency_ms == 0.0  # unchanged meaning: no model time now
    assert second.stored_latency_ms == pytest.approx(first.latency_ms)
    assert second.reported_latency_ms == pytest.approx(first.latency_ms)
    assert client.calls[1]["reported_latency_ms"] == pytest.approx(first.latency_ms)
    assert len(transport.requests) == 1


def test_abstention_score_choice() -> None:
    retr = StubRetriever(scores={"q": 0.77})
    res = retr.search("q")
    assert abstention_score(res, parse_config_id("structural|dense|bge-base")) == 0.77
    res.ranked[0].rerank_score = 3.5
    assert (
        abstention_score(
            res, parse_config_id("fixed_256|hybrid_linear|bge-base|a0.5|rerank-minilm")
        )
        == 3.5
    )
