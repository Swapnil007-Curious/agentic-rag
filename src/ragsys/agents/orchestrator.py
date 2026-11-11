"""Orchestrator: retrieve -> (threshold) -> synthesize -> guards -> critic, with at most one
revision round.

Flow for one question (systems without the critic stop after the guards):

1. retrieval and context packing (``budget_tokens``); when a threshold is set and the
   abstention score is below it, abstain (``low_retrieval_score``) before any LLM call;
2. synthesis; an abstention (or an unreadable reply after one JSON repair) ends with
   ``synth_abstain``;
3. deterministic guards. Synthesizer-only systems record them without acting on them;
   guard-only systems abstain on failure (``guard_fail``);
4. critic (when the guards passed, unless ``critic_on_guard_fail``): ``abstain`` ends with
   ``critic_abstain``; ``accept`` with passing guards releases the answer;
5. otherwise one revision round: the context is extended with the next chunks up to
   ``revision_budget_tokens``, the synthesizer gets the guard reasons and the critic feedback,
   and guards + critic run again. Failing guards then give ``guard_fail``; a critic that still
   says ``revise`` gives ``critic_abstain`` (or releases the answer when
   ``on_final_revise: answer``).

At most ``max_synth_calls`` (2) synthesis rounds and ``max_critic_calls`` (2) critic rounds run
per question; each round may use one JSON-repair retry, counted in the LLM-call totals. Every
step is recorded with its timing; LLM steps report the latency measured when a cached response
was first computed, so end-to-end latency does not depend on the cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ragsys.agents.config import GenerationSettings, LoopSettings
from ragsys.agents.context import ContextBuilder, PackedContext
from ragsys.agents.critic import Critic, CriticOutput
from ragsys.agents.guards import GuardReport, cited_labels, run_guards
from ragsys.agents.llmcall import LLMCall
from ragsys.agents.synthesizer import Revision, Synthesizer, SynthOutput
from ragsys.agents.text import split_sentences
from ragsys.llm.client import ChatModel
from ragsys.profiles import GenerationPrompts
from ragsys.retrieval.config import RetrievalConfig
from ragsys.retrieval.pipeline import RetrievalResult, Retriever

REASONS = ("synth_abstain", "guard_fail", "critic_abstain", "low_retrieval_score")
PRE_LLM_STEPS = ("retrieval", "context", "threshold")


@dataclass
class PipelineSpec:
    system: str
    retrieval: RetrievalConfig
    guards: bool = False  # act on guard failures
    critic: bool = False  # critic + revise loop
    threshold: float | None = None  # retrieval-score abstention (None = off)

    @classmethod
    def from_settings(
        cls,
        settings: GenerationSettings,
        system: str,
        threshold: float | None = None,
        retrieval: RetrievalConfig | None = None,
    ) -> PipelineSpec:
        spec = settings.system(system)
        return cls(
            system=system,
            retrieval=retrieval or settings.retrieval_cfg(system),
            guards=spec.guards,
            critic=spec.critic,
            threshold=threshold if spec.threshold else None,
        )


def abstention_score(res: RetrievalResult, cfg: RetrievalConfig) -> float | None:
    """Top-1 rerank score when reranking is on; otherwise the top-1 dense cosine of the first
    stage (the dense component of hybrid rankings; the fused score for BM25-only rankings)."""
    if cfg.rerank:
        return res.top1_rerank_score
    for r in res.ranked:
        comp = r.components.get("dense")
        if comp is not None and comp.rank == 1:
            return float(comp.score)
    return res.top1_first_stage_score


@dataclass
class Step:
    name: str  # retrieval | context | threshold | synthesis | guards | critic | revision_context
    round: int
    ms: float
    status: str
    detail: dict[str, Any] = field(default_factory=dict)
    calls: list[LLMCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "round": self.round,
            "ms": round(self.ms, 1),
            "status": self.status,
            "detail": self.detail,
            "llm_calls": [c.to_dict() for c in self.calls],
        }


@dataclass
class RoundRecord:
    round: int
    context: dict[str, Any]
    synth: SynthOutput | None = None
    synth_error: str | None = None
    guards: GuardReport | None = None
    critic: CriticOutput | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "context": self.context,
            "synth": self.synth.to_dict() if self.synth else None,
            "synth_error": self.synth_error,
            "guards": self.guards.to_dict() if self.guards else None,
            "critic": self.critic.to_dict() if self.critic else None,
        }


@dataclass
class AnswerResult:
    question: str
    system: str
    plain: bool
    outcome: str  # "answered" | "abstained"
    reason: str | None  # one of REASONS when abstained
    detail: str
    answer: str  # empty when abstained
    citations: list[str]
    withheld_answer: str  # the last generated answer when it was not released
    context: PackedContext  # the context of the last round
    retrieval: RetrievalResult
    abstention_score: float | None
    threshold: float | None
    rounds: list[RoundRecord]
    steps: list[Step]
    e2e_ms: float  # measured wall time with stored latencies for cache hits
    measured_ms: float  # plain wall time of this call
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ accounting

    @property
    def llm_calls(self) -> list[LLMCall]:
        return [c for s in self.steps for c in s.calls]

    @property
    def synth_calls(self) -> int:
        return sum(1 for s in self.steps if s.name == "synthesis")

    @property
    def critic_calls(self) -> int:
        return sum(1 for s in self.steps if s.name == "critic")

    @property
    def pre_llm_ms(self) -> float:
        return sum(s.ms for s in self.steps if s.name in PRE_LLM_STEPS)

    def cost(self) -> dict[str, Any]:
        calls = self.llm_calls
        return {
            "llm_calls": len(calls),
            "synth_calls": self.synth_calls,
            "critic_calls": self.critic_calls,
            "repair_calls": sum(1 for c in calls if c.kind == "repair"),
            "cached_calls": sum(1 for c in calls if c.cached),
            "prompt_tokens": sum(c.prompt_tokens for c in calls),
            "completion_tokens": sum(c.completion_tokens for c in calls),
            "llm_ms": round(sum(c.ms for c in calls), 1),
            "pre_llm_ms": round(self.pre_llm_ms, 1),
            "e2e_ms": round(self.e2e_ms, 1),
            "measured_ms": round(self.measured_ms, 1),
        }

    @property
    def final_guards(self) -> GuardReport | None:
        for r in reversed(self.rounds):
            if r.guards is not None:
                return r.guards
        return None

    def sentences(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in split_sentences(self.answer)]

    def closest_sections(self, n: int = 3) -> list[dict[str, Any]]:
        """The first ``n`` distinct sections of the ranking (shown when abstaining)."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for r in self.retrieval.ranked:
            c = r.chunk
            key = ",".join(c.section_ids) or c.chunk_id
            if key in seen:
                continue
            seen.add(key)
            pages = (
                str(c.page_start) if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
            )
            out.append(
                {
                    "chunk_id": c.chunk_id,
                    "heading_path": c.heading_path,
                    "section_ids": list(c.section_ids),
                    "pages": pages,
                }
            )
            if len(out) == n:
                break
        return out

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable record without passage text (evaluation looks text up by id)."""
        res = self.retrieval
        cited = self.context.subset(cited_labels(self.answer, self.citations))
        return {
            "system": self.system,
            "question": self.question,
            "plain": self.plain,
            "outcome": self.outcome,
            "reason": self.reason,
            "detail": self.detail,
            "answer": self.answer,
            "citations": list(self.citations),
            "sentences": self.sentences(),
            "withheld_answer": self.withheld_answer,
            "cited_chunks": [p.to_dict() for p in cited],
            "context": [p.to_dict() for p in self.context.passages],
            "retrieval": {
                "config_id": res.config_id,
                "top1_first_stage_score": res.top1_first_stage_score,
                "top1_rerank_score": res.top1_rerank_score,
                "abstention_score": self.abstention_score,
                "threshold": self.threshold,
                "timings_ms": res.timings_ms,
                "top_chunks": [r.chunk_id for r in res.ranked[:10]],
            },
            "rounds": [r.to_dict() for r in self.rounds],
            "guards": self.final_guards.to_dict() if self.final_guards else None,
            "steps": [s.to_dict() for s in self.steps],
            "cost": self.cost(),
            "warnings": list(self.warnings),
        }

    def to_api(self) -> dict[str, Any]:
        """Response of the web UI / ask.py: the record plus passage text and closest sections."""
        rec = self.to_record()
        rec["passages"] = {p.label: p.to_dict(include_text=True) for p in self.context.passages}
        rec["closest_sections"] = self.closest_sections()
        return rec


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class Orchestrator:
    def __init__(
        self,
        spec: PipelineSpec,
        retriever: Retriever,
        llm: ChatModel,
        builder: ContextBuilder,
        loop: LoopSettings | None = None,
        *,
        context_window: int | None = None,
        max_output_tokens: int = 512,
        prompts: GenerationPrompts | None = None,
    ) -> None:
        self.spec = spec
        self.retriever = retriever
        self.llm = llm
        self.builder = builder
        self.loop = loop or LoopSettings()
        self.context_window = context_window  # warn when a prompt nears it (None = off)
        self.max_output_tokens = max_output_tokens
        self.prompts = prompts  # domain profile prompts (None: consumer_law)

    def answer(self, question: str, plain: bool = False) -> AnswerResult:
        spec, loop = self.spec, self.loop
        t_start = time.perf_counter()
        adjust = 0.0  # reported minus measured time of cached work
        steps: list[Step] = []
        rounds: list[RoundRecord] = []
        warnings: list[str] = []

        t0 = time.perf_counter()
        res = self.retriever.search(question)
        wall = _ms(t0)
        ret_ms = float(res.timings_ms.get("total", wall))
        adjust += ret_ms - wall
        score = abstention_score(res, spec.retrieval)
        steps.append(
            Step(
                "retrieval",
                0,
                ret_ms,
                "ok",
                {
                    "config_id": res.config_id,
                    "n_ranked": len(res.ranked),
                    "abstention_score": score,
                    "timings_ms": res.timings_ms,
                },
            )
        )
        t0 = time.perf_counter()
        ctx = self.builder.pack(res.ranked)
        steps.append(Step("context", 1, _ms(t0), "ok", ctx.summary()))

        def finish(
            outcome: str, reason: str | None, detail: str, answer: str = "", cites: Any = ()
        ) -> AnswerResult:
            measured = _ms(t_start)
            if self.context_window:
                limit = self.context_window - self.max_output_tokens
                big = [c for s in steps for c in s.calls if c.prompt_tokens >= limit]
                if big:
                    warnings.append(
                        f"{len(big)} prompt(s) of >= {limit} tokens: close to the model's "
                        f"{self.context_window}-token context window (risk of truncation)"
                    )
            withheld = ""
            if outcome == "abstained":
                last = next((r.synth for r in reversed(rounds) if r.synth), None)
                withheld = last.answer if last else ""
            return AnswerResult(
                question=question,
                system=spec.system,
                plain=plain,
                outcome=outcome,
                reason=reason,
                detail=detail,
                answer=answer if outcome == "answered" else "",
                citations=list(cites) if outcome == "answered" else [],
                withheld_answer=withheld,
                context=ctx,
                retrieval=res,
                abstention_score=score,
                threshold=spec.threshold,
                rounds=rounds,
                steps=steps,
                e2e_ms=measured + adjust,
                measured_ms=measured,
                warnings=warnings,
            )

        if spec.threshold is not None:
            below = score is None or score < spec.threshold
            steps.append(
                Step(
                    "threshold",
                    0,
                    0.0,
                    "below" if below else "pass",
                    {"score": score, "threshold": spec.threshold},
                )
            )
            if below:
                return finish(
                    "abstained",
                    "low_retrieval_score",
                    f"top retrieval score {score if score is not None else 'n/a'} is below the "
                    f"abstention threshold {spec.threshold:.4f}",
                )

        synthesizer = Synthesizer(self.llm, plain=plain, prompts=self.prompts)
        critic = Critic(self.llm, prompts=self.prompts) if spec.critic else None
        revision: Revision | None = None
        critic_calls = 0
        n_rounds = loop.max_synth_calls if spec.critic else 1
        for rnd in range(1, n_rounds + 1):
            if rnd > 1:
                t0 = time.perf_counter()
                before = len(ctx.passages)
                ctx = self.builder.extend(ctx, res.ranked)
                new_labels = ctx.labels[before:]
                if revision is not None:
                    revision.new_labels = new_labels
                steps.append(
                    Step(
                        "revision_context",
                        rnd,
                        _ms(t0),
                        "ok",
                        {**ctx.summary(), "added": new_labels},
                    )
                )
            record = RoundRecord(rnd, ctx.summary())
            rounds.append(record)

            t0 = time.perf_counter()
            out, calls, error = synthesizer.run(question, ctx, revision)
            step_ms = _ms(t0) + sum(c.ms - c.measured_ms for c in calls)
            adjust += sum(c.ms - c.measured_ms for c in calls)
            record.synth, record.synth_error = out, error
            status = "error" if out is None else ("abstain" if out.abstain else "answer")
            steps.append(
                Step(
                    "synthesis",
                    rnd,
                    step_ms,
                    status,
                    {
                        "revision": rnd > 1,
                        "answer": out.answer if out else None,
                        "citations": out.citations if out else [],
                        "note": out.note if out else None,
                        "error": error,
                    },
                    calls,
                )
            )
            if out is None:
                return finish(
                    "abstained", "synth_abstain", f"unreadable synthesizer reply: {error}"
                )
            if out.abstain:
                return finish(
                    "abstained",
                    "synth_abstain",
                    "the synthesizer found no answer in the passages"
                    + (f" ({out.note})" if out.note else ""),
                )

            t0 = time.perf_counter()
            report = run_guards(out.answer, out.citations, ctx)
            record.guards = report
            steps.append(
                Step(
                    "guards",
                    rnd,
                    _ms(t0),
                    report.status,
                    {"reasons": report.reasons, "acted_on": spec.guards},
                )
            )
            if not spec.guards:  # synthesizer-only system: guards recorded, not applied
                return finish("answered", None, "synthesizer answer", out.answer, out.citations)
            if critic is None:  # guards only
                if report.passed:
                    return finish("answered", None, "guards passed", out.answer, out.citations)
                return finish(
                    "abstained", "guard_fail", "; ".join(report.reasons), cites=out.citations
                )

            verdict: CriticOutput | None = None
            if critic_calls < loop.max_critic_calls and (
                report.passed or loop.critic_on_guard_fail
            ):
                cited = ctx.subset(cited_labels(out.answer, out.citations))
                t0 = time.perf_counter()
                verdict, ccalls = critic.run(question, cited, out.answer)
                critic_calls += 1
                step_ms = _ms(t0) + sum(c.ms - c.measured_ms for c in ccalls)
                adjust += sum(c.ms - c.measured_ms for c in ccalls)
                record.critic = verdict
                steps.append(
                    Step("critic", rnd, step_ms, verdict.verdict, verdict.to_dict(), ccalls)
                )
                if verdict.error:
                    warnings.append(f"round {rnd}: critic reply unreadable ({verdict.error})")
                if verdict.verdict == "abstain":
                    return finish(
                        "abstained",
                        "critic_abstain",
                        verdict.feedback or "the critic found no answer in the passages",
                    )
            # (critic skipped or budget exhausted: the guards decide)
            if report.passed and (verdict is None or verdict.verdict == "accept"):
                return finish(
                    "answered",
                    None,
                    "guards passed and the critic accepted"
                    if verdict is not None
                    else "guards passed",
                    out.answer,
                    out.citations,
                )
            if rnd == n_rounds:
                if not report.passed:
                    return finish("abstained", "guard_fail", "; ".join(report.reasons))
                if loop.on_final_revise == "answer":
                    warnings.append("released although the critic still asked for a revision")
                    return finish(
                        "answered",
                        None,
                        "critic still asked for a revision (released by configuration)",
                        out.answer,
                        out.citations,
                    )
                return finish(
                    "abstained",
                    "critic_abstain",
                    "the critic still found unsupported content after the revision"
                    + (f": {verdict.feedback}" if verdict and verdict.feedback else ""),
                )
            problems = list(report.reasons)
            if verdict is not None:
                problems += [f"unsupported sentence: {s}" for s in verdict.unsupported_sentences]
                if verdict.feedback:
                    problems.append(f"reviewer feedback: {verdict.feedback}")
            revision = Revision(out.answer, problems, [])
        raise AssertionError("unreachable: every round returns")  # pragma: no cover
