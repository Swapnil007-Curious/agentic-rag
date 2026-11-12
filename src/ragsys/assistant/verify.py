"""On-demand verification of a grounded chat reply with the grounded pipeline's checks.

The deterministic guards (citation validity and number grounding, :mod:`ragsys.agents.guards`)
run on the prose of the reply (code blocks are not claims) and on each sentence with the
passages it cites; the critic (:mod:`ragsys.agents.critic`, the collection's prompt) checks the
whole reply against the cited passages in one LLM call. A sentence is ``supported`` when it
cites a passage that was given, its numbers occur in its cited passages and the critic did not
flag it; otherwise ``unsupported`` with the reasons.
"""

from __future__ import annotations

import difflib
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ragsys.agents.context import PackedContext, Passage
from ragsys.agents.critic import Critic
from ragsys.agents.guards import cited_labels, number_grounding, run_guards
from ragsys.agents.text import labels_in, split_sentences, strip_citations
from ragsys.llm.client import ChatModel
from ragsys.profiles import GenerationPrompts

_FENCE_BLOCK_RE = re.compile(
    r"^\s{0,3}(`{3,}|~{3,})[^\n]*\n.*?(?:^\s{0,3}\1[^\n]*$|\Z)", re.S | re.M
)
_MATH_BLOCK_RE = re.compile(r"\$\$.*?\$\$", re.S)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.M)  # headings are not claims
_EMPHASIS_RE = re.compile(r"(\*\*|__)(.+?)\1")


def prose_of(answer: str) -> str:
    """The reply without fenced code, display math, headings and bold markers."""
    text = _FENCE_BLOCK_RE.sub("\n", answer)
    text = _MATH_BLOCK_RE.sub(" ", text)
    text = _HEADING_RE.sub("", text)
    return _EMPHASIS_RE.sub(r"\2", text)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", strip_citations(text)).strip().lower().rstrip(".")


def _flagged(sentence: str, flagged: Sequence[str]) -> bool:
    s = _norm(sentence)
    for f in flagged:
        g = _norm(f)
        if not g:
            continue
        if g in s or s in g or difflib.SequenceMatcher(None, s, g).ratio() >= 0.8:
            return True
    return False


@dataclass
class VerifyResult:
    sentences: list[dict[str, Any]]
    guards: dict[str, Any]
    critic: dict[str, Any]
    model: str
    ms: float

    def to_api(self) -> dict[str, Any]:
        supported = sum(1 for s in self.sentences if s["status"] == "supported")
        return {
            "sentences": self.sentences,
            "supported": supported,
            "unsupported": len(self.sentences) - supported,
            "total": len(self.sentences),
            "verdict": "supported" if supported == len(self.sentences) else "partly_supported",
            "guards": self.guards,
            "critic": self.critic,
            "model": self.model,
            "ms": round(self.ms, 1),
        }


def verify_answer(
    question: str,
    answer: str,
    passages: Sequence[Passage],
    llm: ChatModel,
    prompts: GenerationPrompts,
) -> VerifyResult:
    t0 = time.perf_counter()
    ctx = PackedContext(list(passages), 0, 0, len(passages))
    prose = prose_of(answer)
    report = run_guards(prose, labels_in(prose), ctx)
    cited = ctx.subset(cited_labels(prose))
    verdict, _calls = Critic(llm, prompts=prompts).run(question, cited, prose)
    known = set(ctx.labels)
    rows: list[dict[str, Any]] = []
    for s in split_sentences(prose):
        if not s.text:
            continue
        reasons: list[str] = []
        valid = [lab for lab in s.labels if lab in known]
        if not s.labels:
            reasons.append("no citation")
        elif len(valid) < len(s.labels):
            reasons.append("cites a passage that was not given")
        if valid:
            numbers = number_grounding(s.raw, ctx.subset(valid))
            reasons += numbers.reasons
        if _flagged(s.text, verdict.unsupported_sentences):
            reasons.append("the critic found it unsupported by the cited passages")
        rows.append(
            {
                "text": s.text,
                "citations": list(s.labels),
                "status": "unsupported" if reasons else "supported",
                "reasons": reasons,
            }
        )
    if verdict.verdict == "abstain":  # the critic says the passages do not answer at all
        for r in rows:
            if r["status"] == "supported":
                r["status"] = "unsupported"
                r["reasons"].append("the critic found that the passages do not answer the question")
    return VerifyResult(
        rows,
        report.to_dict(),
        verdict.to_dict(),
        str(getattr(llm, "model", "") or ""),
        (time.perf_counter() - t0) * 1000.0,
    )
