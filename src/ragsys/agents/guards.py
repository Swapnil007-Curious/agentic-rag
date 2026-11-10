"""Deterministic hallucination guards (no LLM).

* ``citation_validity``: the answer is not empty, every sentence carries at least one citation,
  and every cited label (inline tags and the ``citations`` field) exists in the context.
* ``number_grounding``: every number in the answer, written as digits or English words
  ("30", "thirty", "forty-five days", "Rs. 50,00,000", "fifty lakh", "one crore"), occurs by
  value in at least one cited passage (its heading path, section ids or text; page numbers are
  not used). Numbers are compared by value, so "30 days" and "thirty days" match and "50 lakh"
  matches "50,00,000". A bare "one" (as in "any one of") is not treated as a number.

A failed guard marks the answer ``unsupported`` and lists the reasons.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ragsys.agents.context import PackedContext, Passage
from ragsys.agents.text import labels_in, split_sentences, strip_citations
from ragsys.eval.checks import extract_numbers, number_forms


@dataclass
class GuardResult:
    name: str
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "reasons": list(self.reasons)}


@dataclass
class GuardReport:
    results: list[GuardResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def status(self) -> str:
        return "supported" if self.passed else "unsupported"

    @property
    def reasons(self) -> list[str]:
        return [f"{r.name}: {msg}" for r in self.results for msg in r.reasons]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "reasons": self.reasons,
            "results": [r.to_dict() for r in self.results],
        }


def cited_labels(answer: str, citations: Sequence[str] = ()) -> list[str]:
    """Inline citation labels of the answer, then any extra labels of the citations field."""
    out = labels_in(answer)
    out += [c for c in citations if c not in out]
    return out


def _clip(text: str, n: int = 80) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def citation_validity(
    answer: str, citations: Sequence[str], context_labels: Sequence[str]
) -> GuardResult:
    known = set(context_labels)
    reasons: list[str] = []
    sentences = split_sentences(answer)
    if not sentences:
        reasons.append("the answer is empty")
    for i, s in enumerate(sentences, start=1):
        if not s.labels:
            reasons.append(f'sentence {i} has no citation: "{_clip(s.text)}"')
    unknown = [lab for lab in cited_labels(answer, citations) if lab not in known]
    if unknown:
        reasons.append(f"cited label(s) not in the context: {', '.join(unknown)}")
    return GuardResult("citation_validity", not reasons, reasons)


def answer_numbers(answer: str) -> list[tuple[str, float]]:
    """(surface text, value) of every number the guard checks in ``answer``."""
    text = strip_citations(answer)
    return [(m.text, m.value) for m in extract_numbers(text) if m.text.strip() != "one"]


def number_grounding(answer: str, cited: Sequence[Passage]) -> GuardResult:
    forms = number_forms([p.grounding_text() for p in cited])
    labels = ", ".join(p.label for p in cited) or "none"
    reasons = []
    seen: set[float] = set()
    for surface, value in answer_numbers(answer):
        key = round(value, 4)
        if key in forms or key in seen:
            continue
        seen.add(key)
        reasons.append(f'number "{surface}" does not occur in the cited passages ({labels})')
    return GuardResult("number_grounding", not reasons, reasons)


def run_guards(answer: str, citations: Sequence[str], ctx: PackedContext) -> GuardReport:
    cited = ctx.subset(cited_labels(answer, citations))
    return GuardReport(
        [
            citation_validity(answer, citations, ctx.labels),
            number_grounding(answer, cited),
        ]
    )
