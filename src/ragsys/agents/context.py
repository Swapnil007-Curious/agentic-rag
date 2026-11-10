"""Context builder: retrieve with a :class:`RetrievalConfig`, then pack ranked chunks into a token
budget as labelled passages [C1], [C2], ...

Packing follows the retrieval evaluation's token-budget coverage exactly: chunks are taken in
rank order (from the whole ranked list, not only the top ``final_k``) until the next one would
exceed the budget, counted with chunk ``n_tokens``. So the generator receives precisely the
chunks that ``coverage@1024`` measured. A revision round *extends* the context with the next
chunks of the same ranking up to a larger budget; labels continue (C7, C8, ...).

Document text is untrusted. Every passage is wrapped in explicit delimiters, delimiter
look-alikes inside the text are neutralised so a passage cannot close its own block, and the
agents' system prompts state that passages are data and that instructions inside them must be
ignored. The system prompts never contain document text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ragsys.retrieval.pipeline import RetrievalResult, RetrievedChunk, Retriever

PASSAGE_OPEN = "<<<PASSAGE {label}>>>"
PASSAGE_TEXT = "<<<TEXT>>>"
PASSAGE_CLOSE = "<<<END {label}>>>"
_DELIM_RE = re.compile(r"<{3,}|>{3,}")


def sanitize(text: str) -> str:
    """Neutralise delimiter look-alikes (runs of three or more '<' or '>') in untrusted text."""
    return _DELIM_RE.sub(lambda m: "‹" if m.group().startswith("<") else "›", text)


@dataclass
class Passage:
    label: str  # "C1"
    chunk_id: str
    rank: int  # position in the retriever's ranking
    doc_id: str
    heading_path: str
    section_ids: list[str]
    node_ids: list[str]
    page_start: int
    page_end: int
    n_tokens: int
    text: str  # display_text (exact cleaned document text)
    round_added: int = 1
    provenance: str = "first_stage"
    first_stage_score: float | None = None
    rerank_score: float | None = None

    @property
    def pages(self) -> str:
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"

    def block(self) -> str:
        """The delimited passage as shown to the agents."""
        return "\n".join(
            [
                PASSAGE_OPEN.format(label=self.label),
                f"Source: {sanitize(self.heading_path)}",
                f"Sections: {', '.join(self.section_ids) or '-'}",
                PASSAGE_TEXT,
                sanitize(self.text.strip()),
                PASSAGE_CLOSE.format(label=self.label),
            ]
        )

    def grounding_text(self) -> str:
        """What the agents saw of this passage, minus page numbers (used by the number guard)."""
        return f"{self.heading_path}\n{' '.join(self.section_ids)}\n{self.text}"

    def nli_premise(self) -> str:
        return f"{self.heading_path}\n{self.text}"

    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "label": self.label,
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "doc_id": self.doc_id,
            "heading_path": self.heading_path,
            "section_ids": list(self.section_ids),
            "node_ids": list(self.node_ids),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "pages": self.pages,
            "n_tokens": self.n_tokens,
            "round_added": self.round_added,
            "provenance": self.provenance,
            "first_stage_score": self.first_stage_score,
            "rerank_score": self.rerank_score,
        }
        if include_text:
            d["text"] = self.text
        return d


def passage_from(item: RetrievedChunk, label: str, round_added: int) -> Passage:
    c = item.chunk
    return Passage(
        label=label,
        chunk_id=c.chunk_id,
        rank=item.rank,
        doc_id=c.doc_id,
        heading_path=c.heading_path,
        section_ids=list(c.section_ids),
        node_ids=list(c.node_ids),
        page_start=c.page_start,
        page_end=c.page_end,
        n_tokens=c.n_tokens,
        text=c.display_text,
        round_added=round_added,
        provenance=item.provenance,
        first_stage_score=item.first_stage_score,
        rerank_score=item.rerank_score,
    )


@dataclass
class PackedContext:
    passages: list[Passage]
    budget_tokens: int
    used_tokens: int
    next_index: int  # index in the ranked list where packing stopped
    by_label: dict[str, Passage] = field(init=False)

    def __post_init__(self) -> None:
        self.by_label = {p.label: p for p in self.passages}

    @property
    def labels(self) -> list[str]:
        return [p.label for p in self.passages]

    def render(self) -> str:
        return "\n\n".join(p.block() for p in self.passages)

    def subset(self, labels: Sequence[str]) -> list[Passage]:
        """Passages for ``labels`` that exist, in label order of first appearance."""
        seen: list[Passage] = []
        for lab in labels:
            p = self.by_label.get(lab)
            if p is not None and p not in seen:
                seen.append(p)
        return seen

    def summary(self) -> dict[str, Any]:
        return {
            "n_passages": len(self.passages),
            "used_tokens": self.used_tokens,
            "budget_tokens": self.budget_tokens,
            "labels": self.labels,
        }


class ContextBuilder:
    """Packs a ranking into labelled passages; optionally owns the retriever."""

    def __init__(
        self,
        budget_tokens: int = 1024,
        revision_budget_tokens: int = 2048,
        min_passages: int = 1,
        retriever: Retriever | None = None,
    ) -> None:
        if revision_budget_tokens < budget_tokens:
            raise ValueError("revision budget must be at least the first-round budget")
        self.budget_tokens = budget_tokens
        self.revision_budget_tokens = revision_budget_tokens
        self.min_passages = min_passages
        self.retriever = retriever

    def retrieve(self, question: str) -> RetrievalResult:
        if self.retriever is None:
            raise RuntimeError("ContextBuilder has no retriever")
        return self.retriever.search(question)

    def _fill(
        self,
        ranked: Sequence[RetrievedChunk],
        start: int,
        budget: int,
        passages: list[Passage],
        used: int,
        round_added: int,
    ) -> PackedContext:
        i = start
        while i < len(ranked):
            n = ranked[i].chunk.n_tokens
            if used + n > budget and len(passages) >= self.min_passages:
                break
            passages.append(passage_from(ranked[i], f"C{len(passages) + 1}", round_added))
            used += n
            i += 1
        return PackedContext(passages, budget, used, i)

    def pack(self, ranked: Sequence[RetrievedChunk]) -> PackedContext:
        """First-round context: rank order until the next chunk would exceed the budget."""
        return self._fill(ranked, 0, self.budget_tokens, [], 0, 1)

    def extend(self, ctx: PackedContext, ranked: Sequence[RetrievedChunk]) -> PackedContext:
        """Revision context: the first-round passages plus the next chunks up to the revision
        budget (labels continue)."""
        return self._fill(
            ranked,
            ctx.next_index,
            self.revision_budget_tokens,
            list(ctx.passages),
            ctx.used_tokens,
            2,
        )
