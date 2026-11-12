"""Automatic grounding without an LLM call.

The latest user message (and, for a short follow-up such as "and what about its complexity?",
the previous user message joined with it) is embedded once with bge-small. Every source (the
selected collection, the session's uploads) is searched; a source whose top-1 cosine reaches its
threshold contributes its hits. The pooled hits, best first, are packed as labelled passages
[C1], [C2], ... into the backend's token budget with the grounded pipeline's context packer.
When no source passes, the message is answered as plain chat.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ragsys.agents.context import ContextBuilder, PackedContext
from ragsys.assistant.config import GateSettings
from ragsys.assistant.index import DenseIndex, Hit, LockedEncoder
from ragsys.retrieval.fusion import ComponentHit
from ragsys.retrieval.pipeline import RetrievedChunk

# words that refer back to the conversation
_BACK_REFERENCE_RE = re.compile(
    r"\b(it|its|it's|that|this|those|these|they|them|their|there|above|previous|same|again|"
    r"more|another|instead|example|examples|why|how come)\b",
    re.IGNORECASE,
)
_FOLLOW_UP_START_RE = re.compile(
    r"^\s*(and|but|so|also|then|ok|okay|what about|how about|why|elaborate|explain more|"
    r"tell me more|continue|go on)\b",
    re.IGNORECASE,
)
MAX_QUERY_CHARS = 2000


def is_follow_up(latest: str, max_words: int) -> bool:
    """A short message that refers back to the conversation."""
    if len(latest.split()) > max_words:
        return False
    return bool(_FOLLOW_UP_START_RE.search(latest) or _BACK_REFERENCE_RE.search(latest))


def gate_queries(latest: str, previous: str | None, max_words: int) -> list[str]:
    """Texts to embed: the latest message, plus previous + latest for a follow-up."""
    out = [latest.strip()[:MAX_QUERY_CHARS]]
    if previous and previous.strip() and is_follow_up(latest, max_words):
        out.append(
            f"{previous.strip()[: MAX_QUERY_CHARS // 2]}\n{latest.strip()}"[:MAX_QUERY_CHARS]
        )
    return out


@dataclass
class SourceCheck:
    source: str
    top_score: float | None
    threshold: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "top_score": None if self.top_score is None else round(self.top_score, 4),
            "threshold": self.threshold,
            "passed": self.passed,
        }


@dataclass
class GateDecision:
    grounded: bool
    queries: list[str]
    checks: list[SourceCheck]
    context: PackedContext | None = None
    hits: list[Hit] = field(default_factory=list)
    embed_ms: float = 0.0
    search_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.search_ms

    @property
    def top_score(self) -> float | None:
        scores = [c.top_score for c in self.checks if c.top_score is not None]
        return max(scores) if scores else None

    def source_of(self, chunk_id: str) -> str:
        return next((h.source for h in self.hits if h.chunk.chunk_id == chunk_id), "")


def to_ranked(hits: Sequence[Hit]) -> list[RetrievedChunk]:
    """Hits as a ranking the grounded pipeline's packer and orchestrator understand."""
    return [
        RetrievedChunk(
            rank=i,
            chunk=h.chunk,
            provenance=h.source,
            first_stage_rank=i,
            first_stage_score=h.score,
            components={"dense": ComponentHit(i, h.score)},
        )
        for i, h in enumerate(hits, start=1)
    ]


def merge_hits(groups: Sequence[Sequence[Hit]]) -> list[Hit]:
    """Pool hits of several sources, best score first (stable), one entry per chunk."""
    pooled: dict[str, Hit] = {}
    for group in groups:
        for h in group:
            cur = pooled.get(h.chunk.chunk_id)
            if cur is None or h.score > cur.score:
                pooled[h.chunk.chunk_id] = h
    return sorted(pooled.values(), key=lambda h: -h.score)


class GroundingGate:
    def __init__(self, encoder: LockedEncoder, settings: GateSettings) -> None:
        self.encoder = encoder
        self.settings = settings

    def search(
        self, queries: list[str], sources: Sequence[DenseIndex]
    ) -> tuple[list[list[Hit]], float, float]:
        t0 = time.perf_counter()
        qmat = self.encoder.encode(queries, is_query=True)
        embed_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        groups = [idx.search(qmat, self.settings.search_k) for idx in sources]
        return groups, embed_ms, (time.perf_counter() - t0) * 1000.0

    def decide(
        self,
        latest: str,
        previous: str | None,
        sources: Sequence[tuple[DenseIndex, float]],
        budget_tokens: int,
    ) -> GateDecision:
        """Search every (index, threshold) source and pack the passages of those that pass."""
        queries = gate_queries(latest, previous, self.settings.follow_up_max_words)
        if not sources:
            return GateDecision(False, queries, [])
        groups, embed_ms, search_ms = self.search(queries, [s[0] for s in sources])
        checks: list[SourceCheck] = []
        passing: list[list[Hit]] = []
        for (idx, threshold), hits in zip(sources, groups, strict=True):
            top = hits[0].score if hits else None
            ok = top is not None and top >= threshold
            checks.append(SourceCheck(idx.source, top, threshold, ok))
            if ok:
                passing.append(hits)
        decision = GateDecision(
            bool(passing), queries, checks, embed_ms=embed_ms, search_ms=search_ms
        )
        if passing:
            pooled = merge_hits(passing)[: self.settings.max_passages]
            builder = ContextBuilder(budget_tokens, budget_tokens, 1)
            decision.context = builder.pack(to_ranked(pooled))
            used = {p.chunk_id for p in decision.context.passages}
            decision.hits = [h for h in pooled if h.chunk.chunk_id in used]
        return decision
