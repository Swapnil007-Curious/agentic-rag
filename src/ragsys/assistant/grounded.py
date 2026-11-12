"""The strict "Grounded only" mode: the grounded pipeline's :class:`Orchestrator` (synthesizer,
deterministic guards, critic, one revision round, abstention when unsupported) run over the
assistant's sources (the collection's app index and the session's uploads, dense bge-small).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ragsys.agents.config import ContextSettings, LoopSettings
from ragsys.agents.context import ContextBuilder
from ragsys.agents.orchestrator import AnswerResult, Orchestrator, PipelineSpec
from ragsys.assistant.gate import to_ranked
from ragsys.assistant.index import Hit
from ragsys.llm.client import ChatModel
from ragsys.profiles import GenerationPrompts, _grounded_prompts
from ragsys.retrieval.config import RetrievalConfig
from ragsys.retrieval.pipeline import RetrievalResult, Retriever

ASSISTANT_RETRIEVAL = RetrievalConfig(
    chunker="structural", retriever="dense", dense_model="bge-small"
)

# prompts for questions about the user's own uploads (no collection selected)
UPLOAD_PROMPTS: GenerationPrompts = _grounded_prompts(
    "the user's uploaded documents",
    "documents the user uploaded",
    "The report was published in 2021",
    "numbers, names, dates and quoted terms",
)


class SourcesRetriever(Retriever):
    """A :class:`Retriever` over the assistant's in-memory sources. Only :meth:`search` is
    used by the orchestrator and the context packer."""

    def __init__(self, search: Callable[[str], list[Hit]], config_id: str) -> None:
        # the base class builds BM25 / encoder resources from disk; none are needed here
        self.cfg = ASSISTANT_RETRIEVAL
        self.config_id = config_id
        self._search = search

    def warm_up(self) -> None:
        return None

    def search(self, query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        hits = self._search(query)
        ms = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(query, self.config_id, to_ranked(hits), 10, {"total": round(ms, 3)})


def grounded_answer(
    question: str,
    search: Callable[[str], list[Hit]],
    llm: ChatModel,
    *,
    prompts: GenerationPrompts,
    context: ContextSettings,
    loop: LoopSettings,
    max_output_tokens: int,
    context_window: int | None,
    config_id: str,
) -> AnswerResult:
    """One grounded answer (or abstention) with guards and critic, as system G4 runs it."""
    retriever = SourcesRetriever(search, config_id)
    spec = PipelineSpec("grounded", ASSISTANT_RETRIEVAL, guards=True, critic=True)
    builder = ContextBuilder(
        context.budget_tokens, context.revision_budget_tokens, context.min_passages, retriever
    )
    orch = Orchestrator(
        spec,
        retriever,
        llm,
        builder,
        loop,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        prompts=prompts,
    )
    return orch.answer(question)
