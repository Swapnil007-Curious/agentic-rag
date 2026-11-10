"""Builds orchestrators for the configured systems on shared retrieval resources and one LLM
client; used by scripts/ask.py, the generation runs and the web UI."""

from __future__ import annotations

import threading
from typing import Any

from ragsys.agents.abstention import load_threshold
from ragsys.agents.config import GenerationSettings
from ragsys.agents.context import ContextBuilder
from ragsys.agents.orchestrator import AnswerResult, Orchestrator, PipelineSpec
from ragsys.chunking.base import Chunk
from ragsys.llm.client import ChatModel
from ragsys.retrieval.config import RetrievalSettings
from ragsys.retrieval.pipeline import RetrievalResources, Retriever


class AnswerEngine:
    """Lazily builds one :class:`Orchestrator` per system. Derived systems run live as their
    base pipeline with the extra step switched on (G3: guards; G5: threshold). When the
    threshold file is missing, G5 runs without a threshold and every result carries a warning
    (``require_threshold=True`` raises instead)."""

    def __init__(
        self,
        settings: GenerationSettings,
        retrieval_settings: RetrievalSettings,
        llm: ChatModel,
        resources: RetrievalResources | None = None,
        *,
        threshold: dict[str, Any] | None = None,
        load_threshold_file: bool = True,
        require_threshold: bool = False,
    ) -> None:
        self.settings = settings
        self.retrieval_settings = retrieval_settings
        self.llm = llm
        self.resources = resources or RetrievalResources(retrieval_settings, rerank_cache=True)
        if threshold is None and load_threshold_file:
            threshold = load_threshold(settings.threshold_path())
        self.threshold_info = threshold
        self.require_threshold = require_threshold
        self._retrievers: dict[str, Retriever] = {}
        self._orchestrators: dict[str, Orchestrator] = {}
        self._lock = threading.Lock()

    @property
    def threshold(self) -> float | None:
        return float(self.threshold_info["threshold"]) if self.threshold_info else None

    def spec(self, system: str) -> PipelineSpec:
        sys_spec = self.settings.system(system)
        if sys_spec.threshold and self.threshold is None and self.require_threshold:
            raise FileNotFoundError(
                f"{system} needs {self.settings.threshold_path()} (run scripts/tune_abstention.py)"
            )
        return PipelineSpec.from_settings(self.settings, system, self.threshold)

    def warnings_for(self, system: str) -> list[str]:
        if self.settings.system(system).threshold and self.threshold is None:
            return [
                "abstention threshold not tuned yet (run scripts/tune_abstention.py); "
                f"{system} ran without the retrieval-score threshold"
            ]
        return []

    def retriever(self, config_id: str) -> Retriever:
        if config_id not in self._retrievers:
            from ragsys.retrieval.config import parse_config_id

            r = Retriever(parse_config_id(config_id), self.resources)
            r.warm_up()  # load models now so the first timed question is not penalised
            self._retrievers[config_id] = r
        return self._retrievers[config_id]

    def orchestrator(self, system: str) -> Orchestrator:
        if system not in self._orchestrators:
            spec = self.spec(system)
            ctx = self.settings.context
            retriever = self.retriever(spec.retrieval.config_id())
            builder = ContextBuilder(
                ctx.budget_tokens, ctx.revision_budget_tokens, ctx.min_passages, retriever
            )
            max_out = int(self.settings.llm_overrides.get("max_tokens", 512))
            self._orchestrators[system] = Orchestrator(
                spec,
                retriever,
                self.llm,
                builder,
                self.settings.loop,
                context_window=self.settings.llm_context_window,
                max_output_tokens=max_out,
            )
        return self._orchestrators[system]

    def ask(self, question: str, system: str, plain: bool = False) -> AnswerResult:
        """Answer one question (serialised: the local LLM and CPU handle one at a time)."""
        with self._lock:
            result = self.orchestrator(system).answer(question, plain=plain)
        result.warnings[:0] = self.warnings_for(system)
        return result

    def chunkers(self) -> list[str]:
        return sorted({self.settings.retrieval_cfg(s).chunker for s in self.settings.systems})

    def chunk(self, chunk_id: str) -> Chunk | None:
        """A chunk of one of the configured systems' chunkers (others are refused)."""
        chunker = chunk_id.split(":", 1)[0]
        if chunker not in self.chunkers():
            return None
        return self.resources.corpus(chunker).by_id.get(chunk_id)

    def close(self) -> None:
        self.resources.close()
