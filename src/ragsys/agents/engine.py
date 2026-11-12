"""Builds orchestrators for the configured systems on shared retrieval resources and one LLM
client per generator backend; used by scripts/ask.py, the generation runs and the web UI.

An orchestrator is built per (system, backend). A system runs on its configured backend (G2H /
G4H: hosted) unless the caller names one (the UI's backend selector); the backend decides the
LLM client, the context budgets, the context window and the completion limit. Every answer
records the backend and the model that produced it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from ragsys.agents.abstention import load_threshold
from ragsys.agents.config import GenerationSettings, SelectionMissing
from ragsys.agents.context import ContextBuilder
from ragsys.agents.orchestrator import AnswerResult, Orchestrator, PipelineSpec
from ragsys.chunking.base import Chunk
from ragsys.llm.backends import BackendUnavailable
from ragsys.llm.client import ChatModel
from ragsys.profiles import get_profile
from ragsys.retrieval.config import RetrievalConfig, RetrievalSettings, parse_config_id
from ragsys.retrieval.embed import model_slug
from ragsys.retrieval.pipeline import RetrievalResources, Retriever


class AnswerEngine:
    """Lazily builds one :class:`Orchestrator` per system. Derived systems run live as their
    base pipeline with the extra step switched on (G3: guards; G5: threshold). When the
    threshold file is missing, G5 runs without a threshold and every result carries a warning
    (``require_threshold=True`` raises instead). A ``selected:`` system whose selection file
    does not exist yet raises, unless ``allow_selection_fallback`` (interactive use), which
    uses the first ``selection_fallback`` config whose embeddings are built, with a warning."""

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
        allow_selection_fallback: bool = False,
        backend: str | None = None,
        llm_factory: Callable[[str], ChatModel] | None = None,
    ) -> None:
        self.settings = settings
        # ``llm`` is the client of ``backend`` (default: llm.yaml's default backend); clients of
        # other backends come from ``llm_factory`` when a system or a request needs them
        self.backend = backend or settings.default_backend
        self.llms: dict[str, ChatModel] = {self.backend: llm}
        self.llm_factory = llm_factory
        self.allow_selection_fallback = allow_selection_fallback
        self._fallback_notes: dict[str, str] = {}
        self.prompts = get_profile(settings.profile).generation
        self.retrieval_settings = retrieval_settings
        self.resources = resources or RetrievalResources(retrieval_settings, rerank_cache=True)
        if threshold is None and load_threshold_file:
            threshold = load_threshold(settings.threshold_path())
        self.threshold_info = threshold
        self.require_threshold = require_threshold
        self._retrievers: dict[str, Retriever] = {}
        self._orchestrators: dict[tuple[str, str], Orchestrator] = {}
        self._lock = threading.Lock()

    @property
    def threshold(self) -> float | None:
        return float(self.threshold_info["threshold"]) if self.threshold_info else None

    def index_built(self, cfg: RetrievalConfig) -> bool:
        """True when the chunks (and, for dense retrieval, the embeddings) of ``cfg`` exist."""
        if not (self.resources.processed_dir / "chunks" / f"{cfg.chunker}.jsonl").exists():
            return False
        if cfg.uses_dense:
            name = self.retrieval_settings.dense_model_name(cfg.dense_model)
            return (self.resources.emb_dir / model_slug(name) / f"{cfg.chunker}.npy").exists()
        return True

    def retrieval_config(self, system: str) -> RetrievalConfig:
        """The system's retrieval config; see the class docstring for ``selected:`` systems.
        Interactive use (``allow_selection_fallback``) also falls back when the configured
        config's embeddings are not built yet (an app index before its build), with a note."""
        base = self.settings.runnable_base(system)
        configured: RetrievalConfig | None = None
        try:
            configured = self.settings.retrieval_cfg(system)
        except SelectionMissing:
            if not self.allow_selection_fallback:
                raise
            reason = (
                f"{self.settings.retrieval_id(base)} is not selected yet (run the dev retrieval "
                "grid and select_configs.py)"
            )
        else:
            if not self.allow_selection_fallback or self.index_built(configured):
                return configured
            reason = (
                f"the index of {configured.config_id()} is not built yet (run "
                "scripts/run_app_collections.py or build_index.py)"
            )
        for cid in self.settings.selection_fallback:
            cfg = parse_config_id(cid)
            if not self.index_built(cfg):
                continue
            if configured is not None and cfg.config_id() == configured.config_id():
                continue
            self._fallback_notes[system] = (
                f"{reason}; {system} used the provisional retrieval {cid}"
            )
            return cfg
        if configured is not None:  # nothing to fall back to: embeddings are computed on demand
            return configured
        raise SelectionMissing(
            f"{system}: no selection file and none of the fallback configs "
            f"{self.settings.selection_fallback} has its embeddings built (run build_index.py)"
        )

    def spec(self, system: str) -> PipelineSpec:
        sys_spec = self.settings.system(system)
        if sys_spec.threshold and self.threshold is None and self.require_threshold:
            raise FileNotFoundError(
                f"{system} needs {self.settings.threshold_path()} (run scripts/tune_abstention.py)"
            )
        return PipelineSpec.from_settings(
            self.settings, system, self.threshold, retrieval=self.retrieval_config(system)
        )

    def warnings_for(self, system: str) -> list[str]:
        out: list[str] = []
        if self.settings.system(system).threshold and self.threshold is None:
            out.append(
                "abstention threshold not tuned yet (run scripts/tune_abstention.py); "
                f"{system} ran without the retrieval-score threshold"
            )
        note = self._fallback_notes.get(system)
        if note:
            out.append(note)
        return out

    def retriever(self, config: str | RetrievalConfig) -> Retriever:
        cfg = parse_config_id(config) if isinstance(config, str) else config
        config_id = cfg.config_id()
        if config_id not in self._retrievers:
            r = Retriever(cfg, self.resources)
            r.warm_up()  # load models now so the first timed question is not penalised
            self._retrievers[config_id] = r
        return self._retrievers[config_id]

    @property
    def llm(self) -> ChatModel:
        """The client of the engine's default backend."""
        return self.llms[self.backend]

    def backend_for(self, system: str, backend: str | None = None) -> str:
        """``backend`` when given, else the system's configured one, else the engine's."""
        if backend:
            return backend
        return self.settings.system(self.settings.runnable_base(system)).backend or self.backend

    def llm_for(self, backend: str) -> ChatModel:
        if backend not in self.llms:
            if self.llm_factory is None:
                raise BackendUnavailable(f"no LLM client for backend {backend!r}")
            self.llms[backend] = self.llm_factory(backend)
        return self.llms[backend]

    def orchestrator(self, system: str, backend: str | None = None) -> Orchestrator:
        name = self.backend_for(system, backend)
        key = (system, name)
        if key not in self._orchestrators:
            spec = self.spec(system)
            ctx = self.settings.context_for(name)
            retriever = self.retriever(spec.retrieval)
            builder = ContextBuilder(
                ctx.budget_tokens, ctx.revision_budget_tokens, ctx.min_passages, retriever
            )
            max_out = (
                int(self.settings.llm_overrides.get("max_tokens", 512))
                if name == self.settings.default_backend
                else int(self.settings.llm(name).max_tokens)
            )
            self._orchestrators[key] = Orchestrator(
                spec,
                retriever,
                self.llm_for(name),
                builder,
                self.settings.loop,
                context_window=self.settings.context_window_for(name),
                max_output_tokens=max_out,
                prompts=self.prompts,
            )
        return self._orchestrators[key]

    def ask(
        self, question: str, system: str, plain: bool = False, backend: str | None = None
    ) -> AnswerResult:
        """Answer one question (serialised: the local LLM and CPU handle one at a time)."""
        name = self.backend_for(system, backend)
        with self._lock:
            orch = self.orchestrator(system, name)
            result = orch.answer(question, plain=plain)
        result.backend = name
        result.model = str(getattr(orch.llm, "model", "") or "")
        result.warnings[:0] = self.warnings_for(system)
        return result

    def chunkers(self) -> list[str]:
        out: set[str] = set()
        for s in self.settings.systems:
            try:
                out.add(self.retrieval_config(s).chunker)
            except SelectionMissing:
                continue
        return sorted(out)

    def chunk(self, chunk_id: str) -> Chunk | None:
        """A chunk of one of the configured systems' chunkers (others are refused)."""
        chunker = chunk_id.split(":", 1)[0]
        if chunker not in self.chunkers():
            return None
        return self.resources.corpus(chunker).by_id.get(chunk_id)

    def close(self) -> None:
        self.resources.close()
