"""Shared setup for the generation scripts: settings, the LLM client and the answer engine."""

from __future__ import annotations

from pathlib import Path

from ragsys.agents.config import GenerationSettings, load_generation_settings
from ragsys.agents.engine import AnswerEngine
from ragsys.agents.fake import agent_fake_llm
from ragsys.config import resolve_path
from ragsys.llm.client import ChatModel, LLMClient
from ragsys.retrieval.config import RetrievalSettings, load_retrieval_settings
from ragsys.retrieval.pipeline import RetrievalResources


def load_settings(
    config: str | Path, retrieval_config: str | Path | None = None
) -> tuple[GenerationSettings, RetrievalSettings]:
    gen = load_generation_settings(resolve_path(Path(config)))
    rpath = Path(retrieval_config) if retrieval_config else gen.retrieval_config
    return gen, load_retrieval_settings(resolve_path(rpath))


def make_llm(settings: GenerationSettings, fake: bool = False) -> ChatModel:
    if fake:
        return agent_fake_llm()
    return LLMClient(settings.llm())


def llm_unreachable_message(llm: ChatModel) -> str | None:
    """A hint when the configured endpoint does not answer (None when it does or is a fake)."""
    if isinstance(llm, LLMClient) and not llm.is_reachable():
        return (
            f"the LLM endpoint {llm.base_url} is not reachable; start Ollama and run "
            f"`ollama pull {llm.model}` (or fix configs/llm.yaml)"
        )
    return None


def build_engine(
    settings: GenerationSettings,
    retrieval_settings: RetrievalSettings,
    llm: ChatModel,
    *,
    require_threshold: bool = False,
    load_threshold_file: bool = True,
    allow_selection_fallback: bool = False,
) -> AnswerEngine:
    resources = RetrievalResources(retrieval_settings, rerank_cache=True)
    return AnswerEngine(
        settings,
        retrieval_settings,
        llm,
        resources,
        load_threshold_file=load_threshold_file,
        require_threshold=require_threshold,
        allow_selection_fallback=allow_selection_fallback,
    )
