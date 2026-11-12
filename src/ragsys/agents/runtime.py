"""Shared setup for the generation scripts: settings, the LLM clients per backend and the answer
engine.

Evaluations never switch backends: :func:`require_llms` checks every backend a run needs before
anything is answered and the script stops when one is unavailable (missing key, unreachable
endpoint). ``--fake-llm`` replaces every backend by a deterministic offline stand-in whose model
name says which backend it stands for.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ragsys.agents.config import GenerationSettings, load_generation_settings
from ragsys.agents.engine import AnswerEngine
from ragsys.agents.fake import agent_fake_llm
from ragsys.config import resolve_path
from ragsys.llm.backends import key_present, unavailable_reason
from ragsys.llm.client import API_KEY_ENV, ChatModel, LLMClient
from ragsys.retrieval.config import RetrievalSettings, load_retrieval_settings
from ragsys.retrieval.pipeline import RetrievalResources


def load_settings(
    config: str | Path, retrieval_config: str | Path | None = None
) -> tuple[GenerationSettings, RetrievalSettings]:
    gen = load_generation_settings(resolve_path(Path(config)))
    rpath = Path(retrieval_config) if retrieval_config else gen.retrieval_config
    return gen, load_retrieval_settings(resolve_path(rpath))


def fake_model_name(settings: GenerationSettings, backend: str) -> str:
    return "fake-agent-llm" if backend == settings.default_backend else f"fake-{backend}-llm"


def make_llm(
    settings: GenerationSettings, fake: bool = False, backend: str | None = None
) -> ChatModel:
    """The generation client of ``backend`` (default: llm.yaml's default backend)."""
    name = backend or settings.default_backend
    if fake:
        llm = agent_fake_llm()
        llm.model = fake_model_name(settings, name)
        return llm
    return LLMClient(settings.llm(name))


def llm_unreachable_message(llm: ChatModel) -> str | None:
    """A hint when the endpoint cannot answer (None when it can or is a fake). A backend that
    sends the key and has none fails without a network call."""
    if not isinstance(llm, LLMClient):
        return None
    hosted = llm.backend not in (None, "local") and llm.cfg.send_api_key
    if hosted and not key_present():
        return (
            f"backend {llm.backend!r} ({llm.model} at {llm.base_url}) needs the environment "
            f"variable {API_KEY_ENV}; set it in this shell (PowerShell: $env:{API_KEY_ENV} = "
            '"...") or use --backend local'
        )
    if not llm.is_reachable():
        if hosted:
            return (
                f"backend {llm.backend!r}: the endpoint {llm.base_url} is not reachable (check "
                "the network connection and the key)"
            )
        return (
            f"the LLM endpoint {llm.base_url} is not reachable; start Ollama and run "
            f"`ollama pull {llm.model}` (or fix configs/llm.yaml)"
        )
    return None


def make_llms(
    settings: GenerationSettings, backends: Iterable[str], fake: bool = False
) -> dict[str, ChatModel]:
    return {b: make_llm(settings, fake, b) for b in dict.fromkeys(backends)}


def require_llms(llms: dict[str, ChatModel]) -> str | None:
    """The first problem among the clients a run needs, or None (no fallback is attempted)."""
    for llm in llms.values():
        problem = llm_unreachable_message(llm)
        if problem:
            return problem
    return None


def backend_problem(settings: GenerationSettings, backend: str) -> str | None:
    """Why ``backend`` cannot answer now (key first, then the endpoint), or None."""
    reg = settings.backends()
    return unavailable_reason(reg.spec(backend), backend)


def make_chat_llm(
    settings: GenerationSettings, backend: str | None = None, fake: bool = False
) -> ChatModel:
    """The chat / general-knowledge client of ``backend``: its generation config with the
    backend's chat temperature and completion limit."""
    name = backend or settings.default_backend
    if fake:
        return make_llm(settings, True, name)
    spec = settings.backends().spec(name)
    cfg = settings.llm(name).model_copy(
        update={"temperature": spec.chat_temperature, "max_tokens": spec.chat_max_tokens}
    )
    return LLMClient(cfg)


def chat_history_tokens(settings: GenerationSettings, backend: str) -> int:
    return int(settings.backends().spec(backend).chat_history_tokens)


def serving_retrieval(coll: Any) -> tuple[RetrievalSettings, str]:
    """The retrieval settings ask.py and the web UI serve for a collection: its app index (the
    collection's only index when it is not evaluated), or, while an evaluated collection's app
    index is not built yet, its evaluation index with a note saying so."""
    from ragsys.retrieval.embed import model_slug

    app = load_retrieval_settings(resolve_path(coll.app_retrieval_config))
    if not coll.has_app_index:
        return app, ""
    d = app.defaults
    npy = (
        resolve_path(app.paths.emb_dir)
        / model_slug(app.dense_model_name(d.dense_model))
        / f"{d.chunker}.npy"
    )
    if npy.exists():  # chunks and embeddings of the app index are built
        return app, ""
    extra = [d for d in coll.spec.app_docs if d not in coll.spec.eval_docs]
    note = (
        f"The app index (adding {', '.join(extra)}) is not built yet, so only the evaluated "
        "documents are searched; run scripts/run_app_collections.py to build it."
    )
    return load_retrieval_settings(resolve_path(coll.retrieval_config)), note


def build_engine(
    settings: GenerationSettings,
    retrieval_settings: RetrievalSettings,
    llm: ChatModel,
    *,
    require_threshold: bool = False,
    load_threshold_file: bool = True,
    allow_selection_fallback: bool = False,
    backend: str | None = None,
    llms: dict[str, ChatModel] | None = None,
    fake: bool = False,
) -> AnswerEngine:
    """An engine whose default client is ``llm`` (of ``backend``). Clients of other backends
    come from ``llms`` when given, else they are created on first use."""
    resources = RetrievalResources(retrieval_settings, rerank_cache=True)
    pool = dict(llms or {})

    def factory(name: str) -> ChatModel:
        if name not in pool:
            pool[name] = make_llm(settings, fake, name)
        return pool[name]

    make: Callable[[str], ChatModel] = factory
    return AnswerEngine(
        settings,
        retrieval_settings,
        llm,
        resources,
        load_threshold_file=load_threshold_file,
        require_threshold=require_threshold,
        allow_selection_fallback=allow_selection_fallback,
        backend=backend,
        llm_factory=make,
    )
