"""Builds the chat assistant from the collection registry and configs/llm.yaml (used by
scripts/serve.py and scripts/bench_chat.py)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragsys.agents.config import GenerationSettings, load_generation_settings
from ragsys.agents.fake import agent_responder
from ragsys.agents.runtime import serving_retrieval
from ragsys.assistant.config import AssistantSettings, load_assistant_settings
from ragsys.assistant.grounded import UPLOAD_PROMPTS
from ragsys.assistant.llm import FakeStreamingChat, OpenAIStreamingChat, StreamingChat
from ragsys.assistant.service import AssistantCollection, ChatAssistant, ChatBackend
from ragsys.collections import Collection, load_collections
from ragsys.config import load_ingest_config, resolve_path
from ragsys.llm.backends import LLMFile
from ragsys.llm.client import ChatModel, LLMClient, Message
from ragsys.llm.fake import FakeLLM
from ragsys.profiles import get_profile
from ragsys.retrieval.embed import Encoder, SentenceTransformerEncoder

EXAMPLES_FILE = Path("configs/assistant_examples.json")  # written into the hosted build


def read_sources(path: Path) -> list[dict[str, Any]]:
    from ragsys.app.server import read_sources as _read

    return _read(path)


def sources_csv(coll: Collection, gen: GenerationSettings, served_app: bool) -> Path:
    """The SOURCES.csv of the documents the app serves (an app index's own file once built)."""
    if coll.has_app_index and served_app:
        ing = load_ingest_config(resolve_path(coll.app_ingest_config))
        app = resolve_path(ing.paths.sources_path())
        if app.exists():
            return app
    return resolve_path(gen.ui.sources_csv)


def collection_examples(coll: Collection, stored: dict[str, list[str]]) -> list[str]:
    """The collection's example questions: its configured ones, else answerable DEV questions,
    else those stored in configs/assistant_examples.json (a hosted build has no eval data)."""
    ui = coll.spec.ui
    if ui.examples:
        return list(ui.examples)
    from ragsys.app.server import example_questions

    found = example_questions(resolve_path(coll.questions_file), ui.n_examples)
    return found or list(stored.get(coll.name, []))


def stored_examples(path: Path = EXAMPLES_FILE) -> dict[str, list[str]]:
    p = resolve_path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): [str(x) for x in v] for k, v in data.items()}


def assistant_collections(
    collections_file: str | Path, settings: AssistantSettings
) -> list[AssistantCollection]:
    reg = load_collections(collections_file)
    stored = stored_examples()
    out: list[AssistantCollection] = []
    for name in reg.ordered():
        coll = Collection(name, reg.collections[name])
        gen = load_generation_settings(resolve_path(coll.generation_config))
        rs, note = serving_retrieval(coll)
        ui = coll.spec.ui
        sources = read_sources(sources_csv(coll, gen, served_app=not note))
        fallback = settings.licence_fallbacks.get(name, "")
        for src in sources:
            if not src.get("licence") and fallback:
                src["licence"] = fallback
        out.append(
            AssistantCollection(
                name=name,
                label=ui.label,
                processed_dir=resolve_path(rs.paths.processed_dir),
                emb_dir=resolve_path(rs.paths.emb_dir),
                profile=coll.spec.profile,
                description=ui.eyebrow,
                domain=ui.domain,
                notice=settings.notice(name) or ui.safety_note,
                examples=collection_examples(coll, stored),
                sources=sources,
                evaluated=coll.evaluated,
                generation=gen,
                abstention_message=ui.abstention_message,
            )
        )
    return out


def fake_chat_reply(messages: Sequence[Message]) -> str:
    """The offline stand-in's streamed reply (--fake-llm)."""
    last = messages[-1].get("content", "") if messages else ""
    if "<<<PASSAGE C1>>>" in last:
        return (
            "**Offline stand-in model.** The documents cover this [C1]. "
            "A related passage adds detail [C2]."
        )
    question = last.strip().splitlines()[-1][:200] if last.strip() else ""
    return (
        "**Offline stand-in model.** No documents were used for this reply.\n\n"
        f"You asked: {question}\n\n"
        "```python\nprint('hello')\n```\n\nInline math: $E = mc^2$."
    )


def _fake_json_reply(messages: Sequence[Message]) -> str:
    system = messages[0].get("content", "") if messages else ""
    if system.startswith(UPLOAD_PROMPTS.critic_system[:60]):
        return json.dumps({"verdict": "accept", "unsupported_sentences": [], "feedback": ""})
    if system.startswith(UPLOAD_PROMPTS.synth_system[:60]):
        swapped: list[Message] = [
            {"role": "system", "content": get_profile("cs_ai").generation.synth_system},
            *messages[1:],
        ]
        return agent_responder(swapped)
    return agent_responder(messages)


def fake_json_llm() -> ChatModel:
    return FakeLLM(_fake_json_reply, model="fake-agent-llm")


def chat_backends(
    llm_file: LLMFile,
    names: Sequence[str],
    *,
    fake: bool,
    queue_wait_s: float,
    history_budgets: dict[str, int] | None = None,
) -> list[ChatBackend]:
    out: list[ChatBackend] = []
    for name in names:
        spec = llm_file.spec(name)
        llm: StreamingChat
        if fake:
            llm = FakeStreamingChat(fake_chat_reply, backend=name, model=f"fake-{name}-stream")
        else:
            llm = OpenAIStreamingChat(spec, name, queue_wait_s=queue_wait_s)
        out.append(
            ChatBackend(
                name=name,
                llm=llm,
                label=spec.label,
                provider=spec.provider,
                privacy=spec.privacy,
                max_tokens=spec.chat_max_tokens,
                temperature=spec.chat_temperature,
                history_tokens=(history_budgets or {}).get(name, spec.chat_history_tokens),
                hosted=spec.requires_key,
            )
        )
    return out


def json_llm_factory(
    collections: Sequence[AssistantCollection],
    fallback: GenerationSettings,
    *,
    fake: bool,
    limiters: dict[str, Any] | None = None,
) -> Any:
    """Non-streaming clients for Verify and Grounded only: no response cache (messages are
    never written to disk), one retry, and a short wait limit so a busy provider fails fast;
    they share the streaming client's per-minute budget (``limiters``), which refuses instead
    of waiting long."""
    gens = {c.name: c.generation for c in collections if c.generation is not None}

    def make(backend: str, coll: AssistantCollection | None) -> ChatModel:
        if fake:
            return fake_json_llm()
        gen = gens.get(coll.name, fallback) if coll is not None else fallback
        cfg = gen.llm(backend).model_copy(
            update={"cache_dir": None, "max_retries": 1, "max_retry_wait_s": 10.0}
        )
        client = LLMClient(cfg, use_cache=False)
        shared = (limiters or {}).get(backend)
        if shared is not None:
            client.limiter = shared
        return client

    return make


def bge_encoder(settings: AssistantSettings) -> Encoder:
    return SentenceTransformerEncoder(
        settings.embedding_model, "cpu", 16, settings.query_instruction
    )


def build_assistant(
    collections_file: str | Path,
    llm_file: LLMFile,
    backends: Sequence[str],
    default_backend: str,
    *,
    fake_llm: bool = False,
    public: bool = False,
    settings: AssistantSettings | None = None,
    encoder: Encoder | None = None,
    backend_status: Any = None,
) -> ChatAssistant:
    s = settings or load_assistant_settings()
    colls = assistant_collections(collections_file, s)
    fallback = next(
        (c.generation for c in colls if c.generation is not None),
        None,
    )
    if fallback is None:
        raise ValueError("no collection with a generation config")
    chat = chat_backends(
        llm_file,
        backends,
        fake=fake_llm,
        queue_wait_s=s.limits.queue_wait_s,
        history_budgets=s.history.budget_tokens,
    )
    limiters = {b.name: getattr(b.llm, "limiter", None) for b in chat}
    return ChatAssistant(
        s,
        colls,
        encoder or bge_encoder(s),
        chat,
        default_backend,
        json_llm=json_llm_factory(colls, fallback, fake=fake_llm, limiters=limiters),
        backend_status=backend_status,
        public=public,
    )
