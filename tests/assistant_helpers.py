"""Helpers for the chat assistant tests: a small app index embedded with the dependency-free
hashing encoder, a fake streaming LLM, the assistant and its app, and an SSE parser."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from agent_helpers import TEXTS, make_chunk
from ragsys.agents.config import default_generation_settings
from ragsys.app.server import CollectionEntry, create_multi_app
from ragsys.assistant.config import AssistantSettings
from ragsys.assistant.llm import FakeStreamingChat, Responder, StreamFailure
from ragsys.assistant.service import AssistantCollection, ChatAssistant, ChatBackend
from ragsys.assistant.setup import fake_json_llm
from ragsys.chunking.base import Chunk
from ragsys.llm.client import ChatModel
from ragsys.retrieval.corpus import texts_hash
from ragsys.retrieval.embed import model_slug
from ragsys.retrieval.fakes import HashingEncoder

ENCODER = "fake:hash-64"
SOURCES = [
    {
        "title": "The Test Protection Act",
        "authors": "Test Parliament",
        "edition": "",
        "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_url": "https://example.invalid/act.pdf",
    }
]


def write_index(processed: Path, chunks: Sequence[Chunk], encoder_name: str = ENCODER) -> None:
    """chunks/structural.jsonl plus embeddings with their sidecar, as build_index.py writes."""
    (processed / "chunks").mkdir(parents=True, exist_ok=True)
    with (processed / "chunks" / "structural.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")
    enc = HashingEncoder(int(encoder_name.rsplit("-", 1)[1]))
    emb = enc.encode([c.text for c in chunks], is_query=False)
    d = processed / "emb" / model_slug(encoder_name)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "structural.npy", emb)
    meta = {
        "model": encoder_name,
        "chunk_ids": [c.chunk_id for c in chunks],
        "texts_sha256": texts_hash([c.text for c in chunks]),
    }
    (d / "structural.json").write_text(json.dumps(meta), encoding="utf-8")


def assistant_settings(**update: Any) -> AssistantSettings:
    data: dict[str, Any] = {
        "embedding_model": ENCODER,
        "context_tokens": {"hosted": 2048, "local": 1024},
        "gate": {
            "default_threshold": 0.5,
            "upload_threshold": 0.5,
            "thresholds": {"books": 0.5},
            "follow_up_max_words": 12,
            "search_k": 10,
            "max_passages": 4,
        },
        "notices": {"books": "Not legal advice."},
        "general_examples": ["Explain big-O notation with examples"],
    }
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return AssistantSettings.model_validate(data)


def build_assistant(
    tmp: Path,
    *,
    responder: Responder | Sequence[str] | None = None,
    settings: AssistantSettings | None = None,
    fail: StreamFailure | None = None,
    on_token: Callable[[int], None] | None = None,
    json_llm: Callable[[str, AssistantCollection | None], ChatModel] | None = None,
    tokenizer: Any = None,
    history_tokens: int = 3000,
    texts: Sequence[str] = TEXTS,
) -> tuple[ChatAssistant, FakeStreamingChat]:
    processed = tmp / "books_processed"
    write_index(processed, [make_chunk(i, t, 60) for i, t in enumerate(texts, start=1)])
    coll = AssistantCollection(
        name="books",
        label="Books",
        processed_dir=processed,
        emb_dir=processed / "emb",
        profile="consumer_law",
        description="A test act",
        notice="Not legal advice.",
        examples=["Within how many days must a complaint be decided?"],
        sources=SOURCES,
        evaluated=True,
    )
    llm = FakeStreamingChat(
        responder, backend="hosted", model="fake-hosted", fail=fail, on_token=on_token
    )
    backend = ChatBackend(
        "hosted",
        llm,
        label="Hosted: fake",
        provider="Fake",
        privacy="sent",
        hosted=True,
        max_tokens=2048,
        temperature=0.3,
        history_tokens=history_tokens,
    )
    kwargs: dict[str, Any] = {}
    if tokenizer is not None:
        kwargs["tokenizer"] = lambda: tokenizer
    assistant = ChatAssistant(
        settings or assistant_settings(),
        [coll],
        HashingEncoder(64),
        [backend],
        "hosted",
        json_llm=json_llm or (lambda _b, _c: fake_json_llm()),
        **kwargs,
    )
    return assistant, llm


def _no_engine() -> Any:
    raise AssertionError("the research engine must not be used by the assistant")


def make_client(
    assistant: ChatAssistant, *, public: bool = False, frame_ancestors: str = "'none'"
) -> TestClient:
    entry = CollectionEntry(
        name="books",
        settings=default_generation_settings(),
        engine_factory=_no_engine,
        label="Books",
        evaluated=True,
    )
    app = create_multi_app(
        [entry],
        "books",
        allowed_hosts=["*"] if public else ["127.0.0.1", "localhost", "testserver"],
        assistant=assistant,
        public=public,
        frame_ancestors=frame_ancestors,
    )
    return TestClient(app, base_url="http://127.0.0.1:8000")


def sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for block in text.split("\n\n"):
        name, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            out.append((name, json.loads("\n".join(data))))
    return out


def chat(client: TestClient, content: str, **extra: Any) -> list[tuple[str, dict[str, Any]]]:
    body = {"messages": [{"role": "user", "content": content}], "collection": "books", **extra}
    r = client.post("/api/chat", json=body)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    return sse_events(r.text)


def by_name(events: list[tuple[str, dict[str, Any]]], name: str) -> list[dict[str, Any]]:
    return [d for n, d in events if n == name]
