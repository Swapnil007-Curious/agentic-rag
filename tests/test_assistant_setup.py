"""Assistant set-up: backends from configs/llm.yaml, the assistant's history budget, one shared
per-minute budget for streamed and JSON calls, and a refused budget reported as "busy"."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from assistant_helpers import build_assistant, make_client
from ragsys.agents.config import default_generation_settings
from ragsys.assistant.config import load_assistant_settings
from ragsys.assistant.llm import BUSY_MESSAGE, BudgetWait, OpenAIStreamingChat
from ragsys.assistant.setup import chat_backends, json_llm_factory
from ragsys.llm.backends import load_backends
from ragsys.llm.client import LLMClient


def test_backends_use_the_assistant_history_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    llm_file = load_backends("configs/llm.yaml")
    s = load_assistant_settings()
    backends = chat_backends(
        llm_file,
        ["local", "hosted"],
        fake=False,
        queue_wait_s=5.0,
        history_budgets=s.history.budget_tokens,
    )
    by = {b.name: b for b in backends}
    assert by["hosted"].history_tokens == 3000 and by["hosted"].hosted
    assert by["local"].history_tokens == llm_file.spec("local").chat_history_tokens
    assert (
        isinstance(by["hosted"].llm, OpenAIStreamingChat) and by["hosted"].llm.limiter is not None
    )
    assert by["local"].llm.limiter is None  # type: ignore[attr-defined]
    # the hosted request stays under the account's tokens per minute
    spec = llm_file.spec("hosted")
    worst = 600 + s.context_budget("hosted") + 3000 + s.limits.max_output_tokens
    assert spec.tokens_per_minute is not None and worst <= spec.tokens_per_minute
    # JSON clients (Verify, Grounded only) share the streaming client's budget and never cache
    make = json_llm_factory(
        [], default_generation_settings(), fake=False, limiters={"hosted": by["hosted"].llm.limiter}
    )  # type: ignore[attr-defined]
    client = make("hosted", None)
    assert isinstance(client, LLMClient) and client.limiter is by["hosted"].llm.limiter  # type: ignore[attr-defined]
    assert client.cache_dir is None and client.cfg.max_retries == 1


def test_refused_budget_is_busy_for_verify(tmp_path: Path) -> None:
    class Refusing:
        model = "x"

        def chat(self, *_a: Any, **_k: Any) -> Any:
            raise BudgetWait(40.0)

        def chat_json(self, *_a: Any, **_k: Any) -> Any:
            raise BudgetWait(40.0)

    assistant, _llm = build_assistant(tmp_path, json_llm=lambda _b, _c: Refusing())  # type: ignore[arg-type,return-value]
    c = make_client(assistant)
    body = {
        "question": "q",
        "answer": "Three months [C1].",
        "passages": [{"label": "C1", "chunk_id": "structural:testact:00001"}],
        "collection": "books",
    }
    r = c.post("/api/verify", json=body)
    assert r.status_code == 429 and r.json()["detail"] == BUSY_MESSAGE
