"""Deterministic stand-in for :class:`ragsys.llm.client.LLMClient` (no network)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from ragsys.llm.client import LLMResponse, Message, extract_json

Responder = Callable[[Sequence[Message]], str]


def _draft_responder(messages: Sequence[Message]) -> str:
    """Canned reply for question-drafting prompts: a JSON draft built from the prompt."""
    prompt = "\n".join(m.get("content", "") for m in messages)
    qtype = re.search(r'"type":\s*"([a-z_]+)"', prompt)
    node = re.search(r"PROVISION \[([^\]]+)\]", prompt)
    text = re.search(r'"""\n(.*?)\n"""', prompt, re.DOTALL)
    words = [w for w in re.findall(r"[A-Za-z]{5,}", text.group(1) if text else "")][:3]
    topic = " ".join(words).lower() or "this provision"
    return json.dumps(
        {
            "question": f"What does the law say about {topic}?",
            "gold_answer": f"See {node.group(1) if node else 'the provision'}.",
            "type": qtype.group(1) if qtype else "factual",
            "difficulty": "medium",
        }
    )


class FakeLLM:
    """Returns canned replies: a list (cycled), a function of the messages, or by default a
    deterministic draft-question JSON derived from the prompt."""

    def __init__(
        self, responses: Sequence[str] | Responder | None = None, model: str = "fake-llm"
    ) -> None:
        self.model = model
        self._responses = responses
        self.calls: list[list[Message]] = []

    def chat(self, messages: Sequence[Message], *, json_mode: bool = False) -> LLMResponse:
        self.calls.append(list(messages))
        if self._responses is None:
            content = _draft_responder(messages)
        elif callable(self._responses):
            content = self._responses(messages)
        else:
            content = self._responses[(len(self.calls) - 1) % len(self._responses)]
        return LLMResponse(content=content, model=self.model, usage={"total_tokens": 0})

    def chat_json(self, messages: Sequence[Message]) -> tuple[dict[str, Any], LLMResponse]:
        resp = self.chat(messages, json_mode=True)
        return extract_json(resp.content), resp
