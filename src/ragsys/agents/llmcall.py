"""LLM calls of the agents: strict-JSON replies with one repair retry, and a per-call record of
latency (cache hits report the latency measured when the response was first computed) and token
usage."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

from ragsys.llm.client import ChatModel, LLMJSONError, LLMResponse, Message, extract_json

T = TypeVar("T")

REPAIR_PROMPT = (
    "Your previous reply could not be used ({error}). Reply again with exactly one JSON object "
    "in this format and nothing else:\n{schema}"
)


@dataclass
class LLMCall:
    role: str  # "synthesizer" | "critic" | ...
    kind: str  # "main" | "repair"
    ms: float  # reported latency (stored latency for cache hits)
    measured_ms: float  # wall time spent now
    cached: bool
    prompt_tokens: int
    completion_tokens: int
    ok: bool  # reply parsed and validated
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def timed_chat(
    llm: ChatModel, messages: Sequence[Message], role: str, kind: str
) -> tuple[LLMResponse, LLMCall]:
    t0 = time.perf_counter()
    resp = llm.chat(messages, json_mode=True)
    measured = (time.perf_counter() - t0) * 1000.0
    stored = getattr(resp, "stored_latency_ms", None)
    if resp.cached and stored is not None:
        reported = float(stored)
    elif resp.latency_ms and not resp.cached:
        reported = float(resp.latency_ms)
    else:
        reported = measured
    usage = resp.usage or {}
    call = LLMCall(
        role=role,
        kind=kind,
        ms=round(reported, 1),
        measured_ms=round(measured, 1),
        cached=bool(resp.cached),
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        ok=False,
    )
    return resp, call


def json_with_repair(
    llm: ChatModel,
    messages: Sequence[Message],
    validate: Callable[[dict[str, Any]], T],
    *,
    role: str,
    schema: str,
) -> tuple[T | None, list[LLMCall], str | None, str]:
    """Ask for a JSON object; on an unparseable or invalid reply, ask once more with the reply
    and a repair instruction appended. Returns (value or None, calls, last error, last raw)."""
    calls: list[LLMCall] = []
    resp, call = timed_chat(llm, messages, role, "main")
    calls.append(call)
    try:
        value = validate(extract_json(resp.content))
        call.ok = True
        return value, calls, None, resp.content
    except (LLMJSONError, ValueError, TypeError) as exc:
        error = _short(exc)
        call.error = error
    repair = [
        *messages,
        {"role": "assistant", "content": resp.content},
        {"role": "user", "content": REPAIR_PROMPT.format(error=error, schema=schema)},
    ]
    resp2, call2 = timed_chat(llm, repair, role, "repair")
    calls.append(call2)
    try:
        value = validate(extract_json(resp2.content))
        call2.ok = True
        return value, calls, None, resp2.content
    except (LLMJSONError, ValueError, TypeError) as exc:
        call2.error = _short(exc)
        return None, calls, call2.error, resp2.content


def _short(exc: Exception) -> str:
    msg = str(exc).replace("\n", " ")
    return f"{type(exc).__name__}: {msg[:160]}"


def as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
