"""One chat client for every OpenAI-compatible endpoint (the ``openai`` SDK with ``base_url``).

* Ollama by default (``http://localhost:11434/v1``); any compatible provider via ``base_url`` and
  ``model`` in configs/llm.yaml.
* The API key is read from the ``LLM_API_KEY`` environment variable only. It is passed to the
  SDK and nowhere else: it is not stored on this object, logged, cached or written to reports.
* JSON mode (``response_format={"type": "json_object"}``) with a tolerant JSON extractor as
  fallback; retries with exponential backoff on rate limits, timeouts, connection errors and
  5xx responses (at most ``max_retries``); per-call latency and token usage; an on-disk response
  cache keyed by (base_url, model, messages, params), so reruns are free and reproducible.
* A cache hit reports ``latency_ms = 0`` (no model time was spent now) and carries the latency
  measured when the response was first computed in ``stored_latency_ms``;
  :attr:`LLMResponse.reported_latency_ms` gives the latency to report either way, so latency
  figures do not depend on whether the cache was warm (as for the rerank cache).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx2
import openai
from pydantic import Field

from ragsys.config import StrictModel, load_yaml_config, resolve_path

API_KEY_ENV = "LLM_API_KEY"
log = logging.getLogger("ragsys.llm")

Message = dict[str, str]


class LLMConfig(StrictModel):
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:3b"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=512, gt=0)
    seed: int | None = 0  # sent when not null (Ollama and OpenAI accept it)
    json_mode: bool = True  # request response_format=json_object for chat_json()
    timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=5, ge=0, le=5)
    backoff_initial_s: float = Field(default=1.0, ge=0)
    backoff_max_s: float = Field(default=30.0, ge=0)
    cache_dir: Path | None = Path("data/processed/llm_cache")


def load_llm_config(path: str | Path) -> LLMConfig:
    return load_yaml_config(path, LLMConfig)


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False
    attempts: int = 1
    stored_latency_ms: float | None = None  # cache hits: latency measured on the original call

    @property
    def reported_latency_ms(self) -> float:
        """Measured latency of a fresh call, or the originally measured latency of a cache hit."""
        if self.cached and self.stored_latency_ms is not None:
            return self.stored_latency_ms
        return self.latency_ms


class ChatModel(Protocol):
    model: str

    def chat(self, messages: Sequence[Message], *, json_mode: bool = False) -> LLMResponse: ...

    def chat_json(self, messages: Sequence[Message]) -> tuple[dict[str, Any], LLMResponse]: ...


class LLMJSONError(ValueError):
    """The model's reply contained no parseable JSON object."""


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model reply: plain JSON, a ```json fenced block, or the first
    balanced ``{...}`` in surrounding prose."""
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
        start = text.find("{", start + 1)
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise LLMJSONError(f"no JSON object in model reply: {text[:200]!r}")


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


_RETRYABLE: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class LLMClient:
    """Chat completions against an OpenAI-compatible endpoint with retries and caching."""

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        http_client: httpx2.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        use_cache: bool = True,
    ) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self.base_url = cfg.base_url.rstrip("/")
        self._sleep = sleep
        self.cache_dir = resolve_path(cfg.cache_dir) if (use_cache and cfg.cache_dir) else None
        self.calls: list[dict[str, Any]] = []  # per-call timing / usage records (no secrets)
        self._sdk = openai.OpenAI(
            base_url=self.base_url,
            # Ollama ignores the key; the SDK only requires a non-empty string.
            api_key=os.environ.get(API_KEY_ENV) or "not-needed",
            timeout=cfg.timeout_s,
            max_retries=0,  # retries are handled here, with logging and a fixed cap
            http_client=http_client,
        )

    def __repr__(self) -> str:  # never include credentials
        return f"LLMClient(base_url={self.base_url!r}, model={self.model!r})"

    # ------------------------------------------------------------------ helpers

    def _params(self, json_mode: bool) -> dict[str, Any]:
        params: dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.seed is not None:
            params["seed"] = self.cfg.seed
        if json_mode and self.cfg.json_mode:
            params["response_format"] = {"type": "json_object"}
        return params

    def cache_key(self, messages: Sequence[Message], params: dict[str, Any]) -> str:
        blob = json.dumps(
            {
                "base_url": self.base_url,
                "model": self.model,
                "messages": list(messages),
                "params": params,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None

    def is_reachable(self, timeout_s: float = 3.0) -> bool:
        """True when the endpoint answers a model listing (cheap health check)."""
        try:
            self._sdk.with_options(timeout=timeout_s).models.list()
        except Exception:
            return False
        return True

    # ------------------------------------------------------------------ calls

    def chat(self, messages: Sequence[Message], *, json_mode: bool = False) -> LLMResponse:
        params = self._params(json_mode)
        key = self.cache_key(messages, params)
        path = self._cache_path(key)
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            resp = LLMResponse(
                content=data["response"]["content"],
                model=data["response"]["model"],
                usage=data["response"].get("usage", {}),
                latency_ms=0.0,
                cached=True,
                stored_latency_ms=_float_or_none(data.get("latency_ms")),
            )
            self._record(resp)
            return resp

        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                completion = self._sdk.chat.completions.create(
                    model=self.model,
                    messages=list(messages),  # type: ignore[arg-type]
                    **params,
                )
                break
            except _RETRYABLE as exc:
                if attempt > self.cfg.max_retries:
                    log.error("LLM call failed after %d attempts: %s", attempt, type(exc).__name__)
                    raise
                delay = min(self.cfg.backoff_max_s, self.cfg.backoff_initial_s * 2 ** (attempt - 1))
                log.warning(
                    "LLM call attempt %d failed (%s); retrying in %.1fs",
                    attempt,
                    type(exc).__name__,
                    delay,
                )
                self._sleep(delay)
        latency = (time.perf_counter() - t0) * 1000.0
        choice = completion.choices[0] if completion.choices else None
        content = (choice.message.content if choice and choice.message else None) or ""
        usage = {}
        if completion.usage is not None:
            usage = {
                "prompt_tokens": int(completion.usage.prompt_tokens or 0),
                "completion_tokens": int(completion.usage.completion_tokens or 0),
                "total_tokens": int(completion.usage.total_tokens or 0),
            }
        resp = LLMResponse(
            content=content,
            model=completion.model or self.model,
            usage=usage,
            latency_ms=round(latency, 1),
            attempts=attempt,
        )
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "request": {
                    "base_url": self.base_url,
                    "model": self.model,
                    "messages": list(messages),
                    "params": params,
                },
                "response": {"content": content, "model": resp.model, "usage": usage},
                "latency_ms": resp.latency_ms,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
        self._record(resp)
        return resp

    def chat_json(self, messages: Sequence[Message]) -> tuple[dict[str, Any], LLMResponse]:
        resp = self.chat(messages, json_mode=True)
        return extract_json(resp.content), resp

    def _record(self, resp: LLMResponse) -> None:
        rec = {
            "model": resp.model,
            "latency_ms": resp.latency_ms,
            "reported_latency_ms": resp.reported_latency_ms,
            "usage": resp.usage,
            "cached": resp.cached,
            "attempts": resp.attempts,
        }
        self.calls.append(rec)
        log.info(
            "LLM call model=%s cached=%s attempts=%d latency_ms=%.1f tokens=%s",
            resp.model,
            resp.cached,
            resp.attempts,
            resp.latency_ms,
            resp.usage.get("total_tokens", "?"),
        )
