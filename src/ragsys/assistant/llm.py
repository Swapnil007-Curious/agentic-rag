"""Streaming chat completions for the assistant (OpenAI-compatible endpoints: the hosted provider
with ``stream=True``, Ollama through its OpenAI-compatible streaming), plus a deterministic fake.

Nothing is cached or logged: message contents never reach a log line or the disk. The API key
is read from ``LLM_API_KEY`` and passed to the SDK only. A provider rate limit (HTTP 429) or a
client-side per-minute budget that would need a long wait ends the request with
:class:`StreamFailure` ``busy`` ("try again in a minute"); another model is never used instead.
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol

import openai

from ragsys.llm.backends import BackendSpec
from ragsys.llm.client import API_KEY_ENV, Message
from ragsys.llm.ratelimit import RateLimiter, parse_duration, retry_after_seconds

BUSY_MESSAGE = "The model is busy right now. Please try again in a minute."
QUOTA_MESSAGE = "The hosted model's quota is used up for now. Please try again later."
UNAVAILABLE_MESSAGE = "The model could not be reached. Please try again in a moment."
QUOTA_WAIT_S = 600.0  # a Retry-After longer than this means a daily quota, not a busy minute


class StreamFailure(RuntimeError):
    """A request could not be answered; ``code`` is busy | quota | unavailable | error."""

    def __init__(self, code: str, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retry_after = retry_after


class TokenStream(Protocol):
    usage: dict[str, int]
    finish_reason: str | None
    queued_ms: float  # time spent waiting for the client-side rate limiter

    def __iter__(self) -> Iterator[str]: ...

    def close(self) -> None: ...


class StreamingChat(Protocol):
    backend: str
    model: str

    def stream(
        self, messages: Sequence[Message], *, max_tokens: int, temperature: float
    ) -> TokenStream: ...


def complete(llm: StreamingChat, messages: Sequence[Message], max_tokens: int) -> str:
    """A whole (short) reply, e.g. a conversation summary."""
    stream = llm.stream(messages, max_tokens=max_tokens, temperature=0.2)
    try:
        return "".join(stream).strip()
    finally:
        stream.close()


def estimate_tokens(messages: Sequence[Message], max_tokens: int) -> int:
    chars = sum(len(m.get("content", "")) for m in messages)
    return chars // 4 + 8 * len(messages) + max_tokens


class BudgetWait(Exception):
    """The client-side per-minute budget would need a longer wait than allowed."""

    def __init__(self, seconds: float) -> None:
        super().__init__(seconds)
        self.seconds = seconds


def bounded_limiter(rpm: float | None, tpm: float | None, max_wait_s: float) -> RateLimiter | None:
    """A per-minute token bucket that refuses (instead of sleeping) when a call would have to
    wait longer than ``max_wait_s``."""
    if not rpm and not tpm:
        return None

    def sleep(seconds: float) -> None:
        if seconds > max_wait_s:
            raise BudgetWait(seconds)
        time.sleep(seconds)

    return RateLimiter(rpm, tpm, sleep=sleep)


_TRY_AGAIN_RE = re.compile(r"try again in\s+([0-9hms.]+)", re.IGNORECASE)
_DAILY_RE = re.compile(r"per day|(?:TPD|RPD)", re.IGNORECASE)


def rate_limit_info(exc: openai.RateLimitError) -> tuple[bool, float | None]:
    """(daily quota?, seconds to wait) of a 429: the Retry-After header, else the wait the
    message asks for ("Please try again in 12.5s"), else the token bucket's reset. The reset of
    the request bucket is not used: it measures when the daily request count refills."""
    headers = getattr(exc.response, "headers", None) or {}
    low = {str(k).lower(): str(v) for k, v in headers.items()}
    wait = (
        retry_after_seconds({"retry-after": low["retry-after"]}) if "retry-after" in low else None
    )
    text = str(exc)
    if wait is None:
        m = _TRY_AGAIN_RE.search(text)
        wait = parse_duration(m.group(1).rstrip(".")) if m else None
    if wait is None and "x-ratelimit-reset-tokens" in low:
        wait = parse_duration(low["x-ratelimit-reset-tokens"])
    daily = bool(_DAILY_RE.search(text)) or (wait is not None and wait > QUOTA_WAIT_S)
    return daily, wait


def _failure(exc: Exception) -> StreamFailure:
    if isinstance(exc, openai.RateLimitError):
        daily, wait = rate_limit_info(exc)
        if daily:
            return StreamFailure("quota", QUOTA_MESSAGE, wait)
        return StreamFailure("busy", BUSY_MESSAGE, wait)
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return StreamFailure("unavailable", UNAVAILABLE_MESSAGE)
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code >= 500:
            return StreamFailure("unavailable", UNAVAILABLE_MESSAGE)
        return StreamFailure(
            "error", f"The model rejected the request (HTTP {exc.status_code}). Please rephrase."
        )
    return StreamFailure("error", "The model returned an unexpected error. Please try again.")


class _OpenAITokenStream:
    def __init__(
        self,
        raw: Any,
        on_done: Callable[[dict[str, int]], None],
        queued_ms: float,
    ) -> None:
        self._raw = raw
        self._on_done = on_done
        self._done = False
        self.usage: dict[str, int] = {}
        self.finish_reason: str | None = None
        self.queued_ms = queued_ms

    def _usage_of(self, chunk: Any) -> None:
        usage = getattr(chunk, "usage", None)
        if usage is None:  # some providers put it under x_groq.usage
            extra = getattr(chunk, "x_groq", None)
            usage = extra.get("usage") if isinstance(extra, dict) else None
        if usage is None:
            return
        get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, 0)
        self.usage = {
            "prompt_tokens": int(get("prompt_tokens") or 0),
            "completion_tokens": int(get("completion_tokens") or 0),
            "total_tokens": int(get("total_tokens") or 0),
        }

    def __iter__(self) -> Iterator[str]:
        try:
            for chunk in self._raw:
                self._usage_of(chunk)
                for choice in getattr(chunk, "choices", None) or []:
                    delta = getattr(choice, "delta", None)
                    text = getattr(delta, "content", None) if delta is not None else None
                    if text:
                        yield str(text)
                    if getattr(choice, "finish_reason", None):
                        self.finish_reason = str(choice.finish_reason)
        except openai.APIError as exc:
            raise _failure(exc) from exc
        finally:
            self._finish()

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            self._on_done(self.usage)

    def close(self) -> None:
        with contextlib.suppress(Exception):  # closing a finished stream
            self._raw.close()
        self._finish()


class OpenAIStreamingChat:
    """Streams a reply from one backend of configs/llm.yaml (no cache, no retries)."""

    def __init__(
        self,
        spec: BackendSpec,
        backend: str,
        *,
        queue_wait_s: float = 5.0,
        http_client: Any | None = None,
    ) -> None:
        self.backend = backend
        self.model = spec.model
        self.spec = spec
        key = os.environ.get(API_KEY_ENV) if spec.send_api_key else None
        self.missing_key = bool(spec.requires_key and not key)
        self._sdk = openai.OpenAI(
            base_url=spec.base_url.rstrip("/"),
            api_key=key or "not-needed",
            timeout=spec.timeout_s,
            max_retries=0,
            http_client=http_client,
        )
        self.queue_wait_s = queue_wait_s
        self.limiter = bounded_limiter(
            spec.requests_per_minute, spec.tokens_per_minute, queue_wait_s
        )
        self._stream_options = True  # dropped once if the endpoint rejects it

    def __repr__(self) -> str:  # never include credentials
        return f"OpenAIStreamingChat(backend={self.backend!r}, model={self.model!r})"

    def stream(
        self, messages: Sequence[Message], *, max_tokens: int, temperature: float
    ) -> _OpenAITokenStream:
        if self.missing_key:
            raise StreamFailure(
                "unavailable", f"The {self.backend} model is not configured on this server."
            )
        reserved = estimate_tokens(messages, max_tokens)
        t0 = time.perf_counter()
        if self.limiter is not None:
            try:
                self.limiter.acquire(reserved)
            except BudgetWait as refused:
                raise StreamFailure("busy", BUSY_MESSAGE, refused.seconds) from None
        queued_ms = (time.perf_counter() - t0) * 1000.0
        limiter = self.limiter

        def settle(usage: dict[str, int]) -> None:
            if limiter is not None:
                limiter.settle(reserved, usage.get("total_tokens") or reserved)

        params: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if self.spec.extra_body:
            params["extra_body"] = dict(self.spec.extra_body)
        create: Any = self._sdk.chat.completions.create  # stream=True -> an iterable of chunks
        retries = 0
        while True:
            extra = {"stream_options": {"include_usage": True}} if self._stream_options else {}
            try:
                raw = create(
                    model=self.model, messages=list(messages), stream=True, **params, **extra
                )
            except openai.BadRequestError as exc:
                if self._stream_options and "stream_options" in str(exc):
                    self._stream_options = False
                    continue
                settle({})
                raise _failure(exc) from exc
            except openai.RateLimitError as exc:
                daily, asked = rate_limit_info(exc)
                if limiter is not None:
                    limiter.penalise(min(asked or 30.0, 60.0))
                # a per-minute limit that clears within the allowed wait: wait, then try again
                if not daily and asked is not None and asked <= self.queue_wait_s and retries < 2:
                    retries += 1
                    time.sleep(asked + 0.25)
                    queued_ms += (asked + 0.25) * 1000.0
                    continue
                settle({})
                raise _failure(exc) from exc
            except openai.APIError as exc:
                settle({})
                raise _failure(exc) from exc
            return _OpenAITokenStream(raw, settle, queued_ms)

    def is_reachable(self, timeout_s: float = 3.0) -> bool:
        if self.missing_key:
            return False
        try:
            self._sdk.with_options(timeout=timeout_s).models.list()
        except Exception:
            return False
        return True


# --------------------------------------------------------------------------- fake


Responder = Callable[[Sequence[Message]], str]


def _default_reply(messages: Sequence[Message]) -> str:
    last = messages[-1].get("content", "") if messages else ""
    if "<<<PASSAGE C1>>>" in last:
        return "According to the documents, this is covered [C1]. A second point follows [C2]."
    return "Here is a general answer. It uses no documents."


class FakeTokenStream:
    def __init__(
        self,
        tokens: list[str],
        delay_s: float = 0.0,
        on_token: Callable[[int], None] | None = None,
    ) -> None:
        self._tokens = tokens
        self._delay = delay_s
        self._on_token = on_token
        self.pulled = 0
        self.closed = False
        self.usage: dict[str, int] = {}
        self.finish_reason: str | None = None
        self.queued_ms = 0.0

    def __iter__(self) -> Iterator[str]:
        for tok in self._tokens:
            if self.closed:
                return
            if self._delay:
                time.sleep(self._delay)
            self.pulled += 1
            if self._on_token is not None:  # tests: something happens while a token is produced
                self._on_token(self.pulled)
            yield tok
        self.finish_reason = "stop"
        self.usage = {"prompt_tokens": 10, "completion_tokens": len(self._tokens)}
        self.usage["total_tokens"] = self.usage["prompt_tokens"] + len(self._tokens)

    def close(self) -> None:
        self.closed = True


def split_tokens(text: str) -> list[str]:
    """Word-sized tokens whose concatenation is ``text``."""
    out: list[str] = []
    cur = ""
    for ch in text:
        cur += ch
        if ch in " \n":
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


class FakeStreamingChat:
    """Streams canned replies word by word (offline tests and --fake-llm)."""

    def __init__(
        self,
        responder: Responder | Sequence[str] | None = None,
        *,
        backend: str = "fake",
        model: str = "fake-stream-llm",
        delay_s: float = 0.0,
        fail: StreamFailure | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self._responder = responder
        self.delay_s = delay_s
        self.fail = fail
        self.on_token = on_token
        self.calls: list[dict[str, Any]] = []
        self.streams: list[FakeTokenStream] = []

    def stream(
        self, messages: Sequence[Message], *, max_tokens: int, temperature: float
    ) -> FakeTokenStream:
        self.calls.append(
            {"messages": list(messages), "max_tokens": max_tokens, "temperature": temperature}
        )
        if self.fail is not None:
            raise self.fail
        r = self._responder
        if r is None:
            text = _default_reply(messages)
        elif callable(r):
            text = r(messages)
        else:
            text = r[(len(self.calls) - 1) % len(r)]
        s = FakeTokenStream(split_tokens(text), self.delay_s, self.on_token)
        self.streams.append(s)
        return s
