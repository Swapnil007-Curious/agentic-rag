"""Client-side rate limiting for hosted LLM endpoints, and Retry-After parsing.

:class:`RateLimiter` is a pair of token buckets (requests per minute and tokens per minute),
refilled continuously. Before a call the client reserves one request and an estimate of the
call's tokens (prompt estimate + ``max_tokens``); after the call the estimate is replaced by the
usage the provider reports, so the bucket follows real consumption. Buckets are shared by every
client of the same (base_url, model) in a process (:func:`shared_limiter`), because a provider
counts the account's calls, not one client's.

:func:`retry_after_seconds` reads the wait a provider asks for on a 429: the ``Retry-After``
header (seconds or an HTTP date) or the ``x-ratelimit-reset-*`` headers (``"2.84s"``,
``"1m26.4s"``) some providers send.
"""

from __future__ import annotations

import email.utils
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any


class RateLimiter:
    """Requests/minute and tokens/minute token buckets (``None`` = unlimited)."""

    def __init__(
        self,
        requests_per_minute: float | None,
        tokens_per_minute: float | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        now = clock()
        self._req = float(requests_per_minute or 0.0)  # start full
        self._tok = float(tokens_per_minute or 0.0)
        self._t = now
        self._lock = threading.Lock()
        self.waited_s = 0.0  # total time spent waiting (reported in logs)

    def _refill(self) -> None:
        now = self._clock()
        dt = max(0.0, now - self._t)
        self._t = now
        if self.rpm:
            self._req = min(self.rpm, self._req + dt * self.rpm / 60.0)
        if self.tpm:
            self._tok = min(self.tpm, self._tok + dt * self.tpm / 60.0)

    def acquire(self, tokens: int) -> float:
        """Wait until one request and ``tokens`` tokens are available, then take them.
        Returns the seconds waited. A reservation larger than the whole bucket waits for a
        full bucket (it cannot be split)."""
        waited = 0.0
        with self._lock:
            need_tok = float(min(tokens, self.tpm)) if self.tpm else 0.0
            while True:
                self._refill()
                short_req = (1.0 - self._req) if self.rpm else 0.0
                short_tok = (need_tok - self._tok) if self.tpm else 0.0
                if short_req <= 1e-9 and short_tok <= 1e-9:
                    break
                wait = 0.0
                if self.rpm and short_req > 0:
                    wait = max(wait, short_req * 60.0 / self.rpm)
                if self.tpm and short_tok > 0:
                    wait = max(wait, short_tok * 60.0 / self.tpm)
                wait = max(wait, 0.05)
                self._sleep(wait)
                waited += wait
            if self.rpm:
                self._req -= 1.0
            if self.tpm:
                self._tok -= need_tok
        self.waited_s += waited
        return waited

    def settle(self, reserved: int, actual: int) -> None:
        """Replace a reservation by the tokens actually used (may go into debt)."""
        if not self.tpm:
            return
        with self._lock:
            self._refill()
            self._tok += float(min(reserved, self.tpm)) - float(actual)

    def penalise(self, seconds: float) -> None:
        """After a 429: empty the buckets so that the next call waits about ``seconds``."""
        with self._lock:
            self._refill()
            if self.rpm:
                self._req = min(self._req, 1.0 - seconds * self.rpm / 60.0)
            if self.tpm:
                self._tok = min(self._tok, -seconds * self.tpm / 60.0)


_SHARED: dict[tuple[str, str], RateLimiter] = {}
_SHARED_LOCK = threading.Lock()


def shared_limiter(
    base_url: str, model: str, rpm: float | None, tpm: float | None
) -> RateLimiter | None:
    """The process-wide limiter of one endpoint and model (None when no limit is set)."""
    if not rpm and not tpm:
        return None
    key = (base_url.rstrip("/"), model)
    with _SHARED_LOCK:
        lim = _SHARED.get(key)
        if lim is None or (lim.rpm, lim.tpm) != (rpm, tpm):
            lim = RateLimiter(rpm, tpm)
            _SHARED[key] = lim
        return lim


_DURATION_RE = re.compile(
    r"^(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m(?!s))?(?:(?P<s>\d+(?:\.\d+)?)s)?$"
)
_MS_RE = re.compile(r"^(?P<ms>\d+(?:\.\d+)?)ms$")


def parse_duration(value: str) -> float | None:
    """'2.842s' -> 2.842, '1m26.4s' -> 86.4, '7m' -> 420, '120ms' -> 0.12, '30' -> 30."""
    v = value.strip().lower()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        pass
    m = _MS_RE.match(v)
    if m:
        return float(m.group("ms")) / 1000.0
    m = _DURATION_RE.match(v)
    if not m or not any(m.groupdict().values()):
        return None
    return (
        float(m.group("h") or 0) * 3600 + float(m.group("m") or 0) * 60 + float(m.group("s") or 0)
    )


def retry_after_seconds(
    headers: Mapping[str, Any] | None, now: float | None = None
) -> float | None:
    """Seconds to wait according to a 429 response's headers, or None when it says nothing."""
    if not headers:
        return None
    low = {str(k).lower(): str(v) for k, v in headers.items()}
    ra = low.get("retry-after")
    if ra:
        secs = parse_duration(ra)
        if secs is not None:
            return max(0.0, secs)
        try:
            when = email.utils.parsedate_to_datetime(ra).timestamp()
        except (TypeError, ValueError):
            when = None
        if when is not None:
            return max(0.0, when - (time.time() if now is None else now))
    for k in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        if k in low:
            secs = parse_duration(low[k])
            if secs is not None:
                return max(0.0, secs)
    return None
