"""Abuse protection for public hosting: per-client sliding-window limits, a global daily request
cap and the client address behind a reverse proxy. Everything is in memory; no address or
message is logged or written."""

from __future__ import annotations

import datetime as _dt
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ragsys.assistant.config import RateLimitSettings, WindowLimit

MAX_TRACKED_CLIENTS = 50_000


class SlidingWindowLimiter:
    """Allows at most ``limit.max`` events per ``limit.window_s`` for each key, for every limit
    given (e.g. 20 per 10 minutes and 200 per day)."""

    def __init__(
        self,
        limits: Sequence[WindowLimit],
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = MAX_TRACKED_CLIENTS,
    ) -> None:
        self.limits = list(limits)
        self._clock = clock
        self._max_keys = max_keys
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._horizon = max((lim.window_s for lim in self.limits), default=0.0)

    def take(self, key: str) -> float | None:
        """Record one event for ``key`` and return None, or return the seconds to wait (nothing
        is recorded then)."""
        now = self._clock()
        with self._lock:
            q = self._events.get(key)
            if q is None:
                q = deque()
                self._events[key] = q
            self._events.move_to_end(key)
            while q and now - q[0] >= self._horizon:
                q.popleft()
            wait = 0.0
            for lim in self.limits:
                recent = [t for t in q if now - t < lim.window_s]
                if len(recent) >= lim.max:
                    wait = max(wait, lim.window_s - (now - recent[-lim.max]))
            if wait > 0:
                return wait
            q.append(now)
            while len(self._events) > self._max_keys:
                self._events.popitem(last=False)
            return None


class DailyCap:
    """At most ``cap`` requests per UTC day across all clients."""

    def __init__(self, cap: int, *, today: Callable[[], str] | None = None) -> None:
        self.cap = cap
        self._today = today or (lambda: _dt.datetime.now(_dt.UTC).date().isoformat())
        self._day = ""
        self.count = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            day = self._today()
            if day != self._day:
                self._day, self.count = day, 0
            if self.count >= self.cap:
                return False
            self.count += 1
            return True


def client_address(
    peer: str | None, headers: Mapping[str, str], *, trust_forwarded: bool, hops: int = 1
) -> str:
    """The client address: the connecting peer, or behind ``hops`` trusted reverse proxies the
    address the outermost of them saw (the ``hops``-th entry from the right of
    X-Forwarded-For; entries further left can be forged by the client)."""
    if trust_forwarded:
        xff = headers.get("x-forwarded-for", "")
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[max(0, len(parts) - max(hops, 1))]
    return peer or "unknown"


def _minutes(seconds: float) -> str:
    m = max(1, round(seconds / 60.0))
    return "about a minute" if m == 1 else f"about {m} minutes"


@dataclass
class Refusal:
    status: int
    code: str
    message: str
    retry_after: float | None = None


class AbuseGuard:
    """Per-client message and upload limits plus the global daily cap."""

    def __init__(
        self,
        settings: RateLimitSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], str] | None = None,
    ) -> None:
        self.messages = SlidingWindowLimiter(settings.per_ip, clock=clock)
        self.uploads = SlidingWindowLimiter(settings.uploads_per_ip, clock=clock)
        self.daily = DailyCap(settings.global_daily_requests, today=today)

    def check_message(self, client: str) -> Refusal | None:
        wait = self.messages.take(client)
        if wait is not None:
            return Refusal(
                429,
                "rate_limited",
                f"You have sent many messages in a short time. Please try again in {_minutes(wait)}.",
                wait,
            )
        if not self.daily.take():
            return Refusal(
                503,
                "daily_cap",
                "The assistant is very busy today and has reached its daily limit. Please try again later.",
            )
        return None

    def check_upload(self, client: str) -> Refusal | None:
        wait = self.uploads.take(client)
        if wait is not None:
            return Refusal(
                429,
                "rate_limited",
                f"Too many uploads in a short time. Please try again in {_minutes(wait)}.",
                wait,
            )
        return None
