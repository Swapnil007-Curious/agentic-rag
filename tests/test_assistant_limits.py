"""Abuse protection: sliding-window limits per client, the global daily cap, client addresses
behind proxies, and where the server binds (HOST)."""

from __future__ import annotations

import pytest

from ragsys.assistant.config import RateLimitSettings, WindowLimit
from ragsys.assistant.hosting import binding_from_env
from ragsys.assistant.limits import AbuseGuard, DailyCap, SlidingWindowLimiter, client_address


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_sliding_window_20_per_10_minutes_and_200_per_day() -> None:
    clock = Clock()
    lim = SlidingWindowLimiter(
        [WindowLimit(max=20, window_s=600), WindowLimit(max=200, window_s=86400)], clock=clock
    )
    for _ in range(20):
        assert lim.take("1.2.3.4") is None
        clock.t += 1
    wait = lim.take("1.2.3.4")
    assert wait is not None and 570 < wait <= 600
    assert lim.take("5.6.7.8") is None  # other clients are not affected
    clock.t += 600
    assert lim.take("1.2.3.4") is None
    # the daily limit: 200 messages spread over the day
    day = SlidingWindowLimiter([WindowLimit(max=200, window_s=86400)], clock=clock)
    for _ in range(200):
        assert day.take("x") is None
        clock.t += 60
    assert day.take("x") is not None
    clock.t += 86400
    assert day.take("x") is None


def test_refused_requests_are_not_counted() -> None:
    clock = Clock()
    lim = SlidingWindowLimiter([WindowLimit(max=1, window_s=10)], clock=clock)
    assert lim.take("a") is None
    for _ in range(5):
        assert lim.take("a") is not None
    clock.t += 10.5
    assert lim.take("a") is None


def test_tracked_clients_are_bounded() -> None:
    lim = SlidingWindowLimiter([WindowLimit(max=1, window_s=10)], max_keys=3)
    for k in "abcd":
        lim.take(k)
    assert len(lim._events) == 3 and "a" not in lim._events


def test_daily_cap_resets_each_day() -> None:
    day = ["2026-11-12"]
    cap = DailyCap(2, today=lambda: day[0])
    assert cap.take() and cap.take() and not cap.take()
    day[0] = "2026-11-13"
    assert cap.take()


def test_abuse_guard_messages() -> None:
    clock = Clock()
    guard = AbuseGuard(
        RateLimitSettings(
            per_ip=[WindowLimit(max=1, window_s=600)],
            uploads_per_ip=[WindowLimit(max=1, window_s=3600)],
            global_daily_requests=2,
        ),
        clock=clock,
        today=lambda: "d",
    )
    assert guard.check_message("a") is None
    r = guard.check_message("a")
    assert r is not None and r.status == 429 and "10 minutes" in r.message
    assert guard.check_message("b") is None
    r = guard.check_message("c")
    assert r is not None and r.status == 503 and "try again later" in r.message
    assert guard.check_upload("a") is None and guard.check_upload("a") is not None


def test_client_address_behind_proxies() -> None:
    h = {"x-forwarded-for": "6.6.6.6, 1.1.1.1, 10.0.0.2"}
    assert client_address("10.0.0.9", h, trust_forwarded=False) == "10.0.0.9"
    assert client_address("10.0.0.9", h, trust_forwarded=True, hops=1) == "10.0.0.2"
    assert client_address("10.0.0.9", h, trust_forwarded=True, hops=2) == "1.1.1.1"
    assert client_address("10.0.0.9", h, trust_forwarded=True, hops=9) == "6.6.6.6"
    assert client_address(None, {}, trust_forwarded=True) == "unknown"


def test_binds_loopback_unless_host_is_set() -> None:
    b = binding_from_env({}, 8000)
    assert (b.host, b.port, b.public) == ("127.0.0.1", 8000, False)
    assert b.allowed_hosts == ["127.0.0.1", "localhost"] and b.frame_ancestors == "'none'"
    assert binding_from_env({"HOST": ""}).host == "127.0.0.1"
    local = binding_from_env({"HOST": "127.0.0.1", "PORT": "9000"})
    assert (local.host, local.port, local.public) == ("127.0.0.1", 9000, False)
    space = binding_from_env({"HOST": "0.0.0.0", "PORT": "7860", "TRUST_PROXY": "1"})
    assert (space.host, space.port, space.public) == ("0.0.0.0", 7860, True)
    assert space.allowed_hosts == ["*"] and space.frame_ancestors == "https://huggingface.co"
    assert space.trust_forwarded and space.forwarded_hops == 1
    custom = binding_from_env(
        {
            "HOST": "0.0.0.0",
            "ALLOWED_HOSTS": "a.hf.space, b.example",
            "FRAME_ANCESTORS": "'none'",
            "PROXY_HOPS": "2",
        }
    )
    assert (
        custom.allowed_hosts == ["a.hf.space", "b.example"] and custom.frame_ancestors == "'none'"
    )
    assert custom.forwarded_hops == 2 and not custom.trust_forwarded
    with pytest.raises(ValueError):
        binding_from_env({"HOST": "0.0.0.0", "FRAME_ANCESTORS": "* ; script-src *"})
    with pytest.raises(ValueError):
        binding_from_env({"HOST": "0.0.0.0", "FRAME_ANCESTORS": "http://insecure.example"})
