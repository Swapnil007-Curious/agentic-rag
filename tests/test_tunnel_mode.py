"""Tunnel mode (scripts/serve.py --tunnel): loopback binding with public mode, any host name, and
client addresses from the tunnel's CF-Connecting-IP header; none of it without --tunnel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from assistant_helpers import assistant_settings, build_assistant
from ragsys.agents.config import default_generation_settings
from ragsys.app.server import CollectionEntry, create_multi_app
from ragsys.assistant.hosting import Binding, binding_from_env, resolve_binding
from ragsys.assistant.limits import client_address

ENV_KEYS = ("HOST", "PORT", "ALLOWED_HOSTS", "FRAME_ANCESTORS", "TRUST_PROXY", "PROXY_HOPS")
VERIFY = {"question": "q", "answer": "a", "passages": []}  # passes the limiter, then 400


def _no_engine() -> Any:
    raise AssertionError("the research engine must not be used")


def app_for(tmp: Path, binding: Binding, per_ip: int = 2) -> Any:
    settings = assistant_settings(
        rate_limits={"per_ip": [{"max": per_ip, "window_s": 600}], "global_daily_requests": 1000}
    )
    assistant, _llm = build_assistant(tmp, settings=settings)
    assistant.public = binding.public  # as scripts/serve.py builds it
    entry = CollectionEntry(
        name="books",
        settings=default_generation_settings(),
        engine_factory=_no_engine,
        label="Books",
    )
    return create_multi_app(
        [entry],
        "books",
        allowed_hosts=binding.allowed_hosts,
        assistant=assistant,
        public=binding.public,
        frame_ancestors=binding.frame_ancestors,
        trust_forwarded=binding.trust_forwarded,
        forwarded_hops=binding.forwarded_hops,
        client_ip_header=binding.client_ip_header,
    )


def statuses(client: TestClient, headers: list[dict[str, str]]) -> list[int]:
    return [client.post("/api/verify", json=VERIFY, headers=h).status_code for h in headers]


def test_tunnel_binding_and_unchanged_default() -> None:
    t = resolve_binding({}, 8000, tunnel=True)
    assert (t.host, t.port, t.public, t.tunnel) == ("127.0.0.1", 8000, True, True)
    assert t.allowed_hosts == ["*"] and t.trust_forwarded and t.forwarded_hops == 1
    assert t.client_ip_header == "cf-connecting-ip" and t.frame_ancestors == "'none'"
    # HOST does not move a tunnel off the loopback address; PORT and PROXY_HOPS still apply
    t2 = resolve_binding({"HOST": "0.0.0.0", "PORT": "9001", "PROXY_HOPS": "2"}, 8000, tunnel=True)
    assert (t2.host, t2.port, t2.forwarded_hops) == ("127.0.0.1", 9001, 2)
    # without --tunnel the binding is exactly what HOST decides
    for env in ({}, {"HOST": "127.0.0.1"}, {"HOST": "0.0.0.0", "TRUST_PROXY": "1"}):
        b = resolve_binding(env, 8000, tunnel=False)
        assert b == binding_from_env(env, 8000) and not b.tunnel and b.client_ip_header is None
    assert resolve_binding({}, 8000, tunnel=False).allowed_hosts == ["127.0.0.1", "localhost"]


def test_cf_connecting_ip_is_read_only_with_the_tunnel_header() -> None:
    h = {"cf-connecting-ip": "203.0.113.7", "x-forwarded-for": "198.51.100.1, 10.0.0.2"}
    tunnel = {"trust_forwarded": True, "hops": 1, "client_header": "cf-connecting-ip"}
    assert client_address("127.0.0.1", h, **tunnel) == "203.0.113.7"
    assert (
        client_address("127.0.0.1", {"cf-connecting-ip": "2001:DB8::1"}, **tunnel) == "2001:db8::1"
    )
    # not an address, or missing: X-Forwarded-For, then the peer
    assert (
        client_address("127.0.0.1", {**h, "cf-connecting-ip": "evil, 1.2.3.4"}, **tunnel)
        == "10.0.0.2"
    )
    assert client_address("127.0.0.1", {}, **tunnel) == "127.0.0.1"
    # hosted demo behind a proxy (TRUST_PROXY): X-Forwarded-For only, never CF-Connecting-IP
    assert client_address("10.0.0.9", h, trust_forwarded=True, hops=1) == "10.0.0.2"
    assert (
        client_address("10.0.0.9", {"cf-connecting-ip": "203.0.113.7"}, trust_forwarded=True)
        == "10.0.0.9"
    )
    # local: the peer only
    assert (
        client_address("127.0.0.1", h, trust_forwarded=False, client_header="cf-connecting-ip")
        == "127.0.0.1"
    )


def test_tunnel_app_accepts_any_host_and_limits_per_cf_client(tmp_path: Path) -> None:
    c = TestClient(
        app_for(tmp_path, resolve_binding({}, 8000, tunnel=True)),
        base_url="https://abc.trycloudflare.com",
    )
    assert c.get("/").status_code == 200 and 'id="composer-input"' in c.get("/").text
    assert c.get("/research").status_code == 404 and c.get("/api/health").json()["public"] is True
    a, b = {"CF-Connecting-IP": "203.0.113.7"}, {"CF-Connecting-IP": "203.0.113.8"}
    assert statuses(c, [a, a, a]) == [400, 400, 429]  # per visitor, not per tunnel
    assert statuses(c, [b]) == [400]
    # without the header the tunnel's X-Forwarded-For names the visitor
    x = {"X-Forwarded-For": "198.51.100.4"}
    assert statuses(c, [x, x, x]) == [400, 400, 429]


def test_without_tunnel_host_check_stays_and_cf_header_is_ignored(tmp_path: Path) -> None:
    local = TestClient(
        app_for(tmp_path / "local", resolve_binding({}, 8000, tunnel=False)),
        base_url="http://127.0.0.1:8000",
    )
    assert local.get("/").status_code == 200 and local.get("/research").status_code == 200
    evil = TestClient(local.app, base_url="https://abc.trycloudflare.com")
    assert evil.get("/").status_code == 400  # loopback host names only
    spoof = [{"CF-Connecting-IP": f"203.0.113.{i}"} for i in range(3)]
    assert statuses(local, spoof) == [400, 400, 429]  # one client: the peer
    space = resolve_binding({"HOST": "0.0.0.0", "TRUST_PROXY": "1"}, 7860, tunnel=False)
    pub = TestClient(app_for(tmp_path / "space", space), base_url="https://x.hf.space")
    same_proxy = [
        {"CF-Connecting-IP": f"203.0.113.{i}", "X-Forwarded-For": "198.51.100.9"} for i in range(3)
    ]
    assert statuses(pub, same_proxy) == [400, 400, 429]  # keyed by X-Forwarded-For only


def _run_serve(script, monkeypatch: pytest.MonkeyPatch, *args: str) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    serve = script("serve")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(serve.uvicorn, "run", lambda app, **kw: captured.update(app=app, kw=kw))
    assert serve.main(["--fake-llm", "--no-warm", *args]) == 0
    return captured


def test_serve_tunnel_flag_end_to_end(
    script, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    run = _run_serve(script, monkeypatch, "--tunnel", "--port", "8123")
    out = capsys.readouterr().out
    assert run["kw"]["host"] == "127.0.0.1" and run["kw"]["port"] == 8123
    assert run["kw"]["access_log"] is False
    assert "WARNING: TUNNEL MODE - this app is PUBLICLY REACHABLE through your tunnel." in out
    assert "tunnel, public mode" in out
    c = TestClient(run["app"], base_url="https://random-words.trycloudflare.com")
    assert c.get("/").status_code == 200 and c.get("/research").status_code == 404
    health = c.get("/api/health").json()
    assert health["public"] is True and all(b["hosted"] for b in health["backends"])
    a = {"CF-Connecting-IP": "203.0.113.50"}
    codes = statuses(c, [a] * 21)
    assert codes[:20] == [400] * 20 and codes[20] == 429  # 20 messages per 10 minutes
    assert statuses(c, [{"CF-Connecting-IP": "203.0.113.51"}]) == [400]


def test_serve_without_tunnel_is_unchanged(
    script, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    run = _run_serve(script, monkeypatch, "--port", "8124")
    out = capsys.readouterr().out
    assert run["kw"] == {"host": "127.0.0.1", "port": 8124, "log_level": "warning"}
    assert "TUNNEL MODE" not in out and "local only" in out
    c = TestClient(run["app"], base_url="http://127.0.0.1:8124")
    assert c.get("/research").status_code == 200 and c.get("/api/health").json()["public"] is False
    assert (
        TestClient(run["app"], base_url="https://random-words.trycloudflare.com")
        .get("/")
        .status_code
        == 400
    )
