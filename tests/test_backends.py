"""Generator backends: configs/llm.yaml layouts, per-backend LLM settings and context budgets,
cache keys, the client-side rate limiter and Retry-After, the API key never appearing in logs,
caches or reprs, and evaluations that never fall back from the hosted to the local backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from ragsys.agents.config import default_generation_settings, load_generation_settings
from ragsys.agents.runtime import llm_unreachable_message, make_llm
from ragsys.eval.generation import system_fingerprint
from ragsys.llm.backends import BackendUnavailable, load_backends, require_backend
from ragsys.llm.client import API_KEY_ENV, LLMClient, LLMConfig, QuotaExceeded
from ragsys.llm.ratelimit import RateLimiter, parse_duration, retry_after_seconds, shared_limiter

SECRET = "gsk-test-SECRET-never-log-0123456789"


def completion(content: str = '{"ok": true}') -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "hosted-model",
        "choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
    }


class Transport:
    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        item = self.script[min(len(self.requests) - 1, len(self.script) - 1)]
        if isinstance(item, tuple):  # (status, headers)
            return httpx2.Response(item[0], headers=item[1], json={"error": {"message": "slow"}})
        return httpx2.Response(
            200,
            json=completion(item),
            headers={"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "7950"},
        )


def client(
    tmp_path: Path, script: list[Any], **cfg: Any
) -> tuple[LLMClient, Transport, list[float]]:
    t = Transport(script)
    sleeps: list[float] = []
    c = LLMClient(
        LLMConfig(base_url="http://hosted.test/v1", model="m", cache_dir=tmp_path / "c", **cfg),
        http_client=httpx2.Client(transport=httpx2.MockTransport(t)),
        sleep=sleeps.append,
    )
    return c, t, sleeps


MSGS = [{"role": "user", "content": "hello"}]


# ----------------------------------------------------------------------------- configuration


def test_llm_yaml_defines_local_and_hosted_backends() -> None:
    reg = load_backends("configs/llm.yaml")
    assert reg.default_backend == "local" and reg.names == ["local", "hosted"]
    local, hosted = reg.spec("local"), reg.spec("hosted")
    assert (local.base_url, local.model) == ("http://localhost:11434/v1", "qwen2.5:3b")
    assert not local.send_api_key and not local.requires_key
    assert hosted.base_url == "https://api.groq.com/openai/v1" and hosted.requires_key
    assert hosted.model == "openai/gpt-oss-120b" and hosted.model_selected_on == "2026-11-12"
    assert "models endpoint" in hosted.model_selection and "Llama" in hosted.model_selection
    assert (hosted.context_budget_tokens, hosted.revision_budget_tokens) == (2048, 3072)
    assert (local.context_budget_tokens, local.revision_budget_tokens) == (None, None)
    assert hosted.requests_per_minute and hosted.tokens_per_minute
    with pytest.raises(KeyError):
        reg.spec("nope")


def test_legacy_single_endpoint_layout(tmp_path: Path) -> None:
    p = tmp_path / "llm.yaml"
    p.write_text(
        yaml.safe_dump({"base_url": "http://x.invalid/v1", "model": "m"}), encoding="utf-8"
    )
    reg = load_backends(p)
    assert reg.names == ["local"] and reg.client_config().model == "m"


def test_generation_settings_per_backend() -> None:
    s = default_generation_settings()
    local, hosted = s.llm(), s.llm("hosted")
    assert (local.model, local.temperature, local.max_tokens) == ("qwen2.5:3b", 0.0, 512)
    assert hosted.model == "openai/gpt-oss-120b" and hosted.temperature == 0.0
    assert hosted.max_tokens == 1024 and hosted.extra_body == {
        "reasoning_effort": "low",
        "include_reasoning": False,
    }
    assert (s.context_for().budget_tokens, s.context_for().revision_budget_tokens) == (1024, 2048)
    hc = s.context_for("hosted")
    assert (hc.budget_tokens, hc.revision_budget_tokens) == (2048, 3072)
    assert s.backend_of("G2") == "local" and s.backend_of("G3") == "local"
    assert s.backend_of("G2H") == "hosted" and s.system("G4H").ablation_of == "G4"
    assert s.retrieval_id("G2H") == s.retrieval_id("G2")
    # the ablation systems hash their own backend; the existing ones are unchanged (snapshot test)
    assert system_fingerprint(s, "G2H") != system_fingerprint(s, "G2")
    cs = load_generation_settings("configs/collections/cs_ai/generation.yaml")
    assert cs.retrieval_id("G2H") == cs.retrieval_id("G2") == "selected:best_overall"


def test_cache_keys_name_the_backend_except_local(tmp_path: Path) -> None:
    local = LLMClient(
        LLMConfig(base_url="http://h/v1", model="m", backend="local"), use_cache=False
    )
    legacy = LLMClient(LLMConfig(base_url="http://h/v1", model="m"), use_cache=False)
    hosted = LLMClient(
        LLMConfig(base_url="http://h/v1", model="m", backend="hosted"), use_cache=False
    )
    p = local._params(True)
    assert local.cache_key(MSGS, p) == legacy.cache_key(MSGS, p) != hosted.cache_key(MSGS, p)


# ----------------------------------------------------------------------------- rate limiting


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def test_token_bucket_limits_requests_and_tokens() -> None:
    clk = Clock()
    lim = RateLimiter(2, 1000, clock=clk.now, sleep=clk.sleep)
    assert lim.acquire(400) == 0 and lim.acquire(400) == 0
    waited = lim.acquire(400)  # third request in the same minute: waits for the bucket
    assert waited >= 25 and clk.slept
    lim.settle(400, 100)  # actual usage lower than reserved: tokens refunded
    clk2 = Clock()
    tok = RateLimiter(None, 600, clock=clk2.now, sleep=clk2.sleep)
    tok.acquire(600)
    assert tok.acquire(300) == pytest.approx(30, abs=1)  # 300 tokens at 600/min
    tok.penalise(10)
    assert tok.acquire(1) >= 9
    assert shared_limiter("http://a/v1", "m", None, None) is None
    assert shared_limiter("http://a/v1/", "m", 30, 8000) is shared_limiter(
        "http://a/v1", "m", 30, 8000
    )


def test_retry_after_parsing() -> None:
    assert parse_duration("2.842s") == pytest.approx(2.842)
    assert parse_duration("1m26.4s") == pytest.approx(86.4)
    assert parse_duration("7m") == 420 and parse_duration("120ms") == pytest.approx(0.12)
    assert retry_after_seconds({"Retry-After": "17"}) == 17
    assert retry_after_seconds({"x-ratelimit-reset-tokens": "3.5s"}) == 3.5
    assert retry_after_seconds({}) is None


def test_client_respects_retry_after_and_stops_on_a_long_wait(tmp_path: Path) -> None:
    c, _t, sleeps = client(tmp_path, [(429, {"retry-after": "7"}), '{"a": 1}'])
    r = c.chat(MSGS)
    assert r.content == '{"a": 1}' and r.attempts == 2 and sleeps == [7.0]
    assert c.last_rate_limit["x-ratelimit-limit-tokens"] == "8000"
    c2, _t, sleeps2 = client(tmp_path / "b", [(429, {"retry-after": "3600"})], max_retry_wait_s=120)
    with pytest.raises(QuotaExceeded, match="quota"):
        c2.chat(MSGS)
    assert sleeps2 == []  # no hour-long sleep


def test_client_rate_limiter_is_used(tmp_path: Path) -> None:
    c, _t, _s = client(tmp_path, ['{"a": 1}'], requests_per_minute=30, tokens_per_minute=8000)
    assert c.limiter is not None
    c.chat(MSGS)
    assert c.limiter.tpm == 8000


# ----------------------------------------------------------------------------- the key


def test_key_never_logged_cached_or_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(API_KEY_ENV, SECRET)
    caplog.set_level(logging.DEBUG)
    c, t, _s = client(
        tmp_path, [(429, {"retry-after": "1"}), '{"a": 1}'], backend="hosted", send_api_key=True
    )
    c.chat(MSGS)
    c.chat(MSGS)  # cache hit
    assert t.requests[-1].headers["authorization"] == f"Bearer {SECRET}"  # sent to the API only
    logged = "\n".join(f"{r.getMessage()} {r.args!r}" for r in caplog.records)
    assert caplog.records and SECRET not in logged
    for f in (tmp_path / "c").rglob("*"):
        if f.is_file():
            assert SECRET not in f.read_text(encoding="utf-8")
    assert SECRET not in repr(c) and SECRET not in json.dumps(c.calls)
    # the local backend never receives the key
    lc, lt, _ = client(tmp_path / "l", ['{"a": 1}'], backend="local", send_api_key=False)
    lc.chat(MSGS)
    assert SECRET not in lt.requests[0].headers.get("authorization", "")


# ----------------------------------------------------------------------------- no fallback


def test_missing_key_is_reported_without_a_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    s = default_generation_settings()
    llm = make_llm(s, backend="hosted")
    msg = llm_unreachable_message(llm)
    assert msg is not None and API_KEY_ENV in msg and "hosted" in msg
    reg = s.backends()
    with pytest.raises(BackendUnavailable, match=API_KEY_ENV):
        require_backend(reg.spec("hosted"), "hosted")
    assert llm_unreachable_message(make_llm(s, fake=True, backend="hosted")) is None
    assert make_llm(s, fake=True, backend="hosted").model == "fake-hosted-llm"


def test_eval_generation_never_falls_back(
    script: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the key missing the hosted ablation stops before opening the test split, and a
    backend mismatch is refused (a system's backend is never swapped)."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    ev = script("eval_generation")
    log = Path("collections/cs_ai/results/test_access_log.json")
    before = log.read_bytes()
    rc = ev.main(
        ["--split", "test", "--collection", "cs_ai", "--systems", "G2H", "--backend", "local"]
    )
    assert rc == 2 and "not on --backend local" in capsys.readouterr().out
    rc = ev.main(["--split", "test", "--collection", "cs_ai", "--systems", "G2", "G2H"])
    assert rc == 2 and "on their own" in capsys.readouterr().out
    marker = Path("collections/cs_ai/results/generation/test/_complete.json")
    if marker.exists():  # the standard run is complete here: the key check is what stops it
        rc = ev.main(["--split", "test", "--collection", "cs_ai", "--systems", "G2H", "G4H"])
        out = capsys.readouterr().out
        assert rc == 1 and API_KEY_ENV in out and "never switch" in out
    assert log.read_bytes() == before  # the test split was never opened
