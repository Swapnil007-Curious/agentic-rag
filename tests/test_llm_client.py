"""LLM client: caching, retries/backoff, JSON handling, key never logged (fake transport)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest

from ragsys.llm.client import LLMClient, LLMConfig, LLMJSONError, QuotaExceeded, extract_json
from ragsys.llm.fake import FakeLLM

SECRET = "sk-test-SECRET-do-not-log-12345"


def completion(content: str) -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model",
        "choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


class Transport:
    """Scripted responses; records every request."""

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        item = self.script[min(len(self.requests) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, int):
            return httpx2.Response(item, json={"error": {"message": "slow down"}})
        return httpx2.Response(200, json=completion(item))


def make_client(
    tmp_path: Path, script: list[Any], **cfg: Any
) -> tuple[LLMClient, Transport, list[float]]:
    transport = Transport(script)
    sleeps: list[float] = []
    client = LLMClient(
        LLMConfig(base_url="http://llm.test/v1", model="m", cache_dir=tmp_path / "cache", **cfg),
        http_client=httpx2.Client(transport=httpx2.MockTransport(transport)),
        sleep=sleeps.append,
    )
    return client, transport, sleeps


MSGS = [{"role": "user", "content": "Say hi as JSON"}]


def test_cache_makes_reruns_free(tmp_path: Path) -> None:
    client, transport, _ = make_client(tmp_path, ['{"a": 1}'])
    r1 = client.chat(MSGS)
    assert (r1.content, r1.cached, r1.usage["total_tokens"]) == ('{"a": 1}', False, 10)
    r2 = client.chat(MSGS)
    assert r2.cached and r2.content == r1.content and len(transport.requests) == 1
    client.chat([{"role": "user", "content": "different"}])
    assert len(transport.requests) == 2
    # a new client (new process) reuses the cache on disk
    client2, transport2, _ = make_client(tmp_path, ["unused"])
    assert client2.chat(MSGS).cached and transport2.requests == []
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 2
    assert [c["cached"] for c in client.calls] == [False, True, False]


def test_request_parameters_and_json_mode(tmp_path: Path) -> None:
    client, transport, _ = make_client(tmp_path, ['Sure! ```json\n{"q": "x"}\n```'], seed=3)
    data, resp = client.chat_json(MSGS)
    assert data == {"q": "x"} and resp.attempts == 1
    body = json.loads(transport.requests[0].content)
    assert body["response_format"] == {"type": "json_object"} and body["seed"] == 3
    assert body["model"] == "m" and body["temperature"] == 0.2 and body["max_tokens"] == 512
    client_nojson, transport2, _ = make_client(tmp_path / "b", ["{}"], json_mode=False, seed=None)
    client_nojson.chat_json(MSGS)
    body2 = json.loads(transport2.requests[0].content)
    assert "response_format" not in body2 and "seed" not in body2


def test_retries_with_exponential_backoff(tmp_path: Path) -> None:
    client, transport, sleeps = make_client(
        tmp_path, [429, 503, httpx2.ReadTimeout("timed out"), '{"ok": true}']
    )
    resp = client.chat(MSGS)
    assert resp.attempts == 4 and len(transport.requests) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_retries_give_up_after_max(tmp_path: Path) -> None:
    client, transport, sleeps = make_client(tmp_path, [429], backoff_max_s=5.0)
    # a rate limit that outlasts every retry is reported as an exhausted quota (exit code 75
    # in the evaluation scripts, so a waiting runner can try again later)
    with pytest.raises(QuotaExceeded) as info:
        client.chat(MSGS)
    assert isinstance(info.value.__cause__, openai.RateLimitError)
    assert len(transport.requests) == 6  # first attempt + 5 retries
    assert sleeps == [1.0, 2.0, 4.0, 5.0, 5.0]
    client2, transport2, _ = make_client(tmp_path / "b", [400])
    with pytest.raises(openai.BadRequestError):
        client2.chat(MSGS)
    assert len(transport2.requests) == 1  # client errors are not retried


def test_api_key_is_sent_but_never_logged_or_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LLM_API_KEY", SECRET)
    caplog.set_level(logging.DEBUG)
    client, transport, _ = make_client(tmp_path, [429, '{"x": 1}'])
    client.chat(MSGS)
    client.chat(MSGS)  # cached
    assert transport.requests[0].headers["authorization"] == f"Bearer {SECRET}"
    assert SECRET not in caplog.text
    assert SECRET not in repr(client) and SECRET not in json.dumps(client.calls)
    assert SECRET not in json.dumps({k: str(v) for k, v in vars(client).items() if k != "_sdk"})
    for f in (tmp_path / "cache").rglob("*"):
        if f.is_file():
            assert SECRET not in f.read_text(encoding="utf-8")


def test_extract_json_variants() -> None:
    assert extract_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}
    assert extract_json('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('noise {"q": "uses {braces} and \\"quotes\\""} tail') == {
        "q": 'uses {braces} and "quotes"'
    }
    with pytest.raises(LLMJSONError):
        extract_json("no json here")
    with pytest.raises(LLMJSONError):
        extract_json("[1, 2]")


def test_fake_llm_is_deterministic() -> None:
    fake = FakeLLM(["first", '{"k": 2}'])
    assert fake.chat(MSGS).content == "first"
    assert fake.chat_json(MSGS)[0] == {"k": 2}
    assert len(fake.calls) == 2
    fn = FakeLLM(lambda m: json.dumps({"echo": m[-1]["content"]}))
    assert fn.chat_json(MSGS)[0] == {"echo": "Say hi as JSON"}
    d1 = FakeLLM().chat_json([{"role": "user", "content": '"type": "numeric" PROVISION [a:s1]'}])
    d2 = FakeLLM().chat_json([{"role": "user", "content": '"type": "numeric" PROVISION [a:s1]'}])
    assert d1[0] == d2[0] and d1[0]["type"] == "numeric"
