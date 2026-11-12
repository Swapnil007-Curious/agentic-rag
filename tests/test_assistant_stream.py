"""Chat assistant streaming endpoint: tokens in order, the grounding gate above and below its
threshold, Stop, friendly errors, request limits, the pages and public mode. No network: a fake
streaming LLM and the hashing encoder."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import httpx2
import pytest

from assistant_helpers import (
    assistant_settings,
    build_assistant,
    by_name,
    chat,
    make_client,
    sse_events,
)
from ragsys.assistant.citations import GENERAL_LINE
from ragsys.assistant.history import Turn
from ragsys.assistant.llm import (
    BUSY_MESSAGE,
    OpenAIStreamingChat,
    StreamFailure,
    split_tokens,
)
from ragsys.assistant.prompts import CITE_REMINDER, GROUNDING_RULES
from ragsys.assistant.service import ChatRequestData
from ragsys.llm.backends import BackendSpec

GROUNDED_Q = "Every complaint shall be decided within a period of three months where no testing of goods is required?"
GENERAL_Q = "write a poem about the sea and the stars"


def test_streams_tokens_in_order_and_ends_with_done(tmp_path: Path) -> None:
    reply = "First part. **Second** part with `code` and $x^2$."
    assistant, llm = build_assistant(tmp_path, responder=[reply])
    events = chat(make_client(assistant), GENERAL_Q)
    names = [n for n, _ in events]
    assert names[0] == "meta" and names[1] == "context" and names[-1] == "done"
    assert set(names[2:-1]) == {"token"}
    tokens = [d["t"] for d in by_name(events, "token")]
    assert tokens == split_tokens(reply) and "".join(tokens) == reply
    done = by_name(events, "done")[0]
    assert done["answer"] == reply and done["stopped"] is False
    assert done["model"] == "fake-hosted" and done["backend"] == "hosted"
    assert done["timings"]["ttft_ms"] is not None and done["timings"]["total_ms"] >= 0
    assert len(llm.calls) == 1  # one LLM call per message


def test_below_gate_is_plain_chat(tmp_path: Path) -> None:
    assistant, llm = build_assistant(tmp_path, responder=["A poem [C1] without sources."])
    events = chat(make_client(assistant), GENERAL_Q)
    ctx = by_name(events, "context")[0]
    assert ctx["grounded"] is False and ctx["passages"] == []
    assert ctx["checks"][0]["passed"] is False and ctx["checks"][0]["top_score"] < 0.5
    done = by_name(events, "done")[0]
    # no passages were given: every label is invalid and stripped
    assert done["answer"] == "A poem without sources." and done["removed_labels"] == ["C1"]
    assert done["grounded"] is False and done["source_line"] == GENERAL_LINE
    system, user = llm.calls[0]["messages"][0], llm.calls[0]["messages"][-1]
    assert GROUNDING_RULES not in system["content"] and "Not legal advice." in system["content"]
    assert user["content"] == GENERAL_Q and "<<<PASSAGE" not in user["content"]


def test_above_gate_attaches_passages_and_cites_them(tmp_path: Path) -> None:
    reply = "Three months [C1]. Five months with testing [C2]. Invented [C9]."
    assistant, llm = build_assistant(tmp_path, responder=[reply])
    events = chat(make_client(assistant), GROUNDED_Q)
    ctx = by_name(events, "context")[0]
    assert ctx["grounded"] is True and ctx["checks"][0]["top_score"] >= 0.5
    assert [p["label"] for p in ctx["passages"]][:2] == ["C1", "C2"]
    assert ctx["passages"][0]["chunk_id"] == "structural:testact:00001"
    assert "three months" in ctx["passages"][0]["text"]
    done = by_name(events, "done")[0]
    assert done["citations"] == ["C1", "C2"] and done["removed_labels"] == ["C9"]
    assert "[C9]" not in done["answer"] and done["answer"].endswith("Invented.")
    assert done["source_line"] == "Answered with documents (2 sources)"
    assert [p["label"] for p in done["passages"]] == ["C1", "C2"]
    msgs = llm.calls[0]["messages"]
    assert GROUNDING_RULES in msgs[0]["content"]
    assert (
        "<<<PASSAGE C1>>>" in msgs[-1]["content"]
        and f"My message: {GROUNDED_Q}" in msgs[-1]["content"]
    )
    assert msgs[-1]["content"].endswith(CITE_REMINDER)
    # passage budget: the hosted backend packs at most 2048 index tokens (4 passages of 60 here)
    assert len(ctx["passages"]) <= 4


def test_stop_ends_the_reply_after_the_current_token(tmp_path: Path) -> None:
    holder: dict[str, Any] = {}

    def on_token(n: int) -> None:
        if n == 3:  # a Stop request arrives while the third token is produced
            for sid in holder["assistant"].active_streams():
                assert holder["client"].post("/api/chat/stop", json={"stream_id": sid}).json() == {
                    "stopped": True
                }

    reply = "one two three four five six seven"
    assistant, llm = build_assistant(tmp_path, responder=[reply], on_token=on_token)
    holder["assistant"] = assistant
    holder["client"] = make_client(assistant)
    events = chat(holder["client"], GENERAL_Q)
    tokens = [d["t"] for d in by_name(events, "token")]
    assert tokens == ["one ", "two ", "three "]
    done = by_name(events, "done")[0]
    assert done["stopped"] is True and done["answer"] == "one two three "
    assert llm.streams[0].closed and llm.streams[0].pulled == 3  # upstream closed, not drained
    assert assistant.active_streams() == []
    assert holder["client"].post("/api/chat/stop", json={"stream_id": "x" * 16}).json() == {
        "stopped": False
    }


def test_closed_connection_stops_and_closes_the_upstream_stream(tmp_path: Path) -> None:
    assistant, llm = build_assistant(tmp_path, responder=["a b c d e f g h"])
    stop = threading.Event()
    gen = assistant.chat_events(ChatRequestData([Turn("user", GENERAL_Q)], "books"), stop)
    names = []
    for name, _data in ((e["event"], e["data"]) for e in gen):
        names.append(name)
        if names.count("token") == 2:
            break
    gen.close()  # the server drops the generator when the client goes away
    assert llm.streams[0].closed and llm.streams[0].pulled == 2
    assert assistant.active_streams() == []
    # the route's on-close hook sets the stop event: a running reply ends at the next token
    stop2 = threading.Event()
    stop2.set()
    events = list(assistant.chat_events(ChatRequestData([Turn("user", GENERAL_Q)], "books"), stop2))
    assert [e["event"] for e in events] == ["meta", "context"]


def test_rate_limited_provider_gives_busy_message(tmp_path: Path) -> None:
    assistant, _llm = build_assistant(tmp_path, fail=StreamFailure("busy", BUSY_MESSAGE, 30.0))
    events = chat(make_client(assistant), GENERAL_Q)
    err = by_name(events, "error")[0]
    assert err["code"] == "busy" and "try again in a minute" in err["message"]
    assert not by_name(events, "done")


def test_request_validation(tmp_path: Path) -> None:
    assistant, _llm = build_assistant(tmp_path)
    c = make_client(assistant)
    long = {"messages": [{"role": "user", "content": "x" * 4001}]}
    r = c.post("/api/chat", json=long)
    assert r.status_code == 413 and "4000 characters" in r.json()["detail"]
    r = c.post("/api/chat", json={"messages": [{"role": "assistant", "content": "hi"}]})
    assert r.status_code == 400
    r = c.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "collection": "nope"}
    )
    assert r.status_code == 404
    r = c.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "backend": "local"}
    )
    assert r.status_code == 400
    r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "extra": 1})
    assert r.status_code == 422


def test_per_client_rate_limit_and_daily_cap(tmp_path: Path) -> None:
    settings = assistant_settings(
        rate_limits={"per_ip": [{"max": 2, "window_s": 600}], "global_daily_requests": 3}
    )
    assistant, _llm = build_assistant(tmp_path, settings=settings)
    c = make_client(assistant)
    body = {"messages": [{"role": "user", "content": "hello"}]}
    assert c.post("/api/chat", json=body).status_code == 200
    assert c.post("/api/chat", json=body).status_code == 200
    r = c.post("/api/chat", json=body)
    assert r.status_code == 429 and "try again in about 10 minutes" in r.json()["detail"]
    assert int(r.headers["retry-after"]) > 500
    # another client address: allowed until the global daily cap (3) is reached
    assistant.guard.messages.limits[0] = assistant.guard.messages.limits[0].model_copy(
        update={"max": 100}
    )
    assert c.post("/api/chat", json=body).status_code == 200
    r = c.post("/api/chat", json=body)
    assert r.status_code == 503 and "daily limit" in r.json()["detail"]


def test_conversation_summary_is_returned_and_reused(tmp_path: Path) -> None:
    def responder(messages: Any) -> str:
        if "summarise" in messages[0]["content"]:
            return "The user asked about many things."
        return "Reply."

    assistant, llm = build_assistant(tmp_path, responder=responder, history_tokens=300)
    c = make_client(assistant)
    history = []
    for i in range(6):
        history += [
            {"role": "user", "content": f"question {i} " + "word " * 60},
            {"role": "assistant", "content": f"answer {i} " + "text " * 60},
        ]
    history.append({"role": "user", "content": "and the last one?"})
    events = sse_events(c.post("/api/chat", json={"messages": history}).text)
    done = by_name(events, "done")[0]
    assert done["summary"]["text"] == "The user asked about many things."
    assert done["history"]["summarised"] is True and len(llm.calls) == 2  # summary + reply
    sent = llm.calls[1]["messages"]
    assert sent[1]["role"] == "user" and "The user asked about many things." in sent[1]["content"]
    # next turn: the stored summary still covers the same messages -> no new summary call
    history += [{"role": "assistant", "content": "Reply."}, {"role": "user", "content": "thanks"}]
    events = sse_events(
        c.post("/api/chat", json={"messages": history, "summary": done["summary"]}).text
    )
    assert by_name(events, "done")[0]["summary"] is None and len(llm.calls) == 3


def test_pages_csp_and_public_mode(tmp_path: Path) -> None:
    assistant, _llm = build_assistant(tmp_path)
    c = make_client(assistant)
    r = c.get("/")
    html = r.text
    assert r.status_code == 200 and 'id="composer-input"' in html and 'id="conv-list"' in html
    assert (
        "<script>" not in html
        and "style=" not in html
        and "http" not in html.split("<body>")[0].replace("http-equiv", "")
    )
    for asset in ("/static/vendor/katex/katex.min.js", "/static/vendor/highlight/highlight.min.js"):
        assert f'<script src="{asset}" defer></script>' in html
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert c.get("/research").status_code == 200 and 'id="ask-form"' in c.get("/research").text
    cols = c.get("/api/collections").json()["collections"]
    assert cols[0]["notice"] == "Not legal advice." and cols[0]["examples"]
    assert cols[0]["sources"][0]["licence"] == "CC BY 4.0"
    health = c.get("/api/health").json()
    assert health["public"] is False and health["limits"]["max_message_chars"] == 4000
    font = c.get("/static/vendor/katex/fonts/KaTeX_Main-Regular.woff2")
    assert font.status_code == 200 and font.headers["content-type"] == "font/woff2"

    pub = make_client(assistant, public=True, frame_ancestors="https://huggingface.co")
    assert pub.get("/").status_code == 200
    assert (
        "frame-ancestors https://huggingface.co" in pub.get("/").headers["content-security-policy"]
    )
    for path in ("/research", "/api/meta", "/summary"):
        assert pub.get(path).status_code == 404, path
    assert pub.post("/ask", json={"question": "Who may file?"}).status_code == 404
    assert pub.post("/feedback", json={}).status_code == 404


class _SSE:
    """A fake OpenAI-compatible endpoint answering with a streamed completion or an error."""

    def __init__(self, status: int = 200, retry_after: str | None = None, used: int = 8) -> None:
        self.status = status
        self.retry_after = retry_after
        self.used = used
        self.bodies: list[bytes] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(request.content)
        if self.status != 200:
            headers = {"retry-after": self.retry_after} if self.retry_after else {}
            return httpx2.Response(self.status, json={"error": {"message": "x"}}, headers=headers)
        chunks = []
        for i, piece in enumerate(["Hel", "lo", " world"]):
            chunks.append(
                '{"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
                f'"choices":[{{"index":0,"delta":{{"content":"{piece}"}},"finish_reason":null}}]}}'
            )
            del i
        chunks.append(
            '{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],'
            f'"usage":{{"prompt_tokens":5,"completion_tokens":3,"total_tokens":{self.used}}}}}'
        )
        body = "".join(f"data: {c}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx2.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )


def _client(transport: _SSE, monkeypatch: pytest.MonkeyPatch) -> OpenAIStreamingChat:
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    spec = BackendSpec(
        base_url="https://api.example.invalid/v1",
        model="m",
        requires_key=True,
        extra_body={"reasoning_effort": "low"},
        requests_per_minute=30,
        tokens_per_minute=8000,
    )
    return OpenAIStreamingChat(
        spec, "hosted", http_client=httpx2.Client(transport=httpx2.MockTransport(transport))
    )


def test_openai_streaming_client_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _SSE()
    llm = _client(transport, monkeypatch)
    stream = llm.stream([{"role": "user", "content": "hi"}], max_tokens=50, temperature=0.3)
    assert list(stream) == ["Hel", "lo", " world"]
    assert stream.usage["total_tokens"] == 8
    sent = transport.bodies[0].decode()
    assert '"stream":true' in sent.replace(" ", "") and '"reasoning_effort":"low"' in sent.replace(
        " ", ""
    )
    assert "test-key-not-real" not in repr(llm)


@pytest.mark.parametrize(
    ("status", "retry_after", "code"),
    [(429, "20", "busy"), (429, "7200", "quota"), (503, None, "unavailable"), (400, None, "error")],
)
def test_openai_streaming_client_errors(
    monkeypatch: pytest.MonkeyPatch, status: int, retry_after: str | None, code: str
) -> None:
    llm = _client(_SSE(status, retry_after), monkeypatch)
    with pytest.raises(StreamFailure) as info:
        llm.stream([{"role": "user", "content": "hi"}], max_tokens=50, temperature=0.3)
    assert info.value.code == code
    if code == "busy":
        assert info.value.message == BUSY_MESSAGE


def test_client_side_budget_refuses_instead_of_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _client(_SSE(used=8000), monkeypatch)  # the first call uses the whole minute
    big = [{"role": "user", "content": "x" * 30000}]
    list(llm.stream(big, max_tokens=500, temperature=0.3))
    with pytest.raises(StreamFailure) as info:
        llm.stream(big, max_tokens=500, temperature=0.3)
    assert info.value.code == "busy"


def test_missing_key_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    llm = OpenAIStreamingChat(BackendSpec(model="m", requires_key=True), "hosted")
    with pytest.raises(StreamFailure) as info:
        llm.stream([{"role": "user", "content": "hi"}], max_tokens=5, temperature=0.0)
    assert info.value.code == "unavailable" and not llm.is_reachable()


class _Seq(_SSE):
    """429 responses (with a message and headers) before a normal streamed reply."""

    def __init__(self, failures: list[tuple[str, dict[str, str]]]) -> None:
        super().__init__()
        self.failures = list(failures)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if self.failures:
            message, headers = self.failures.pop(0)
            self.bodies.append(request.content)
            return httpx2.Response(429, json={"error": {"message": message}}, headers=headers)
        return super().__call__(request)


def test_per_minute_limit_with_short_wait_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    tpm = "Rate limit reached on tokens per minute (TPM): Limit 8000. Please try again in 0.2s."
    transport = _Seq([(tpm, {"x-ratelimit-reset-requests": "37m26.4s"})])
    llm = _client(transport, monkeypatch)
    stream = llm.stream([{"role": "user", "content": "hi"}], max_tokens=50, temperature=0.3)
    assert "".join(stream) == "Hello world" and len(transport.bodies) == 2
    assert stream.queued_ms >= 200  # the wait is reported as queueing, not as model latency


@pytest.mark.parametrize(
    ("message", "headers", "code"),
    [
        # the daily request bucket's reset is not a wait for this request: busy, not quota
        (
            "Rate limit reached on tokens per minute (TPM).",
            {"x-ratelimit-reset-requests": "37m26.4s"},
            "busy",
        ),
        ("Please try again in 45s.", {}, "busy"),
        ("Rate limit reached on tokens per day (TPD): Limit 200000.", {}, "quota"),
        ("Rate limit reached on requests per day (RPD).", {"retry-after": "3"}, "quota"),
    ],
)
def test_rate_limit_classification(
    monkeypatch: pytest.MonkeyPatch, message: str, headers: dict[str, str], code: str
) -> None:
    llm = _client(_Seq([(message, headers)] * 3), monkeypatch)
    with pytest.raises(StreamFailure) as info:
        llm.stream([{"role": "user", "content": "hi"}], max_tokens=50, temperature=0.3)
    assert info.value.code == code
