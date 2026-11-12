"""Verify (guards + critic per sentence, on demand) and the strict Grounded-only mode."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from agent_helpers import TEXTS
from assistant_helpers import build_assistant, by_name, chat, make_client
from ragsys.assistant.verify import prose_of
from ragsys.llm.client import Message
from ragsys.llm.fake import FakeLLM

C1 = "structural:testact:00001"
C3 = "structural:testact:00003"


def _critic(unsupported: list[str], verdict: str = "revise") -> FakeLLM:
    def reply(messages: Sequence[Message]) -> str:
        return json.dumps(
            {"verdict": verdict, "unsupported_sentences": unsupported, "feedback": ""}
        )

    return FakeLLM(reply, model="fake-critic")


def test_verify_marks_sentences(tmp_path: Path) -> None:
    critic = _critic(["Appeals go to the Supreme Court."])
    assistant, _llm = build_assistant(tmp_path, json_llm=lambda _b, _c: critic)
    c = make_client(assistant)
    answer = (
        "A complaint is decided within three months [C1]. "
        "An appeal lies within forty-five days [C2]. "
        "It costs 999 rupees [C1]. "
        "Appeals go to the Supreme Court [C2]. "
        "This is a summary.\n\n```python\nprint(42)\n```"
    )
    body = {
        "question": "How fast is a complaint decided?",
        "answer": answer,
        "passages": [{"label": "C1", "chunk_id": C1}, {"label": "C2", "chunk_id": C3}],
        "collection": "books",
    }
    r = c.post("/api/verify", json=body)
    assert r.status_code == 200, r.text
    v = r.json()
    status = {s["text"]: (s["status"], s["reasons"]) for s in v["sentences"]}
    assert status["A complaint is decided within three months."] == ("supported", [])
    assert status["An appeal lies within forty-five days."][0] == "supported"
    assert status["It costs 999 rupees."][0] == "unsupported"
    assert "999" in status["It costs 999 rupees."][1][0]
    assert status["Appeals go to the Supreme Court."][0] == "unsupported"
    assert "critic" in status["Appeals go to the Supreme Court."][1][0]
    assert status["This is a summary."] == ("unsupported", ["no citation"])
    assert not any("print" in s for s in status)  # code is not a claim
    assert v["supported"] == 2 and v["total"] == 5 and v["verdict"] == "partly_supported"
    assert v["guards"]["passed"] is False and v["critic"]["verdict"] == "revise"
    assert len(critic.calls) == 1  # one critic call


def test_verify_refusals(tmp_path: Path) -> None:
    assistant, _llm = build_assistant(tmp_path)
    c = make_client(assistant)
    base = {"question": "q", "answer": "A [C1].", "collection": "books"}
    r = c.post("/api/verify", json={**base, "passages": []})
    assert r.status_code == 400 and "nothing to verify" in r.json()["detail"]
    r = c.post(
        "/api/verify",
        json={**base, "passages": [{"label": "C1", "chunk_id": "structural:up_0123456789:00000"}]},
    )
    assert r.status_code == 409 and "no longer available" in r.json()["detail"]
    r = c.post("/api/verify", json={**base, "passages": [{"label": "X1", "chunk_id": C1}]})
    assert r.status_code == 422


def test_grounded_only_answers_or_abstains(tmp_path: Path) -> None:
    assistant, llm = build_assistant(tmp_path)
    c = make_client(assistant)
    events = chat(
        c,
        "Within how many days may an appeal be filed before the State Commission?",
        grounded_only=True,
    )
    assert [n for n, _ in events] == ["meta", "status", "done"]
    done = by_name(events, "done")[0]
    assert done["mode"] == "grounded_only" and done["outcome"] == "answered"
    assert done["citations"] == ["C1"] and done["passages"][0]["chunk_id"] == C3
    assert done["source_line"] == "Answered with documents (1 source)"
    assert all(s["status"] == "supported" for s in done["sentences"])
    assert done["cost"]["critic_calls"] == 1 and llm.calls == []  # no streamed chat call
    events = chat(c, "Write a haiku about autumn leaves", grounded_only=True)
    done = by_name(events, "done")[0]
    assert done["outcome"] == "abstained" and done["answer"] == ""
    assert done["source_line"] == "Not found in these documents" and done["closest"]
    events = chat(c, "anything", grounded_only=True, collection=None)
    assert by_name(events, "error")[0]["code"] == "no_documents"


def test_prose_of_drops_code_math_and_markup() -> None:
    text = "## Heading\n**Bold** claim [C1].\n\n$$x^2$$\n\n```js\nlet a = 1;\n```\nEnd [C2]."
    out = prose_of(text)
    assert "let a" not in out and "x^2" not in out and "Heading" not in out and "**" not in out
    assert "Bold claim [C1]." in out and "End [C2]." in out


def test_texts_fixture_unchanged() -> None:
    assert "forty-five days" in TEXTS[2]
