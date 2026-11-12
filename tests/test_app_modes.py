"""Web UI with modes and backends: grounded and hybrid answers (/ask), chat and code (/chat) with
and without documents, the backend list and refusals, evaluated / not evaluated badges, the
summary page, and the static assets (no inline script, no innerHTML, markdown built from text
nodes)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_helpers import write_generation_settings, write_two_backend_llm
from ragsys.agents.engine import AnswerEngine
from ragsys.agents.fake import agent_fake_llm
from ragsys.agents.modes import GENERAL_BANNER
from ragsys.app.server import CollectionEntry, create_multi_app
from ragsys.llm.client import QuotaExceeded
from ragsys.retrieval.pipeline import RetrievalResources
from retrieval_helpers import fake_settings, write_questions, write_settings

SUMMARY = "# Books: pipeline summary\n\n## Generation (test)\n\n| system | accuracy |\n|---|---|\n| G2 | 0.5 |\n"


@pytest.fixture()
def ui(retrieval_dir: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    entries = []
    status = {"hosted": None, "local": None}
    all_llms = {}
    for name, evaluated, code in (("books", True, True), ("law", False, False)):
        d = tmp_path / name
        d.mkdir()
        questions = d / "questions.jsonl"
        write_questions(questions)
        rs = fake_settings(retrieval_dir, d / "results", questions)
        rs.paths.emb_dir = d / "emb"
        _cfg, settings = write_generation_settings(d, write_settings(d / "r.yaml", rs))
        settings.llm_config = write_two_backend_llm(d)
        llms = {"local": agent_fake_llm(), "hosted": agent_fake_llm()}
        llms["hosted"].model = "fake-hosted-llm"
        all_llms[name] = llms

        def factory(rs=rs, settings=settings, llms=llms) -> AnswerEngine:  # type: ignore[no-untyped-def]
            return AnswerEngine(
                settings,
                rs,
                llms["local"],
                RetrievalResources(rs),
                load_threshold_file=False,
                llm_factory=lambda b, llms=llms: llms[b],
            )

        entries.append(
            CollectionEntry(
                name=name,
                settings=settings,
                engine_factory=factory,
                label=name.title(),
                texts={
                    "heading": f"Ask the {name}",
                    "disclaimer": "Disclaimer.",
                    "safety_note": "Not legal advice." if name == "law" else "",
                    "domain": name,
                    "code_assistant": "true" if code else "",
                    "abstention_message": "Not found in these documents",
                },
                examples=["Who may file a complaint?"],
                feedback_path=d / "feedback.jsonl",
                evaluated=evaluated,
                summary_text=(lambda: SUMMARY) if evaluated else None,
                index_note="Only the evaluated documents are searched." if evaluated else "",
                chat_llm_factory=lambda b, llms=llms: llms[b],
            )
        )
    client = TestClient(
        create_multi_app(
            entries,
            "books",
            backends=[
                {"name": "local", "label": "Local: fake", "model": "fake", "privacy": "stays here"},
                {
                    "name": "hosted",
                    "label": "Hosted: fake",
                    "model": "fake-hosted-llm",
                    "privacy": "sent",
                },
            ],
            default_backend="hosted",
            backend_status=lambda b: status[b],
        ),
        base_url="http://127.0.0.1:8000",
    )
    client.app.state.test_llms = all_llms  # type: ignore[attr-defined]
    return client, status, tmp_path


def test_collections_backends_and_meta(ui) -> None:  # type: ignore[no-untyped-def]
    c, status, _tmp = ui
    cols = c.get("/api/collections").json()["collections"]
    assert [(x["id"], x["evaluated"]) for x in cols] == [("books", True), ("law", False)]
    b = c.get("/api/backends").json()
    assert b["default"] == "hosted" and [x["name"] for x in b["backends"]] == ["local", "hosted"]
    assert all(x["available"] for x in b["backends"])
    status["hosted"] = "LLM_API_KEY is not set"
    hosted = next(x for x in c.get("/api/backends").json()["backends"] if x["name"] == "hosted")
    assert not hosted["available"] and "LLM_API_KEY" in hosted["reason"]
    status["hosted"] = None
    books = c.get("/api/meta", params={"collection": "books"}).json()
    law = c.get("/api/meta", params={"collection": "law"}).json()
    assert books["evaluated"] and books["summary_url"] == "/summary?collection=books"
    assert (
        books["code_assistant"]
        and books["index_note"]
        and books["modes"] == ["grounded", "hybrid", "chat"]
    )
    assert (
        not law["evaluated"]
        and law["summary_url"] is None
        and law["safety_note"] == "Not legal advice."
    )


def test_grounded_and_hybrid(ui) -> None:  # type: ignore[no-untyped-def]
    c, _status, _tmp = ui
    q = {
        "question": "Who may file a complaint and how can it be filed?",
        "system": "G4",
        "collection": "books",
    }
    r = c.post("/ask", json=q)
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "grounded" and body["generator"] == {
        "backend": "hosted",
        "model": "fake-hosted-llm",
    }
    assert body["general"] is None and "sentences" in body
    local = c.post("/ask", json={**q, "backend": "local"}).json()
    assert local["generator"]["model"] == "fake-agent-llm"
    # a question the passages cannot answer: the fake synthesizer abstains
    h = c.post("/ask", json={**q, "question": "zzz qqq xyzzy plugh?", "mode": "hybrid"}).json()
    assert h["outcome"] == "abstained" and h["answer"] == ""
    assert h["general"]["banner"] == GENERAL_BANNER and h["general"]["blocks"]
    assert h["general"]["related_reading"] == h["closest_sections"]
    g = c.post("/ask", json={**q, "question": "zzz qqq xyzzy plugh?"}).json()
    assert g["outcome"] == "abstained" and g["general"] is None


def test_chat_code_and_documents(ui) -> None:  # type: ignore[no-untyped-def]
    c, _status, _tmp = ui
    msgs = [{"role": "user", "content": "Write a palindrome check in Python"}]
    r = c.post("/chat", json={"messages": msgs, "collection": "books"})
    assert r.status_code == 200
    chat = r.json()["chat"]
    assert chat["code"] is True and any(b["type"] == "code" for b in chat["blocks"])
    assert r.json()["mode"] == "chat" and r.json()["generator"]["backend"] == "hosted"
    plain = c.post("/chat", json={"messages": msgs, "collection": "law"}).json()
    assert plain["chat"]["code"] is False and plain["safety_note"] == "Not legal advice."
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "Who may file a complaint and how can it be filed?"},
    ]
    docs = c.post(
        "/chat",
        json={"messages": history, "collection": "law", "use_documents": True, "code": False},
    ).json()["chat"]
    assert docs["use_documents"] and docs["passages"] and docs["citations"] == ["C1"]
    assert docs["history_used"] == 3
    bad = c.post(
        "/chat", json={"messages": [{"role": "assistant", "content": "x"}], "collection": "law"}
    )
    assert bad.status_code == 400
    assert c.post("/chat", json={"messages": [], "collection": "law"}).status_code == 422


def test_backend_refusals(ui) -> None:  # type: ignore[no-untyped-def]
    c, status, _tmp = ui
    q = {"question": "Who may file a complaint?", "system": "G4", "collection": "books"}
    assert c.post("/ask", json={**q, "backend": "nope"}).status_code == 400
    assert c.post("/ask", json={**q, "backend": "bad name!"}).status_code == 422
    status["hosted"] = "LLM_API_KEY is not set"
    r = c.post("/ask", json=q)  # the default backend is hosted: no silent switch to local
    assert r.status_code == 503 and "LLM_API_KEY" in r.text
    assert (
        c.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 503
    )
    assert c.post("/ask", json={**q, "backend": "local"}).status_code == 200


def test_summary_page_and_static_assets(ui) -> None:  # type: ignore[no-untyped-def]
    c, _status, _tmp = ui
    s = c.get("/api/summary", params={"collection": "books"}).json()
    assert s["blocks"][0]["type"] == "heading" and any(b["type"] == "table" for b in s["blocks"])
    assert c.get("/api/summary", params={"collection": "law"}).status_code == 404
    page = c.get("/summary?collection=books")
    assert (
        page.status_code == 200 and "script-src 'self'" in page.headers["content-security-policy"]
    )
    assert (
        '<script src="/static/summary.js" defer></script>' in page.text
        and "<script>" not in page.text
    )
    html = c.get("/").text
    for needle in (
        'name="mode"',
        'value="hybrid"',
        "Chat &amp; Code",
        'label for="backend"',
        'id="new-chat"',
        'id="use-docs"',
        'id="code-mode"',
        'id="collection-badge"',
        "/static/markdown.js",
    ):
        assert needle in html, needle
    for asset in ("/static/app.js", "/static/markdown.js", "/static/summary.js"):
        js = c.get(asset).text
        assert ".innerHTML" not in js and "insertAdjacentHTML" not in js and "eval(" not in js, (
            asset
        )
    md = c.get("/static/markdown.js").text
    assert "textContent" in md and "createTextNode" in md and "href" not in md
    css = c.get("/static/app.css").text
    assert ".md-pre" in css and "var(--mono)" in css and "@media (max-width: 420px)" in css


def test_feedback_records_mode_and_model(ui) -> None:  # type: ignore[no-untyped-def]
    c, _status, tmp = ui
    fb = {
        "question": "q?",
        "system": "G4",
        "rating": "up",
        "collection": "books",
        "mode": "hybrid",
        "model": "m",
    }
    assert c.post("/feedback", json=fb).json() == {"ok": True}
    row = json.loads((tmp / "books" / "feedback.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["mode"] == "hybrid" and row["model"] == "m"


def test_generator_failure_is_a_503_and_nothing_falls_back(ui) -> None:  # type: ignore[no-untyped-def]
    client, _status, _ = ui
    hosted = client.app.state.test_llms["books"]["hosted"]

    def quota(*_a: object, **_k: object) -> None:
        raise QuotaExceeded("the provider asks to wait 3600 s")

    hosted.chat = quota  # the hosted model refuses every call
    hosted.chat_json = quota
    msg = {
        "messages": [{"role": "user", "content": "Explain a hash table."}],
        "collection": "books",
    }
    r = client.post("/chat", json={**msg, "backend": "hosted"})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "hosted generator failed" in detail and "quota" in detail and "No other model" in detail
    r = client.post("/ask", json={"question": "Who may file a complaint?", "collection": "books"})
    assert r.status_code == 503 and "hosted" in r.json()["detail"]  # default backend: hosted
    r = client.post("/chat", json={**msg, "backend": "local"})  # chosen explicitly: answers
    assert r.status_code == 200 and r.json()["generator"]["backend"] == "local"
