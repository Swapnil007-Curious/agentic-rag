"""FastAPI endpoints (ask, passage lookup, feedback, meta) and the served single page."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_helpers import ScriptedLLM, critic_reply, synth_reply, write_generation_settings
from ragsys.agents.engine import AnswerEngine
from ragsys.app.server import DISCLAIMER, create_app
from ragsys.retrieval.pipeline import RetrievalResources
from retrieval_helpers import fake_settings, write_questions, write_settings

SOURCES = (
    "file_name,doc_id,official_title,source_url,sha256,bytes,pages,downloaded_on\n"
    "testact.pdf,testact,The Test Protection Act,https://example.invalid/a.pdf,"
    "0123456789abcdef0123,100,3,\n"
)


@pytest.fixture()
def client(retrieval_dir: Path, tmp_path: Path) -> tuple[TestClient, ScriptedLLM, Path]:
    questions = tmp_path / "questions.jsonl"
    write_questions(questions)
    rs = fake_settings(retrieval_dir, tmp_path / "results", questions)
    rs.paths.emb_dir = tmp_path / "emb"
    _cfg, settings = write_generation_settings(tmp_path, write_settings(tmp_path / "r.yaml", rs))
    (tmp_path / "SOURCES.csv").write_text(SOURCES, encoding="utf-8")
    llm = ScriptedLLM(
        synth=[synth_reply("A complaint may be filed in writing [C1].", ["C1"])],
        critic=[critic_reply()],
    )
    engine = AnswerEngine(settings, rs, llm, RetrievalResources(rs), load_threshold_file=False)
    app = create_app(settings, engine, health=lambda: None)
    return TestClient(app, base_url="http://127.0.0.1:8000"), llm, tmp_path


def test_page_and_static_assets(client: tuple[TestClient, ScriptedLLM, Path]) -> None:
    c, _llm, _tmp = client
    r = c.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    html = r.text
    assert DISCLAIMER in html and 'label for="question"' in html and 'label for="system"' in html
    assert 'label for="plain"' in html and 'role="switch"' in html and 'aria-live="polite"' in html
    assert '<script src="/static/app.js" defer></script>' in html and "<script>" not in html
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    css = c.get("/static/app.css")
    assert (
        css.status_code == 200
        and ":focus-visible" in css.text
        and "prefers-color-scheme: dark" in css.text
    )
    js = c.get("/static/app.js")
    assert js.status_code == 200 and ".innerHTML" not in js.text and "textContent" in js.text
    assert c.get("/static/../server.py").status_code == 404
    evil = TestClient(c.app, base_url="http://evil.example")
    assert evil.get("/").status_code == 400  # loopback host names only


def test_meta(client: tuple[TestClient, ScriptedLLM, Path]) -> None:
    c, _llm, _tmp = client
    meta = c.get("/api/meta").json()
    assert [s["id"] for s in meta["systems"]] == ["G2", "G4", "G5", "G6"]
    assert meta["default_system"] == "G5" and meta["threshold_tuned"] is False
    assert meta["disclaimer"] == DISCLAIMER
    doc = meta["documents"][0]
    assert doc["title"] == "The Test Protection Act" and doc["version"] == "sha256 0123456789ab"


def test_ask_and_passage_lookup(client: tuple[TestClient, ScriptedLLM, Path]) -> None:
    c, _llm, _tmp = client
    r = c.post("/ask", json={"question": "Who may file a complaint?", "system": "G4"})
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "answered" and body["system"] == "G4"
    assert body["sentences"] == [
        {"text": "A complaint may be filed in writing.", "citations": ["C1"]}
    ]
    c1 = body["passages"]["C1"]
    assert c1["text"] and c1["heading_path"] and c1["section_ids"] and c1["pages"]
    assert [s["name"] for s in body["steps"]] == [
        "retrieval",
        "context",
        "synthesis",
        "guards",
        "critic",
    ]
    assert body["cost"]["llm_calls"] == 2 and len(body["closest_sections"]) <= 3
    p = c.get("/passage", params={"chunk_id": c1["chunk_id"]})
    assert p.status_code == 200 and p.json()["text"] == c1["text"]
    assert c.get("/passage", params={"chunk_id": "structural:nope:00000"}).status_code == 404
    assert c.get("/passage", params={"chunk_id": "../../etc:passwd"}).status_code == 404
    # G5 without a tuned threshold still answers, with a warning
    g5 = c.post("/ask", json={"question": "Who may file a complaint?"}).json()
    assert g5["system"] == "G5" and any("threshold" in w for w in g5["warnings"])
    assert c.post("/ask", json={"question": "Who?", "system": "G1"}).status_code == 400  # not in UI
    assert c.post("/ask", json={"question": "  ", "system": "G4"}).status_code == 422
    assert c.post("/ask", json={"question": "abc def", "system": "G4", "x": 1}).status_code == 422


def test_abstained_answer_lists_closest_sections(
    client: tuple[TestClient, ScriptedLLM, Path],
) -> None:
    c, llm, _tmp = client
    llm.queues["synthesizer"] = [synth_reply(abstain=True)]
    llm.used["synthesizer"] = 0
    body = c.post(
        "/ask", json={"question": "What is the GST rate on vegetables?", "system": "G2"}
    ).json()
    assert (
        body["outcome"] == "abstained"
        and body["reason"] == "synth_abstain"
        and body["answer"] == ""
    )
    assert 1 <= len(body["closest_sections"]) <= 3
    assert all(
        {"chunk_id", "heading_path", "section_ids", "pages"} <= set(s)
        for s in body["closest_sections"]
    )


def test_feedback_is_stored_locally(client: tuple[TestClient, ScriptedLLM, Path]) -> None:
    c, _llm, tmp = client
    ok = c.post(
        "/feedback",
        json={
            "question": "q?",
            "system": "G4",
            "outcome": "answered",
            "answer": "a",
            "rating": "up",
        },
    )
    assert ok.status_code == 200 and ok.json() == {"ok": True}
    assert (
        c.post("/feedback", json={"question": "q?", "system": "G4", "rating": "meh"}).status_code
        == 422
    )
    rows = [
        json.loads(x) for x in (tmp / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1 and rows[0]["rating"] == "up" and rows[0]["utc_time"]


def test_unreachable_llm_gives_503(retrieval_dir: Path, tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    write_questions(questions)
    rs = fake_settings(retrieval_dir, tmp_path / "results", questions)
    _cfg, settings = write_generation_settings(tmp_path, write_settings(tmp_path / "r.yaml", rs))
    engine = AnswerEngine(
        settings, rs, ScriptedLLM(), RetrievalResources(rs), load_threshold_file=False
    )
    app = create_app(settings, engine, health=lambda: "endpoint down")
    r = TestClient(app, base_url="http://localhost").post(
        "/ask", json={"question": "Who may file?", "system": "G4"}
    )
    assert r.status_code == 503 and "endpoint down" in r.text
