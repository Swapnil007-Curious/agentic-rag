"""Web UI with two collections: the selector endpoint, per-collection metadata (texts,
disclaimer, documents with licence and source links, dev-split example questions), questions,
passages and feedback routed to the selected collection, lazily built engines, and refusals
for unknown or unbuilt collections."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_helpers import ScriptedLLM, critic_reply, synth_reply, write_generation_settings
from ragsys.agents.engine import AnswerEngine
from ragsys.app.server import CollectionEntry, create_multi_app, example_questions, read_sources
from ragsys.eval.split import split_of
from ragsys.retrieval.pipeline import RetrievalResources
from retrieval_helpers import fake_settings, write_questions, write_settings

TEXTBOOK_SOURCES = (
    "file_name,doc_id,title,authors,edition,licence,licence_url,licence_source,source_url,"
    "sha256,bytes,pages,pages_kept,chapters,downloaded_on\n"
    "tb.pdf,tb,The Testing Book,Ann Author,First,CC BY-SA 4.0 International,"
    'https://creativecommons.org/licenses/by-sa/4.0/,"PDF p.2: ""licensed""",'
    "https://example.invalid/tb,0123456789abcdef0123,10,9,4,Basics (pp. 4-6),2026-11-11\n"
)


@pytest.fixture()
def app(retrieval_dir: Path, tmp_path: Path) -> tuple[TestClient, dict[str, int], Path]:
    built: dict[str, int] = {"law": 0, "books": 0}
    entries = []
    for name, disclaimer in (("law", "Not legal advice."), ("books", "Open textbooks only.")):
        d = tmp_path / name
        d.mkdir()
        questions = d / "questions.jsonl"
        write_questions(questions)
        rs = fake_settings(retrieval_dir, d / "results", questions)
        rs.paths.emb_dir = d / "emb"
        _cfg, settings = write_generation_settings(d, write_settings(d / "r.yaml", rs))
        llm = ScriptedLLM(
            synth=[synth_reply(f"Answer from {name} [C1].", ["C1"])], critic=[critic_reply()]
        )

        def factory(rs=rs, settings=settings, llm=llm, name=name) -> AnswerEngine:  # type: ignore[no-untyped-def]
            built[name] += 1
            return AnswerEngine(
                settings, rs, llm, RetrievalResources(rs), load_threshold_file=False
            )

        sources = d / "SOURCES.csv"
        if name == "books":
            sources.write_text(TEXTBOOK_SOURCES, encoding="utf-8")
        entries.append(
            CollectionEntry(
                name=name,
                settings=settings,
                engine_factory=factory,
                label=name.title(),
                texts={
                    "heading": f"Ask the {name}",
                    "disclaimer": disclaimer,
                    "abstention_message": "Not found in these documents",
                },
                sources=read_sources(sources),
                examples=example_questions(questions, 5),
                feedback_path=d / "feedback.jsonl",
            )
        )
    entries.append(
        CollectionEntry(
            name="later",
            settings=entries[0].settings,
            engine_factory=lambda: (_ for _ in ()).throw(AssertionError("must not build")),
            label="Later",
            available=False,
            note="not built yet",
        )
    )
    client = TestClient(
        create_multi_app(entries, "books", health=lambda: None), base_url="http://127.0.0.1:8000"
    )
    return client, built, tmp_path


def test_collections_and_meta(app: tuple[TestClient, dict[str, int], Path]) -> None:
    c, built, _tmp = app
    cols = c.get("/api/collections").json()
    assert cols["default"] == "books"
    assert [x["id"] for x in cols["collections"]] == ["law", "books", "later"]
    assert [x["available"] for x in cols["collections"]] == [True, True, False]
    books = c.get("/api/meta", params={"collection": "books"}).json()
    law = c.get("/api/meta", params={"collection": "law"}).json()
    default = c.get("/api/meta").json()
    assert default["collection"] == "books" and default["disclaimer"] == "Open textbooks only."
    assert law["disclaimer"] == "Not legal advice." and law["heading"] == "Ask the law"
    assert books["abstention_message"] == "Not found in these documents"
    doc = books["documents"][0]
    assert doc["title"] == "The Testing Book" and doc["authors"] == "Ann Author"
    assert doc["licence"] == "CC BY-SA 4.0 International" and doc["licence_url"].startswith(
        "https://"
    )
    assert doc["source_url"] == "https://example.invalid/tb" and doc["pages_kept"] == "4"
    assert law["documents"] == []
    # example questions come from the dev split only (answerable items)
    assert books["examples"] and all(isinstance(q, str) for q in books["examples"])
    questions = [
        json.loads(x)
        for x in (_tmp / "books" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    dev = {q["question"] for q in questions if split_of(q["id"]) == "dev" and q["answerable"]}
    assert set(books["examples"]) <= dev
    assert built == {"law": 0, "books": 0}  # metadata never loads models
    assert c.get("/api/meta", params={"collection": "nope"}).status_code == 404


def test_ask_passage_and_feedback_follow_the_collection(
    app: tuple[TestClient, dict[str, int], Path],
) -> None:
    c, built, tmp = app
    r = c.post(
        "/ask", json={"question": "Who may file a complaint?", "system": "G4", "collection": "law"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["collection"] == "law" and body["sentences"][0]["text"] == "Answer from law."
    assert built == {"law": 1, "books": 0}
    r2 = c.post("/ask", json={"question": "Who may file a complaint?", "system": "G4"})
    assert r2.json()["collection"] == "books" and "books" in r2.json()["answer"]
    assert built == {"law": 1, "books": 1}  # each engine is built once
    c.post(
        "/ask", json={"question": "Who may file a complaint?", "system": "G4", "collection": "law"}
    )
    assert built == {"law": 1, "books": 1}
    chunk_id = body["passages"]["C1"]["chunk_id"]
    p = c.get("/passage", params={"chunk_id": chunk_id, "collection": "law"})
    assert p.status_code == 200 and p.json()["chunk_id"] == chunk_id
    fb = {"question": "q?", "system": "G4", "rating": "up", "collection": "books"}
    assert c.post("/feedback", json=fb).json() == {"ok": True}
    rows = (tmp / "books" / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["collection"] == "books"
    assert not (tmp / "law" / "feedback.jsonl").exists()


def test_unknown_and_unbuilt_collections_are_refused(
    app: tuple[TestClient, dict[str, int], Path],
) -> None:
    c, _built, _tmp = app
    q = {"question": "Who may file a complaint?", "system": "G4"}
    assert c.post("/ask", json={**q, "collection": "nope"}).status_code == 404
    r = c.post("/ask", json={**q, "collection": "later"})
    assert r.status_code == 503 and "not built yet" in r.text
    assert c.post("/ask", json={**q, "collection": "../x"}).status_code == 422
    assert (
        c.get(
            "/passage", params={"chunk_id": "structural:x:00000", "collection": "later"}
        ).status_code
        == 503
    )
    html = c.get("/").text
    assert 'id="collection"' in html and 'id="examples"' in html
    js = c.get("/static/app.js").text
    assert "/api/collections" in js and ".innerHTML" not in js and "collection: collection" in js
