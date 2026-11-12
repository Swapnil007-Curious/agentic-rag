"""Uploaded PDFs: limits and rejections, in-memory processing (nothing written, nothing logged),
searchable alongside the collection, and cleanup (removal, session end, inactivity)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from assistant_helpers import assistant_settings, build_assistant, by_name, make_client, sse_events
from ragsys.assistant.config import UploadSettings
from ragsys.assistant.index import LockedEncoder
from ragsys.assistant.uploads import (
    UploadError,
    UploadStore,
    check_pdf,
    clean_name,
    ingest_pdf,
)
from ragsys.retrieval.fakes import HashingEncoder

SECRET_LINE = "The Zorblat station recorded a melt rate of 4.2 centimetres per day in July."


def make_pdf(pages: int = 2, *, text: bool = True, password: str | None = None) -> bytes:
    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page()
        if text:
            page.insert_text((72, 80), f"Section {p + 1}. Findings", fontsize=16, fontname="hebo")
            page.insert_textbox(
                pymupdf.Rect(72, 100, 520, 300),
                SECRET_LINE + f" Page {p + 1} adds that the director leads five researchers.",
                fontsize=10,
            )
        else:
            page.draw_rect(pymupdf.Rect(72, 72, 300, 300), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    kwargs: dict[str, Any] = {}
    if password:
        kwargs = {
            "encryption": pymupdf.PDF_ENCRYPT_AES_256,
            "user_pw": password,
            "owner_pw": password,
        }
    data: bytes = doc.tobytes(**kwargs)
    return data


def test_check_pdf_limits() -> None:
    s = UploadSettings(max_mb=1, max_pages=3)
    assert check_pdf(make_pdf(2), s) == 2
    with pytest.raises(UploadError) as e:
        check_pdf(b"hello, not a pdf", s)
    assert e.value.status == 415 and "Only PDF" in e.value.message
    with pytest.raises(UploadError) as e:
        check_pdf(b"%PDF-1.7 broken", s)
    assert e.value.status == 415
    with pytest.raises(UploadError) as e:
        check_pdf(make_pdf(1, password="pw"), s)
    assert e.value.status == 422 and "encrypted" in e.value.message
    with pytest.raises(UploadError) as e:
        check_pdf(make_pdf(4), s)
    assert e.value.status == 413 and "4 pages" in e.value.message
    with pytest.raises(UploadError) as e:
        check_pdf(b"%PDF-" + b"0" * (1024 * 1024 + 10), s)
    assert e.value.status == 413 and "1 MB" in e.value.message


def test_ingest_in_memory_writes_and_logs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tokenizer: Any,
) -> None:
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)
    calls: list[str] = []
    doc = ingest_pdf(
        make_pdf(2),
        "C:\\Users\\me\\report<1>.pdf",
        LockedEncoder(HashingEncoder(64)),
        tokenizer,
        UploadSettings(),
        lambda stage, done, total: calls.append(stage),
    )
    assert doc.name == "report 1 .pdf" and doc.pages == 2 and len(doc.index) >= 1
    assert doc.doc_id.startswith("up_") and all(
        c.chunk_id.startswith(f"structural:{doc.doc_id}:") for c in doc.index.chunks
    )
    assert any("Zorblat" in c.display_text for c in doc.index.chunks)
    assert {"reading", "structuring", "embedding"} <= set(calls)
    assert list(tmp_path.iterdir()) == []  # nothing written
    assert not any(
        "Zorblat" in r.getMessage() or "report" in r.getMessage() for r in caplog.records
    )


def test_scanned_pdf_is_rejected(tokenizer: Any) -> None:
    with pytest.raises(UploadError) as e:
        ingest_pdf(
            make_pdf(2, text=False),
            "scan.pdf",
            LockedEncoder(HashingEncoder(64)),
            tokenizer,
            UploadSettings(),
        )
    assert e.value.status == 422 and "No text" in e.value.message


def test_store_limits_and_cleanup() -> None:
    t = [0.0]
    store = UploadStore(UploadSettings(max_files=2, ttl_s=100, max_sessions=2), clock=lambda: t[0])
    s = store.reserve(None)
    sid = s.session_id
    assert len(sid) >= 20 and store.get(sid) is s

    class Doc:
        def __init__(self, i: str) -> None:
            self.doc_id = i
            self.index = type("I", (), {"by_id": {}})()

        def to_dict(self) -> dict[str, str]:
            return {"doc_id": self.doc_id}

    store.release(s, Doc("up_1"))  # type: ignore[arg-type]
    s2 = store.reserve(sid)
    store.release(s2, Doc("up_2"))  # type: ignore[arg-type]
    with pytest.raises(UploadError) as e:
        store.reserve(sid)
    assert e.value.status == 409 and "At most 2" in e.value.message
    assert store.remove(sid, "up_1") and not store.remove(sid, "up_1")
    other = store.reserve(None)
    store.release(other, None)
    with pytest.raises(UploadError) as e:
        store.reserve(None)  # max_sessions
    assert e.value.status == 503
    t[0] = 50.0
    assert store.get(sid) is not None  # activity restarts the clock
    t[0] = 140.0
    assert store.purge_expired() == 1 and store.get(sid) is not None  # the other one expired
    t[0] = 300.0
    assert store.get(sid) is None and len(store) == 0
    s3 = store.reserve(None)
    assert store.end(s3.session_id) and store.get(s3.session_id) is None
    assert store.get("../../etc") is None and store.indexes("bad id") == []


def test_upload_endpoint_then_question_then_cleanup(tmp_path: Path, tokenizer: Any) -> None:
    assistant, _llm = build_assistant(tmp_path, tokenizer=tokenizer)
    c = make_client(assistant)
    r = c.post(
        "/api/upload",
        content=make_pdf(2),
        headers={"Content-Type": "application/pdf", "X-File-Name": "station%20report.pdf"},
    )
    assert r.status_code == 200, r.text
    events = sse_events(r.text)
    assert events[0][0] == "session" and any(n == "progress" for n, _ in events)
    done = by_name(events, "done")[0]
    sid = done["session_id"]
    assert done["document"]["name"] == "station report.pdf" and done["document"]["pages"] == 2
    assert c.get("/api/uploads", params={"session_id": sid}).json()["documents"][0]["pages"] == 2
    # the upload is searched next to the collection
    body = {
        "messages": [{"role": "user", "content": SECRET_LINE}],
        "collection": "books",
        "session_id": sid,
    }
    ev = sse_events(c.post("/api/chat", json=body).text)
    ctx = by_name(ev, "context")[0]
    assert ctx["grounded"] and ctx["passages"][0]["source"] == "upload"
    assert ctx["passages"][0]["title"] == "station report.pdf"
    upload_check = next(ch for ch in ctx["checks"] if ch["source"] == "upload")
    assert upload_check["passed"] and upload_check["threshold"] == 0.5
    # verify resolves upload passages from the session
    p = ctx["passages"][0]
    v = c.post(
        "/api/verify",
        json={
            "question": "melt?",
            "answer": "It was 4.2 centimetres per day [C1].",
            "passages": [{"label": "C1", "chunk_id": p["chunk_id"]}],
            "session_id": sid,
        },
    )
    assert v.status_code == 200 and v.json()["sentences"][0]["status"] == "supported"
    # session end deletes everything
    assert c.post("/api/session/end", content=f'{{"session_id": "{sid}"}}').json() == {
        "ended": True
    }
    assert c.get("/api/uploads", params={"session_id": sid}).json()["documents"] == []
    ev = sse_events(c.post("/api/chat", json=body).text)
    assert all(ch["source"] != "upload" for ch in by_name(ev, "context")[0]["checks"])


def test_upload_rejections_over_http(tmp_path: Path, tokenizer: Any) -> None:
    settings = assistant_settings(uploads={"max_files": 1, "max_mb": 1, "max_pages": 3})
    assistant, _llm = build_assistant(tmp_path, tokenizer=tokenizer, settings=settings)
    c = make_client(assistant)
    pdf = {"Content-Type": "application/pdf", "X-File-Name": "a.pdf"}
    r = c.post("/api/upload", content=b"plain text", headers={"X-File-Name": "a.txt"})
    assert r.status_code == 415 and r.json()["detail"] == "Only PDF files can be uploaded."
    r = c.post("/api/upload", content=make_pdf(1, password="secret"), headers=pdf)
    assert r.status_code == 422 and "encrypted" in r.json()["detail"]
    r = c.post("/api/upload", content=make_pdf(5), headers=pdf)
    assert r.status_code == 413 and "5 pages" in r.json()["detail"]
    r = c.post("/api/upload", content=b"%PDF-" + b"x" * (1024 * 1024 + 1), headers=pdf)
    assert r.status_code == 413
    r = c.post("/api/upload", content=make_pdf(1), headers=pdf)
    sid = by_name(sse_events(r.text), "done")[0]["session_id"]
    r = c.post("/api/upload", content=make_pdf(1), headers={**pdf, "X-Session-Id": sid})
    assert r.status_code == 409 and "At most 1" in r.json()["detail"]
    doc_id = c.get("/api/uploads", params={"session_id": sid}).json()["documents"][0]["doc_id"]
    r = c.post("/api/upload/delete", json={"session_id": sid, "doc_id": doc_id})
    assert r.json() == {"removed": True, "documents": []}


def test_clean_name() -> None:
    assert clean_name("../../etc/passwd") == "passwd"
    assert clean_name('a<script>"x".pdf') == "a script x .pdf"
    assert clean_name("") == "document.pdf" and len(clean_name("a" * 500)) == 120
