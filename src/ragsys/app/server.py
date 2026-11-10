"""FastAPI app for the local web UI.

Routes:

* ``GET /``: the single page (static HTML, CSS and vanilla JS under ``/static``);
* ``GET /api/meta``: selectable systems, the default, the disclaimer and the corpus documents
  with their versions (from ``data/raw/SOURCES.csv``);
* ``POST /ask``: ``{"question", "system", "plain"}`` -> the answer record with sentences,
  citations, cited passages (text included), closest sections when abstaining, the step trace
  with timings and the cost;
* ``GET /passage?chunk_id=...``: one passage (heading path, section ids, pages, text);
* ``POST /feedback``: thumbs up/down appended to ``data/feedback.jsonl`` (git-ignored).

The server is meant for one local user: requests are answered one at a time, the host header must
be a loopback name, and responses carry a strict Content-Security-Policy (no inline script).
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ragsys.agents.config import GenerationSettings
from ragsys.agents.engine import AnswerEngine
from ragsys.agents.runtime import llm_unreachable_message
from ragsys.config import resolve_path

STATIC_DIR = Path(__file__).resolve().parent / "static"
DISCLAIMER = "Information from the Consumer Protection Act, 2019 and four Rules. Not legal advice."
LOOPBACK_HOSTS = ["127.0.0.1", "localhost"]
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=1000)
    system: str | None = Field(default=None, max_length=16)
    plain: bool = False


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    system: str = Field(min_length=1, max_length=16)
    outcome: str = Field(default="", max_length=20)
    answer: str = Field(default="", max_length=5000)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=1000)


def read_sources(path: Path) -> list[dict[str, Any]]:
    """Corpus documents and their versions (file hash, pages) from SOURCES.csv."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        sha = (r.get("sha256") or "").strip()
        out.append(
            {
                "doc_id": r.get("doc_id", ""),
                "title": r.get("official_title", ""),
                "file": r.get("file_name", ""),
                "pages": r.get("pages", ""),
                "version": f"sha256 {sha[:12]}" if sha else "",
                "source_url": r.get("source_url", ""),
                "downloaded_on": r.get("downloaded_on", ""),
            }
        )
    return out


def create_app(
    settings: GenerationSettings,
    engine: AnswerEngine,
    *,
    feedback_path: Path | None = None,
    sources_path: Path | None = None,
    allowed_hosts: Sequence[str] = LOOPBACK_HOSTS,
    health: Callable[[], str | None] | None = None,
) -> FastAPI:
    app = FastAPI(title="Consumer law answers", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    feedback_file = feedback_path or resolve_path(settings.ui.feedback_file)
    sources = read_sources(sources_path or resolve_path(settings.ui.sources_csv))
    check = health or (lambda: llm_unreachable_message(engine.llm))
    write_lock = threading.Lock()

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Any]) -> Response:
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "systems": [
                {"id": s, "description": settings.system(s).description}
                for s in settings.ui.systems
            ],
            "default_system": settings.ui.default_system,
            "threshold_tuned": engine.threshold is not None,
            "disclaimer": DISCLAIMER,
            "documents": sources,
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> JSONResponse:
        system = req.system or settings.ui.default_system
        if system not in settings.ui.systems:
            raise HTTPException(400, f"unknown system {system!r}")
        question = " ".join(req.question.split())
        if len(question) < 3:
            raise HTTPException(400, "please type a question")
        problem = check()
        if problem:
            raise HTTPException(503, problem)
        result = engine.ask(question, system, plain=req.plain)
        body = result.to_api()
        body["system_description"] = settings.system(system).description
        return JSONResponse(body)

    @app.get("/passage")
    def passage(chunk_id: str = Query(min_length=3, max_length=200)) -> dict[str, Any]:
        chunk = engine.chunk(chunk_id)
        if chunk is None:
            raise HTTPException(404, "passage not found")
        pages = (
            str(chunk.page_start)
            if chunk.page_start == chunk.page_end
            else f"{chunk.page_start}-{chunk.page_end}"
        )
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "heading_path": chunk.heading_path,
            "section_ids": list(chunk.section_ids),
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "pages": pages,
            "text": chunk.display_text,
        }

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> dict[str, Any]:
        row = {
            "utc_time": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            **req.model_dump(),
        }
        with write_lock:
            feedback_file.parent.mkdir(parents=True, exist_ok=True)
            with feedback_file.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"ok": True}

    return app
