"""FastAPI app for the local web UI, serving one or more collections.

Routes:

* ``GET /``: the single page (static HTML, CSS and vanilla JS under ``/static``);
* ``GET /api/collections``: the selectable collections and the one shown first;
* ``GET /api/meta?collection=``: one collection's texts (heading, disclaimer, abstention
  message), systems, the default system, its documents with titles, authors, licence and
  source links (from the collection's ``SOURCES.csv``) and example questions (dev split only);
* ``POST /ask``: ``{"question", "system", "plain", "collection"}`` -> the answer record with
  sentences, citations, cited passages (text included), closest sections when abstaining, the
  step trace with timings and the cost;
* ``GET /passage?chunk_id=...&collection=``: one passage (heading path, section ids, pages, text);
* ``POST /feedback``: thumbs up/down appended to the collection's feedback file (git-ignored).

The server is meant for one local user: requests are answered one at a time, the host header must
be a loopback name, and responses carry a strict Content-Security-Policy (no inline script).
Collections build their retrieval models on first use; switching collections releases the
encoders of the others, so only one collection's models are held at a time.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
ABSTENTION_MESSAGE = "Not found in these documents"
LOOPBACK_HOSTS = ["127.0.0.1", "localhost"]
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
_COLLECTION_ID = r"^[A-Za-z0-9_-]{1,40}$"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=1000)
    system: str | None = Field(default=None, max_length=16)
    plain: bool = False
    collection: str | None = Field(default=None, pattern=_COLLECTION_ID)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    system: str = Field(min_length=1, max_length=16)
    outcome: str = Field(default="", max_length=20)
    answer: str = Field(default="", max_length=5000)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=1000)
    collection: str | None = Field(default=None, pattern=_COLLECTION_ID)


def read_sources(path: Path) -> list[dict[str, Any]]:
    """Corpus documents with their versions, authors, licence and links from SOURCES.csv
    (either the statute columns or the textbook columns)."""
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
                "title": r.get("official_title") or r.get("title", ""),
                "authors": r.get("authors", ""),
                "edition": r.get("edition", ""),
                "licence": r.get("licence", ""),
                "licence_url": r.get("licence_url", ""),
                "licence_source": r.get("licence_source", ""),
                "chapters": r.get("chapters", ""),
                "file": r.get("file_name", ""),
                "pages": r.get("pages", ""),
                "pages_kept": r.get("pages_kept", ""),
                "version": f"sha256 {sha[:12]}" if sha else "",
                "source_url": r.get("source_url", ""),
                "downloaded_on": r.get("downloaded_on", ""),
            }
        )
    return out


def example_questions(questions_path: Path, n: int) -> list[str]:
    """Up to ``n`` answerable DEV-split questions, one per question type in turn (by id)."""
    if n <= 0 or not questions_path.exists():
        return []
    from ragsys.eval.split import questions_for_split
    from ragsys.eval.validate import load_questions

    questions, _errors = load_questions(questions_path)
    dev = sorted(
        (q for q in questions_for_split(questions, "dev") if q.answerable), key=lambda q: q.id
    )
    by_type: dict[str, list[str]] = {}
    for q in dev:
        by_type.setdefault(q.type, []).append(q.question)
    out: list[str] = []
    while len(out) < n and any(by_type.values()):
        for t in list(by_type):
            if by_type[t] and len(out) < n:
                out.append(by_type[t].pop(0))
    return out


@dataclass
class CollectionEntry:
    """One collection served by the app; the engine is built on first use."""

    name: str
    settings: GenerationSettings
    engine_factory: Callable[[], AnswerEngine]
    label: str = ""
    texts: dict[str, str] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    feedback_path: Path | None = None
    available: bool = True
    note: str = ""
    _engine: AnswerEngine | None = None

    def engine(self) -> AnswerEngine:
        if self._engine is None:
            self._engine = self.engine_factory()
        return self._engine

    @property
    def built(self) -> bool:
        return self._engine is not None

    def ui_text(self, key: str, default: str = "") -> str:
        return self.texts.get(key, default)


def _default_texts(disclaimer: str = DISCLAIMER) -> dict[str, str]:
    return {
        "eyebrow": "Consumer Protection Act, 2019 · four Rules",
        "heading": "Ask the consumer law documents",
        "lede": (
            "Answers come only from the Act and its Rules. Every sentence links to the passage "
            "it rests on, and when the documents do not say, you are told so."
        ),
        "placeholder": (
            "For example: Within how many days must a complaint be decided if no testing of "
            "goods is needed?"
        ),
        "disclaimer": disclaimer,
        "abstention_message": ABSTENTION_MESSAGE,
    }


def create_app(
    settings: GenerationSettings,
    engine: AnswerEngine,
    *,
    feedback_path: Path | None = None,
    sources_path: Path | None = None,
    allowed_hosts: Sequence[str] = LOOPBACK_HOSTS,
    health: Callable[[], str | None] | None = None,
) -> FastAPI:
    """The app for a single collection (named "default") with the consumer-law texts."""
    entry = CollectionEntry(
        name="default",
        settings=settings,
        engine_factory=lambda: engine,
        label="Documents",
        texts=_default_texts(),
        sources=read_sources(sources_path or resolve_path(settings.ui.sources_csv)),
        feedback_path=feedback_path or resolve_path(settings.ui.feedback_file),
        _engine=engine,
    )
    check = health or (lambda: llm_unreachable_message(engine.llm))
    return create_multi_app([entry], "default", allowed_hosts=allowed_hosts, health=check)


def create_multi_app(
    entries: Sequence[CollectionEntry],
    default: str,
    *,
    allowed_hosts: Sequence[str] = LOOPBACK_HOSTS,
    health: Callable[[], str | None] | None = None,
) -> FastAPI:
    by_name = {e.name: e for e in entries}
    if default not in by_name:
        raise ValueError(f"default collection {default!r} is not served")
    app = FastAPI(title="Document answers", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    write_lock = threading.Lock()
    work_lock = threading.Lock()  # one question at a time across all collections

    def entry_for(name: str | None) -> CollectionEntry:
        e = by_name.get(name or default)
        if e is None:
            raise HTTPException(404, f"unknown collection {name!r}")
        if not e.available:
            raise HTTPException(503, e.note or f"collection {e.name!r} is not built yet")
        return e

    def activate(e: CollectionEntry) -> AnswerEngine:
        """Build the collection's engine; release the encoders of every other collection."""
        for other in entries:
            if other is not e and other.built:
                other.engine().resources.release_encoders()
        return e.engine()

    def check(e: CollectionEntry) -> str | None:
        if health is not None:
            return health()
        return llm_unreachable_message(e.engine().llm) if e.built else None

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

    @app.get("/api/collections")
    def collections() -> dict[str, Any]:
        return {
            "default": default,
            "collections": [
                {"id": e.name, "label": e.label or e.name, "available": e.available, "note": e.note}
                for e in entries
            ],
        }

    @app.get("/api/meta")
    def meta(
        collection: str | None = Query(default=None, pattern=_COLLECTION_ID),
    ) -> dict[str, Any]:
        e = by_name.get(collection or default)
        if e is None:
            raise HTTPException(404, f"unknown collection {collection!r}")
        s = e.settings
        threshold_tuned = (
            e.engine().threshold is not None if e.built else s.threshold_path().exists()
        )
        return {
            "collection": e.name,
            "label": e.label or e.name,
            "available": e.available,
            "note": e.note,
            "eyebrow": e.ui_text("eyebrow"),
            "heading": e.ui_text("heading"),
            "lede": e.ui_text("lede"),
            "placeholder": e.ui_text("placeholder"),
            "systems": [{"id": n, "description": s.system(n).description} for n in s.ui.systems],
            "default_system": s.ui.default_system,
            "threshold_tuned": threshold_tuned,
            "disclaimer": e.ui_text("disclaimer"),
            "abstention_message": e.ui_text("abstention_message", ABSTENTION_MESSAGE),
            "documents": e.sources,
            "examples": e.examples,
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> JSONResponse:
        e = entry_for(req.collection)
        s = e.settings
        system = req.system or s.ui.default_system
        if system not in s.ui.systems:
            raise HTTPException(400, f"unknown system {system!r}")
        question = " ".join(req.question.split())
        if len(question) < 3:
            raise HTTPException(400, "please type a question")
        with work_lock:
            engine = activate(e)
            problem = check(e)
            if problem:
                raise HTTPException(503, problem)
            result = engine.ask(question, system, plain=req.plain)
        body = result.to_api()
        body["system_description"] = s.system(system).description
        body["collection"] = e.name
        body["abstention_message"] = e.ui_text("abstention_message", ABSTENTION_MESSAGE)
        return JSONResponse(body)

    @app.get("/passage")
    def passage(
        chunk_id: str = Query(min_length=3, max_length=200),
        collection: str | None = Query(default=None, pattern=_COLLECTION_ID),
    ) -> dict[str, Any]:
        e = entry_for(collection)
        chunk = e.engine().chunk(chunk_id)
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
        e = by_name.get(req.collection or default)
        if e is None:
            raise HTTPException(404, f"unknown collection {req.collection!r}")
        path = e.feedback_path or resolve_path(e.settings.ui.feedback_file)
        row = {
            "utc_time": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            **req.model_dump(exclude={"collection"}),
        }
        if req.collection:
            row["collection"] = req.collection
        with write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"ok": True}

    return app
