"""FastAPI app of the web UI, serving one or more collections.

Routes:

* ``GET /``: the chat assistant (:mod:`ragsys.assistant`; static HTML, CSS and vanilla JS under
  ``/static`` with vendored KaTeX and highlight.js) when an assistant is attached, else the
  grounded research page;
* ``GET /research``: the grounded research page (static HTML, CSS and vanilla JS);
* the assistant's API (``/api/chat``, ``/api/verify``, ``/api/upload``, ``/api/health``, ...,
  see :mod:`ragsys.assistant.routes`);
* ``GET /api/collections``: the selectable collections (with "Evaluated" / "Not evaluated") and
  the one shown first;
* ``GET /api/backends``: the generator backends (label, model, privacy note, availability) and
  the default one (hosted when LLM_API_KEY is set and the endpoint answers, else local);
* ``GET /api/meta?collection=``: one collection's texts (heading, disclaimer, safety note,
  abstention message), systems, the default system, its documents with titles, authors, licence
  and source links (from the collection's ``SOURCES.csv``), example questions, whether it is
  evaluated (with the link of its summary page) and the index it serves;
* ``POST /ask``: ``{"question", "system", "plain", "collection", "mode", "backend"}`` with mode
  ``grounded`` (default) or ``hybrid`` -> the grounded answer record (sentences, citations,
  cited passages, closest sections when abstaining, step trace, cost), the generator that
  produced it and, in hybrid mode after an abstention, a separate general-knowledge answer
  under its banner;
* ``POST /chat``: ``{"messages", "collection", "backend", "use_documents", "code"}`` -> a chat
  or coding answer as a safe Markdown block tree (and the passages it cites with documents);
* ``GET /passage?chunk_id=...&collection=``: one passage (heading path, section ids, pages, text);
* ``POST /feedback``: thumbs up/down appended to the collection's feedback file (git-ignored);
* ``GET /summary?collection=`` and ``GET /api/summary?collection=``: an evaluated collection's
  SUMMARY.md rendered as a page (tables included).

Locally the server is meant for one user: grounded requests are answered one at a time, the host
header must be a loopback name, and responses carry a strict Content-Security-Policy (no inline
script, no external resources). In public mode (a hosted demo) the research routes are switched
off, any host name may be allowed and the page may be framed by the hosting site only. Model text reaches the page only as text nodes: Markdown is parsed on the
server into a block tree (:mod:`ragsys.app.markdown`) and built with DOM calls. Collections build
their retrieval models on first use; switching collections releases the encoders of the others,
so only one collection's models are held at a time.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import mimetypes
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import openai
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ragsys.agents.config import GenerationSettings
from ragsys.agents.engine import AnswerEngine
from ragsys.agents.modes import Assistant, ChatTurn, ModeSettings
from ragsys.agents.runtime import llm_unreachable_message
from ragsys.app.markdown import parse_markdown
from ragsys.assistant.routes import build_router
from ragsys.assistant.service import ChatAssistant
from ragsys.config import resolve_path
from ragsys.llm.backends import BackendUnavailable
from ragsys.llm.client import ChatModel, QuotaExceeded

STATIC_DIR = Path(__file__).resolve().parent / "static"
mimetypes.add_type("font/woff2", ".woff2")  # vendored KaTeX fonts (served with nosniff)
DISCLAIMER = "Information from the Consumer Protection Act, 2019 and four Rules. Not legal advice."
ABSTENTION_MESSAGE = "Not found in these documents"
LOOPBACK_HOSTS = ["127.0.0.1", "localhost"]
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
# research routes, switched off in public mode (the assistant offers "Grounded only" instead)
RESEARCH_PATHS = frozenset(
    {
        "/research",
        "/ask",
        "/chat",
        "/passage",
        "/feedback",
        "/api/meta",
        "/api/backends",
        "/summary",
        "/api/summary",
    }
)


def content_security_policy(frame_ancestors: str = "'none'") -> str:
    """The strict policy; only who may frame the page can change (a hosting site's page)."""
    return CSP.replace("frame-ancestors 'none'", f"frame-ancestors {frame_ancestors}")


_COLLECTION_ID = r"^[A-Za-z0-9_-]{1,40}$"
# failures of a generator backend during a request: answered with 503 and the reason; the app
# never switches to another backend on its own
GENERATOR_ERRORS = (QuotaExceeded, BackendUnavailable, openai.APIError)


def generator_failure(backend: str, exc: BaseException) -> str:
    if isinstance(exc, QuotaExceeded):
        why = "its API quota is used up for now"
    elif isinstance(exc, openai.APIConnectionError):
        why = "its endpoint could not be reached"
    elif isinstance(exc, openai.APIStatusError):
        why = f"its API answered with an error (HTTP {exc.status_code})"
    else:
        why = str(exc)[:300]
    return (
        f"The {backend} generator failed: {why}. No other model was used; try again later or "
        "choose another backend."
    )


_BACKEND_ID = r"^[A-Za-z0-9_-]{1,20}$"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=1000)
    system: str | None = Field(default=None, max_length=16)
    plain: bool = False
    collection: str | None = Field(default=None, pattern=_COLLECTION_ID)
    mode: Literal["grounded", "hybrid"] = "grounded"
    backend: str | None = Field(default=None, pattern=_BACKEND_ID)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1, max_length=60)
    collection: str | None = Field(default=None, pattern=_COLLECTION_ID)
    backend: str | None = Field(default=None, pattern=_BACKEND_ID)
    use_documents: bool = False
    code: bool | None = None  # None: the collection's default (coding assistant for cs_ai)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    system: str = Field(min_length=1, max_length=16)
    outcome: str = Field(default="", max_length=20)
    answer: str = Field(default="", max_length=5000)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=1000)
    collection: str | None = Field(default=None, pattern=_COLLECTION_ID)
    mode: str = Field(default="", max_length=16)
    model: str = Field(default="", max_length=120)


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
    evaluated: bool = True
    summary_text: Callable[[], str | None] | None = None  # SUMMARY.md text (evaluated only)
    index_note: str = ""  # which index is served (e.g. app index not built yet)
    chat_llm_factory: Callable[[str], ChatModel] | None = None  # chat client per backend
    history_tokens: Callable[[str], int] | None = None
    _engine: AnswerEngine | None = None
    _assistant: Assistant | None = None

    def engine(self) -> AnswerEngine:
        if self._engine is None:
            self._engine = self.engine_factory()
        return self._engine

    def assistant(self) -> Assistant:
        if self._assistant is None:
            engine = self.engine()
            chat = self.chat_llm_factory or engine.llm_for
            self._assistant = Assistant(
                engine,
                ModeSettings(
                    collection=self.name,
                    domain=self.ui_text("domain"),
                    safety_note=self.ui_text("safety_note"),
                    code_assistant=self.ui_text("code_assistant") == "true",
                    abstention_message=self.ui_text("abstention_message", ABSTENTION_MESSAGE),
                ),
                chat,
                self.history_tokens,
            )
        return self._assistant

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
        "safety_note": "Not legal advice.",
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
    backends: Sequence[dict[str, Any]] | None = None,
    default_backend: str | None = None,
    backend_status: Callable[[str], str | None] | None = None,
    assistant: ChatAssistant | None = None,
    public: bool = False,
    frame_ancestors: str = "'none'",
    trust_forwarded: bool = False,
    forwarded_hops: int = 1,
) -> FastAPI:
    """``backends``: what /api/backends lists (name, label, model, provider, privacy);
    ``default_backend``: the backend used when a request names none; ``backend_status(name)``:
    why a backend cannot answer now (None when it can). ``health`` (tests) overrides both.
    ``assistant``: the chat assistant served at "/" (the research page moves to /research);
    ``public``: hosted demo (research routes off); ``frame_ancestors``: who may frame the page;
    ``trust_forwarded`` / ``forwarded_hops``: client addresses from X-Forwarded-For behind that
    many reverse proxies (rate limits)."""
    by_name = {e.name: e for e in entries}
    if default not in by_name:
        raise ValueError(f"default collection {default!r} is not served")
    backend_list = [dict(b) for b in (backends or [])]
    backend_names = [str(b["name"]) for b in backend_list]
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

    def backend_name(requested: str | None, e: CollectionEntry) -> str:
        if requested is not None:
            if backend_names and requested not in backend_names:
                raise HTTPException(400, f"unknown backend {requested!r}")
            return requested
        return default_backend or e.engine().backend

    def check(e: CollectionEntry, backend: str) -> str | None:
        if health is not None:
            return health()
        if backend_status is not None:
            return backend_status(backend)
        return llm_unreachable_message(e.engine().llm_for(backend)) if e.built else None

    policy = content_security_policy(frame_ancestors)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Any]) -> Response:
        if public and request.url.path in RESEARCH_PATHS:
            response: Response = JSONResponse(
                {"detail": "not available in the public demo"}, status_code=404
            )
        else:
            response = await call_next(request)
        response.headers["Content-Security-Policy"] = policy
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        page = "chat.html" if assistant is not None else "index.html"
        return FileResponse(STATIC_DIR / page, media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.get("/research", include_in_schema=False)
    def research() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/summary", include_in_schema=False)
    def summary_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "summary.html", media_type="text/html")

    @app.get("/api/collections")
    def collections() -> dict[str, Any]:
        extra = {c["id"]: c for c in assistant.collections_info()} if assistant else {}
        rows = []
        for e in entries:
            row: dict[str, Any] = {
                "id": e.name,
                "label": e.label or e.name,
                "available": e.available,
                "note": e.note,
                "evaluated": e.evaluated,
            }
            if e.name in extra:
                more = extra[e.name]
                row = {**more, **row, "available": e.available and bool(more["available"])}
            rows.append(row)
        return {"default": default, "collections": rows}

    @app.get("/api/backends")
    def list_backends() -> dict[str, Any]:
        out = []
        for b in backend_list:
            reason = backend_status(str(b["name"])) if backend_status else None
            out.append({**b, "available": reason is None, "reason": reason or ""})
        return {"default": default_backend, "backends": out}

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
            "safety_note": e.ui_text("safety_note"),
            "abstention_message": e.ui_text("abstention_message", ABSTENTION_MESSAGE),
            "documents": e.sources,
            "examples": e.examples,
            "evaluated": e.evaluated,
            "summary_url": f"/summary?collection={e.name}" if e.evaluated else None,
            "code_assistant": e.ui_text("code_assistant") == "true",
            "index_note": e.index_note,
            "modes": ["grounded", "hybrid", "chat"],
        }

    @app.get("/api/summary")
    def summary(
        collection: str | None = Query(default=None, pattern=_COLLECTION_ID),
    ) -> dict[str, Any]:
        e = by_name.get(collection or default)
        if e is None:
            raise HTTPException(404, f"unknown collection {collection!r}")
        if not e.evaluated:
            raise HTTPException(404, f"{e.label or e.name} is not evaluated")
        text = e.summary_text() if e.summary_text is not None else None
        if not text:
            raise HTTPException(404, "no summary has been written for this collection yet")
        return {
            "collection": e.name,
            "label": e.label or e.name,
            "blocks": parse_markdown(text, tables=True),
        }

    def _system(e: CollectionEntry, requested: str | None) -> str:
        s = e.settings
        system = requested or s.ui.default_system
        if system not in s.ui.systems:
            raise HTTPException(400, f"unknown system {system!r}")
        return system

    @app.post("/ask")
    def ask(req: AskRequest) -> JSONResponse:
        e = entry_for(req.collection)
        system = _system(e, req.system)
        question = " ".join(req.question.split())
        if len(question) < 3:
            raise HTTPException(400, "please type a question")
        with work_lock:
            activate(e)
            backend = backend_name(req.backend, e)
            problem = check(e, backend)
            if problem:
                raise HTTPException(503, problem)
            try:
                result = e.assistant().ask(
                    question, system, req.mode, backend=backend, plain=req.plain
                )
            except GENERATOR_ERRORS as exc:
                raise HTTPException(503, generator_failure(backend, exc)) from exc
        body = result.to_api()
        body["system_description"] = e.settings.system(system).description
        body["collection"] = e.name
        body["abstention_message"] = e.ui_text("abstention_message", ABSTENTION_MESSAGE)
        return JSONResponse(body)

    @app.post("/chat")
    def chat(req: ChatRequest) -> JSONResponse:
        e = entry_for(req.collection)
        if req.messages[-1].role != "user":
            raise HTTPException(400, "the last message must be the user's")
        turns = [ChatTurn(m.role, m.content) for m in req.messages]
        with work_lock:
            activate(e)
            backend = backend_name(req.backend, e)
            problem = check(e, backend)
            if problem:
                raise HTTPException(503, problem)
            try:
                result = e.assistant().chat(
                    turns,
                    system=e.settings.ui.default_system,
                    backend=backend,
                    use_documents=req.use_documents,
                    code=req.code,
                )
            except GENERATOR_ERRORS as exc:
                raise HTTPException(503, generator_failure(backend, exc)) from exc
        return JSONResponse(result.to_api())

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
            **req.model_dump(exclude={"collection", "mode", "model"}),
        }
        if req.mode:
            row["mode"] = req.mode
        if req.model:
            row["model"] = req.model
        if req.collection:
            row["collection"] = req.collection
        with write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"ok": True}

    if assistant is not None:
        app.include_router(
            build_router(assistant, trust_forwarded=trust_forwarded, forwarded_hops=forwarded_hops)
        )
    return app
