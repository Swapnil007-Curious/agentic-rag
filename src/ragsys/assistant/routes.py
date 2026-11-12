"""HTTP routes of the chat assistant.

* ``POST /api/chat``: a chat request -> Server-Sent Events (``meta``, ``context``, ``token``...,
  ``done`` or ``error``; see :mod:`ragsys.assistant.service`);
* ``POST /api/chat/stop``: ``{"stream_id"}`` stops a running reply (the browser also aborts
  the request; a closed connection stops the reply too);
* ``POST /api/verify``: guards + critic for one grounded reply, per sentence;
* ``POST /api/upload``: the raw PDF as the request body (``X-File-Name``, optional
  ``X-Session-Id``) -> Server-Sent Events with progress, then the document;
* ``GET /api/uploads``, ``POST /api/upload/delete``, ``POST /api/session/end``: list, remove one
  document, or delete the session (the page sends the last one when it closes);
* ``GET /api/health``: backends, limits and upload limits.

Requests are limited per client address and per day (:mod:`ragsys.assistant.limits`); message
contents are never logged.
"""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Literal
from urllib.parse import unquote

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from ragsys.assistant.history import Summary, Turn
from ragsys.assistant.limits import Refusal, client_address
from ragsys.assistant.llm import StreamFailure
from ragsys.assistant.service import ChatAssistant, ChatRequestData, Event, RequestError
from ragsys.assistant.uploads import SESSION_RE, UploadError, check_pdf

_COLLECTION = r"^[A-Za-z0-9_-]{1,40}$"
_BACKEND = r"^[A-Za-z0-9_-]{1,20}$"
_SESSION = r"^[A-Za-z0-9_-]{20,64}$"
SSE_HEADERS = {"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"}


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class SummaryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(max_length=4000)
    upto: int = Field(ge=0, le=1000)
    digest: str = Field(max_length=64)


class ChatBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[MessageIn] = Field(min_length=1, max_length=200)
    collection: str | None = Field(default=None, pattern=_COLLECTION)
    session_id: str | None = Field(default=None, pattern=_SESSION)
    grounded_only: bool = False
    backend: str | None = Field(default=None, pattern=_BACKEND)
    summary: SummaryIn | None = None


class StopBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stream_id: str = Field(min_length=8, max_length=40)


class PassageRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(pattern=r"^C\d{1,3}$")
    chunk_id: str = Field(min_length=3, max_length=200)


class VerifyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=20000)
    passages: list[PassageRef] = Field(default_factory=list, max_length=20)
    collection: str | None = Field(default=None, pattern=_COLLECTION)
    session_id: str | None = Field(default=None, pattern=_SESSION)
    backend: str | None = Field(default=None, pattern=_BACKEND)


class DeleteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(pattern=_SESSION)
    doc_id: str = Field(pattern=r"^up_[0-9a-f]{6,16}$")


def format_sse(ev: Event) -> str:
    return f"event: {ev['event']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"


async def sse_stream(events: Iterator[Event], on_close: Callable[[], None]) -> AsyncIterator[str]:
    """Events as SSE text; ``on_close`` runs when the stream ends or the client goes away."""
    try:
        async for ev in iterate_in_threadpool(events):
            yield format_sse(ev)
    finally:
        on_close()


def refusal_response(r: Refusal) -> JSONResponse:
    headers = {"Retry-After": str(int(r.retry_after) + 1)} if r.retry_after else None
    return JSONResponse({"detail": r.message, "code": r.code}, r.status, headers=headers)


def error_response(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse({"detail": message, "code": code}, status)


def build_router(
    assistant: ChatAssistant, *, trust_forwarded: bool = False, forwarded_hops: int = 1
) -> APIRouter:
    router = APIRouter()

    def client(request: Request) -> str:
        peer = request.client.host if request.client else None
        return client_address(
            peer, request.headers, trust_forwarded=trust_forwarded, hops=forwarded_hops
        )

    @router.get("/api/health")
    def health() -> dict[str, Any]:
        return assistant.health()

    @router.post("/api/chat", response_model=None)
    def chat(body: ChatBody, request: Request) -> Response:
        req = ChatRequestData(
            turns=[Turn(m.role, m.content) for m in body.messages],
            collection=body.collection,
            session_id=body.session_id,
            grounded_only=body.grounded_only,
            backend=body.backend,
            summary=Summary(body.summary.text, body.summary.upto, body.summary.digest)
            if body.summary
            else None,
        )
        try:
            assistant.validate(req)
            assistant.collection(req.collection)
            backend = assistant.backend(req.backend)
        except RequestError as exc:
            return error_response(exc.status, exc.message, exc.code)
        problem = assistant.backend_status(backend.name)
        if problem:
            return error_response(503, problem, "unavailable")
        refusal = assistant.guard.check_message(client(request))
        if refusal is not None:
            return refusal_response(refusal)
        stop = threading.Event()
        return StreamingResponse(
            sse_stream(assistant.chat_events(req, stop), stop.set),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @router.post("/api/chat/stop")
    def stop(body: StopBody) -> dict[str, Any]:
        return {"stopped": assistant.stop(body.stream_id)}

    @router.post("/api/verify", response_model=None)
    async def verify(body: VerifyBody, request: Request) -> Response:
        refusal = assistant.guard.check_message(client(request))
        if refusal is not None:
            return refusal_response(refusal)
        try:
            result = await run_in_threadpool(
                assistant.verify,
                body.question,
                body.answer,
                [(p.label, p.chunk_id) for p in body.passages],
                collection=body.collection,
                session_id=body.session_id,
                backend=body.backend,
            )
        except RequestError as exc:
            return error_response(exc.status, exc.message, exc.code)
        except StreamFailure as exc:
            return error_response(429 if exc.code == "busy" else 503, exc.message, exc.code)
        return JSONResponse(result)

    @router.post("/api/upload", response_model=None)
    async def upload(request: Request) -> Response:
        limits = assistant.settings.uploads
        refusal = assistant.guard.check_upload(client(request))
        if refusal is not None:
            return refusal_response(refusal)
        size = request.headers.get("content-length")
        if size and size.isdigit() and int(size) > limits.max_bytes:
            return error_response(413, f"The file is larger than {limits.max_mb:g} MB.", "upload")
        data = bytearray()
        async for part in request.stream():
            data.extend(part)
            if len(data) > limits.max_bytes:
                return error_response(
                    413, f"The file is larger than {limits.max_mb:g} MB.", "upload"
                )
        name = unquote(request.headers.get("x-file-name", "document.pdf"))[:300]
        session_id = request.headers.get("x-session-id") or None
        if session_id is not None and not SESSION_RE.match(session_id):
            session_id = None
        raw = bytes(data)
        try:
            await run_in_threadpool(check_pdf, raw, limits)
            session = assistant.uploads.reserve(session_id)
        except UploadError as exc:
            return error_response(exc.status, exc.message, "upload")
        return StreamingResponse(
            sse_stream(assistant.upload_events(raw, name, session), lambda: None),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @router.get("/api/uploads")
    def uploads(session_id: str | None = None) -> dict[str, Any]:
        return {"documents": assistant.uploads.documents(session_id)}

    @router.post("/api/upload/delete")
    def delete_upload(body: DeleteBody) -> dict[str, Any]:
        removed = assistant.uploads.remove(body.session_id, body.doc_id)
        return {"removed": removed, "documents": assistant.uploads.documents(body.session_id)}

    @router.post("/api/session/end")
    async def end_session(request: Request) -> dict[str, Any]:
        raw = (await request.body())[:500]
        try:
            sid = json.loads(raw.decode("utf-8")).get("session_id")
        except (ValueError, AttributeError):
            sid = None
        ended = isinstance(sid, str) and bool(SESSION_RE.match(sid)) and assistant.uploads.end(sid)
        return {"ended": bool(ended)}

    return router
