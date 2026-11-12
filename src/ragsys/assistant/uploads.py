"""Uploaded PDFs: parsed, chunked and embedded in memory, searchable for one browser session.

A PDF is checked (at most ``max_mb``, a real PDF, not encrypted, at most ``max_pages`` pages,
with a text layer), extracted from memory with the ingest pipeline's line extraction and
cleaner, structured with the :class:`GenericParser`, cut into structural chunks and embedded
with bge-small. Nothing is written to disk or logged; the bytes are dropped after processing.
A session holds at most ``max_files`` documents and is deleted when the page ends it or after
``ttl_s`` seconds without activity.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from ragsys.assistant.config import UploadSettings
from ragsys.assistant.index import DenseIndex, LockedEncoder
from ragsys.chunking.base import Chunk, DocInput
from ragsys.chunking.structural import StructuralChunker
from ragsys.chunking.tokenizer import TokenCounter
from ragsys.config import CleaningConfig, DocumentConfig
from ragsys.ingest.clean import clean_document
from ragsys.ingest.extract import (
    _PUA_ONLY_RE,
    Page,
    RawDocument,
    Span,
    _make_span,
    _mark_scripts,
    _merge_same_baseline,
)
from ragsys.structure.generic import GenericParser

UPLOAD_SOURCE = "upload"
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
_NAME_BAD_RE = re.compile(r"[\x00-\x1f\x7f<>\\/:\"|?*]+")
Progress = Callable[[str, int, int], None]  # (stage, done, total)


class UploadError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def clean_name(raw: str) -> str:
    """A display name for an uploaded file (no path, no control characters)."""
    name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    name = _NAME_BAD_RE.sub(" ", name).strip() or "document.pdf"
    return name[:120]


@dataclass
class UploadedDoc:
    doc_id: str
    name: str
    pages: int
    index: DenseIndex
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "name": self.name,
            "pages": self.pages,
            "chunks": len(self.index),
            "size_bytes": self.size_bytes,
        }


@dataclass
class UploadSession:
    session_id: str
    last_seen: float
    docs: dict[str, UploadedDoc] = field(default_factory=dict)
    pending: int = 0  # uploads being processed


class UploadStore:
    """Sessions of uploaded documents, in memory only."""

    def __init__(
        self, settings: UploadSettings, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.settings = settings
        self._clock = clock
        self._sessions: dict[str, UploadSession] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def purge_expired(self) -> int:
        """Delete sessions inactive for ``ttl_s`` (one with an upload in progress gets twice
        as long, so an abandoned upload cannot keep a session alive)."""
        now = self._clock()
        ttl = self.settings.ttl_s
        with self._lock:
            dead = [
                sid
                for sid, s in self._sessions.items()
                if now - s.last_seen > (2 * ttl if s.pending else ttl)
            ]
            for sid in dead:
                del self._sessions[sid]
        return len(dead)

    def get(self, session_id: str | None) -> UploadSession | None:
        """The live session (its inactivity clock restarts), or None."""
        if not session_id or not SESSION_RE.match(session_id):
            return None
        self.purge_expired()
        with self._lock:
            s = self._sessions.get(session_id)
            if s is not None:
                s.last_seen = self._clock()
            return s

    def reserve(self, session_id: str | None) -> UploadSession:
        """The session that receives a new upload (created when needed); counts the upload as
        pending so the session is not purged while it is processed."""
        self.purge_expired()
        with self._lock:
            s = self._sessions.get(session_id or "") if session_id else None
            if s is None:
                if len(self._sessions) >= self.settings.max_sessions:
                    raise UploadError(
                        503, "Too many people are uploading right now. Please try again later."
                    )
                sid = secrets.token_urlsafe(24)
                s = UploadSession(sid, self._clock())
                self._sessions[sid] = s
            if len(s.docs) + s.pending >= self.settings.max_files:
                raise UploadError(
                    409,
                    f"At most {self.settings.max_files} PDFs per session. Remove one to add another.",
                )
            s.pending += 1
            s.last_seen = self._clock()
            return s

    def release(self, session: UploadSession, doc: UploadedDoc | None) -> None:
        with self._lock:
            session.pending = max(0, session.pending - 1)
            session.last_seen = self._clock()
            if doc is not None and session.session_id in self._sessions:
                session.docs[doc.doc_id] = doc

    def remove(self, session_id: str, doc_id: str) -> bool:
        s = self.get(session_id)
        if s is None:
            return False
        with self._lock:
            return s.docs.pop(doc_id, None) is not None

    def end(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def indexes(self, session_id: str | None) -> list[DenseIndex]:
        s = self.get(session_id)
        if s is None:
            return []
        with self._lock:
            return [d.index for d in s.docs.values()]

    def chunk(self, session_id: str | None, chunk_id: str) -> Chunk | None:
        s = self.get(session_id)
        if s is None:
            return None
        with self._lock:
            for d in s.docs.values():
                c = d.index.by_id.get(chunk_id)
                if c is not None:
                    return c
        return None

    def documents(self, session_id: str | None) -> list[dict[str, Any]]:
        s = self.get(session_id)
        if s is None:
            return []
        with self._lock:
            return [d.to_dict() for d in s.docs.values()]


# --------------------------------------------------------------------------- PDF processing


def check_pdf(data: bytes, settings: UploadSettings) -> int:
    """Validate an upload; returns its page count or raises :class:`UploadError`."""
    if len(data) > settings.max_bytes:
        raise UploadError(413, f"The file is larger than {settings.max_mb:g} MB.")
    if not data or b"%PDF-" not in data[:1024]:
        raise UploadError(415, "Only PDF files can be uploaded.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise UploadError(415, "The file could not be read as a PDF.") from exc
    with doc:
        meta = doc.metadata or {}
        if doc.needs_pass or doc.is_encrypted or meta.get("encryption"):
            raise UploadError(
                422,
                "The PDF is encrypted or password-protected. Please upload an unprotected copy.",
            )
        pages = int(doc.page_count)
    if pages == 0:
        raise UploadError(422, "The PDF has no pages.")
    if pages > settings.max_pages:
        raise UploadError(
            413, f"The PDF has {pages} pages; at most {settings.max_pages} are allowed."
        )
    return pages


def extract_pdf_bytes(
    data: bytes, name: str, progress: Progress | None = None, marker_max_size_ratio: float = 0.8
) -> RawDocument:
    """Text lines with font metadata from an in-memory PDF (the same line extraction as the
    ingest pipeline; small raised spans are kept as ``^x``)."""
    out_pages: list[Page] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
        total = int(doc.page_count)
        for index in range(total):
            page = doc[index]
            info = page.get_text("dict")
            raw_lines: list[list[Span]] = []
            rotated = 0
            for block in info.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for ln in block.get("lines", []):
                    dx, dy = ln.get("dir", (1.0, 0.0))
                    if abs(float(dy)) > 0.02 or float(dx) < 0.98:
                        rotated += 1
                        continue
                    spans = [_make_span(s) for s in ln.get("spans", []) if s.get("text")]
                    spans = [s for s in spans if not _PUA_ONLY_RE.match(s.text)]
                    if spans:
                        raw_lines.append(spans)
            lines = _merge_same_baseline(raw_lines, index + 1)
            markers = sum(_mark_scripts(ln, marker_max_size_ratio) for ln in lines)
            rect = page.rect
            out_pages.append(
                Page(index + 1, float(rect.width), float(rect.height), lines, rotated, markers)
            )
            if progress is not None and ((index + 1) % 10 == 0 or index + 1 == total):
                progress("reading", index + 1, total)
    return RawDocument(path=Path(name), pages=out_pages)


def new_doc_id() -> str:
    return "up_" + secrets.token_hex(5)


def ingest_pdf(
    data: bytes,
    name: str,
    encoder: LockedEncoder,
    tokenizer: TokenCounter,
    settings: UploadSettings,
    progress: Progress | None = None,
    *,
    doc_id: str | None = None,
) -> UploadedDoc:
    """Check, parse, chunk and embed one PDF in memory."""
    pages = check_pdf(data, settings)
    did = doc_id or new_doc_id()
    display = clean_name(name)
    title = display[:-4] if display.lower().endswith(".pdf") else display
    raw = extract_pdf_bytes(data, display, progress)
    if not raw.has_text_layer:
        raise UploadError(
            422, "No text was found in this PDF (scanned pages are not supported; OCR is off)."
        )
    cfg = DocumentConfig(
        file=display,
        doc_id=did,
        title=title or "Uploaded document",
        doc_type="generic",
        parser="generic",
    )
    clean = clean_document(raw, cfg, CleaningConfig())
    if progress is not None:
        progress("structuring", 0, 1)
    parsed = GenericParser().parse(clean, cfg)
    doc = DocInput(
        doc_id=did,
        title=cfg.title,
        doc_type="generic",
        text=clean.text,
        page_spans=clean.page_spans,
        nodes=parsed.nodes,
    )
    chunks = StructuralChunker(tokenizer, 384, 48, True, orphan_text=True).chunk_document(doc)
    if not chunks:
        raise UploadError(422, "No text could be extracted from this PDF.")
    if len(chunks) > settings.max_chunks_per_file:
        raise UploadError(
            413, "The PDF has too much text for this demo. Please upload a shorter one."
        )

    def embedded(done: int, total: int) -> None:
        if progress is not None:
            progress("embedding", done, total)

    index = DenseIndex.build(chunks, encoder, UPLOAD_SOURCE, progress=embedded)
    return UploadedDoc(did, display, pages, index, len(data))
