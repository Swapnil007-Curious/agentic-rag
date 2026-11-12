"""The chat assistant: one request -> a sequence of events for the browser (sent as Server-Sent
Events by :mod:`ragsys.assistant.routes`).

Chat (default): ``meta`` -> ``context`` (the gate's decision and the passages given to the
model) -> ``token`` ... (the streamed reply) -> ``done`` (the reply with invalid citation labels
removed, the cited passages, the source line, timings and, when one was written, the new
conversation summary). Grounded only: ``meta`` -> ``status`` -> ``done`` (the grounded
pipeline's answer or abstention). A failure ends the sequence with ``error``.
"""

from __future__ import annotations

import queue
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai

from ragsys.agents.config import ContextSettings, GenerationSettings, LoopSettings
from ragsys.agents.context import Passage, passage_from
from ragsys.agents.text import labels_in
from ragsys.assistant.citations import clean_citations, source_line
from ragsys.assistant.config import AssistantSettings
from ragsys.assistant.gate import GateDecision, GroundingGate, gate_queries, merge_hits
from ragsys.assistant.grounded import UPLOAD_PROMPTS, grounded_answer
from ragsys.assistant.history import (
    HistoryPlan,
    Summary,
    Turn,
    plan_history,
    summary_message,
    transcript,
)
from ragsys.assistant.index import DenseIndex, Hit, IndexUnavailable, LockedEncoder
from ragsys.assistant.limits import AbuseGuard
from ragsys.assistant.llm import (
    BUSY_MESSAGE,
    QUOTA_MESSAGE,
    UNAVAILABLE_MESSAGE,
    BudgetWait,
    StreamFailure,
    StreamingChat,
    complete,
)
from ragsys.assistant.prompts import (
    SUMMARY_PROMPT,
    grounded_user_message,
    summary_request,
    system_prompt,
)
from ragsys.assistant.uploads import (
    UPLOAD_SOURCE,
    UploadError,
    UploadSession,
    UploadStore,
    ingest_pdf,
)
from ragsys.assistant.verify import verify_answer
from ragsys.chunking.base import Chunk
from ragsys.chunking.tokenizer import TokenCounter, get_tokenizer
from ragsys.llm.backends import BackendUnavailable
from ragsys.llm.client import ChatModel, Message, QuotaExceeded
from ragsys.profiles import GenerationPrompts, get_profile
from ragsys.retrieval.embed import Encoder, model_slug
from ragsys.retrieval.pipeline import RetrievedChunk

Event = dict[str, Any]  # {"event": name, "data": {...}}
ABSTENTION_MESSAGE = "Not found in these documents"


def event(name: str, **data: Any) -> Event:
    return {"event": name, "data": data}


class RequestError(Exception):
    """A request the assistant refuses before answering (HTTP status + message)."""

    def __init__(self, status: int, message: str, code: str = "bad_request") -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


@dataclass
class AssistantCollection:
    """A collection as the assistant serves it: its app index and texts."""

    name: str
    label: str
    processed_dir: Path
    emb_dir: Path
    profile: str | None = None
    description: str = ""
    domain: str = ""
    notice: str = ""
    examples: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    evaluated: bool = False
    generation: GenerationSettings | None = None
    abstention_message: str = ABSTENTION_MESSAGE

    def prompts(self) -> GenerationPrompts:
        return get_profile(self.profile).generation if self.profile else UPLOAD_PROMPTS


@dataclass
class ChatBackend:
    """A generator backend for chat: its streaming client and chat settings."""

    name: str
    llm: StreamingChat
    label: str = ""
    provider: str = ""
    privacy: str = ""
    max_tokens: int = 1500
    temperature: float = 0.3
    history_tokens: int = 3000
    hosted: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label or f"{self.name}: {self.llm.model}",
            "model": self.llm.model,
            "provider": self.provider,
            "privacy": self.privacy,
            "hosted": self.hosted,
        }


@dataclass
class ChatRequestData:
    turns: list[Turn]
    collection: str | None = None
    session_id: str | None = None
    grounded_only: bool = False
    backend: str | None = None
    summary: Summary | None = None


JsonLLMFactory = Callable[[str, AssistantCollection | None], ChatModel]


def passage_dict(p: Passage, source: str, title: str = "") -> dict[str, Any]:
    return {
        "label": p.label,
        "chunk_id": p.chunk_id,
        "doc_id": p.doc_id,
        "title": title,
        "heading_path": p.heading_path,
        "section_ids": list(p.section_ids),
        "pages": p.pages,
        "text": p.text,
        "source": source,
        "score": None if p.first_stage_score is None else round(p.first_stage_score, 4),
    }


def generator_failure(exc: Exception) -> StreamFailure:
    """The non-streaming client's errors (verify, grounded only) as user messages."""
    if isinstance(exc, (QuotaExceeded, BudgetWait)):
        return StreamFailure("busy", BUSY_MESSAGE)
    if isinstance(exc, openai.RateLimitError):
        return StreamFailure("busy", BUSY_MESSAGE)
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, BackendUnavailable)):
        return StreamFailure("unavailable", UNAVAILABLE_MESSAGE)
    if isinstance(exc, openai.APIStatusError) and exc.status_code == 429:
        return StreamFailure("quota", QUOTA_MESSAGE)
    return StreamFailure("error", "The model returned an unexpected error. Please try again.")


class ChatAssistant:
    def __init__(
        self,
        settings: AssistantSettings,
        collections: Sequence[AssistantCollection],
        encoder: Encoder,
        backends: Sequence[ChatBackend],
        default_backend: str,
        *,
        json_llm: JsonLLMFactory,
        backend_status: Callable[[str], str | None] | None = None,
        tokenizer: Callable[[], TokenCounter] = get_tokenizer,
        public: bool = False,
        uploads: UploadStore | None = None,
        guard: AbuseGuard | None = None,
    ) -> None:
        self.settings = settings
        self.collections = {c.name: c for c in collections}
        self.order = [c.name for c in collections]
        self.encoder = LockedEncoder(encoder)
        self.backends = {b.name: b for b in backends}
        if default_backend not in self.backends:
            raise ValueError(f"default backend {default_backend!r} is not configured")
        self.default_backend = default_backend
        self.json_llm = json_llm
        self.backend_status = backend_status or (lambda _b: None)
        self._tokenizer_factory = tokenizer
        self._tokenizer: TokenCounter | None = None
        self.public = public
        self.uploads = uploads or UploadStore(settings.uploads)
        self.guard = guard or AbuseGuard(settings.rate_limits)
        self.gate = GroundingGate(self.encoder, settings.gate)
        self._indexes: dict[str, DenseIndex] = {}
        self._index_lock = threading.Lock()
        self._streams: dict[str, threading.Event] = {}
        self._streams_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(settings.limits.max_concurrent_streams)
        self._upload_slot = threading.Semaphore(1)
        self._grounded_lock = threading.Lock()

    # ------------------------------------------------------------------ collections

    def collection(self, name: str | None) -> AssistantCollection | None:
        if name is None:
            return None
        coll = self.collections.get(name)
        if coll is None:
            raise RequestError(404, f"unknown collection {name!r}", "unknown_collection")
        return coll

    def index_files(self, coll: AssistantCollection) -> tuple[Path, Path]:
        chunks = coll.processed_dir / "chunks" / f"{self.settings.chunker}.jsonl"
        npy = (
            coll.emb_dir
            / model_slug(self.settings.embedding_model)
            / f"{self.settings.chunker}.npy"
        )
        return chunks, npy

    def available(self, coll: AssistantCollection) -> bool:
        return all(p.exists() for p in self.index_files(coll))

    def index(self, coll: AssistantCollection) -> DenseIndex:
        with self._index_lock:
            idx = self._indexes.get(coll.name)
            if idx is None:
                idx = DenseIndex.load(
                    coll.processed_dir,
                    coll.emb_dir,
                    self.settings.embedding_model,
                    self.settings.chunker,
                    coll.name,
                )
                self._indexes[coll.name] = idx
            return idx

    def warm_up(self, names: Sequence[str] | None = None) -> dict[str, float]:
        """Load indexes (and the encoder and tokenizer) now; returns seconds per index."""
        out: dict[str, float] = {}
        for name in names if names is not None else self.order:
            coll = self.collections[name]
            if not self.available(coll):
                continue
            t0 = time.perf_counter()
            self.index(coll)
            out[name] = time.perf_counter() - t0
        self.encoder.encode(["warm up"], is_query=True)
        return out

    def tokenizer(self) -> TokenCounter:
        if self._tokenizer is None:
            self._tokenizer = self._tokenizer_factory()
        return self._tokenizer

    def collections_info(self) -> list[dict[str, Any]]:
        out = []
        for name in self.order:
            c = self.collections[name]
            out.append(
                {
                    "id": c.name,
                    "label": c.label,
                    "available": self.available(c),
                    "evaluated": c.evaluated,
                    "description": c.description,
                    "notice": c.notice,
                    "examples": list(c.examples),
                    "sources": [
                        {
                            k: s.get(k, "")
                            for k in (
                                "title",
                                "authors",
                                "edition",
                                "licence",
                                "licence_url",
                                "source_url",
                            )
                        }
                        for s in c.sources
                    ],
                }
            )
        return out

    def backend(self, name: str | None) -> ChatBackend:
        b = self.backends.get(name or self.default_backend)
        if b is None:
            raise RequestError(400, f"unknown backend {name!r}", "unknown_backend")
        return b

    def health(self) -> dict[str, Any]:
        backends = []
        for b in self.backends.values():
            reason = self.backend_status(b.name)
            backends.append({**b.describe(), "available": reason is None, "reason": reason or ""})
        u = self.settings.uploads
        lim = self.settings.limits
        return {
            "status": "ok",
            "public": self.public,
            "default_backend": self.default_backend,
            "backends": backends,
            "limits": {
                "max_message_chars": lim.max_message_chars,
                "max_output_tokens": lim.max_output_tokens,
                "per_ip": [r.model_dump() for r in self.settings.rate_limits.per_ip],
            },
            "uploads": {
                "max_files": u.max_files,
                "max_mb": u.max_mb,
                "max_pages": u.max_pages,
                "ttl_s": u.ttl_s,
            },
            "general_examples": list(self.settings.general_examples),
        }

    # ------------------------------------------------------------------ streams

    def active_streams(self) -> list[str]:
        with self._streams_lock:
            return list(self._streams)

    def stop(self, stream_id: str) -> bool:
        with self._streams_lock:
            ev = self._streams.get(stream_id)
        if ev is None:
            return False
        ev.set()
        return True

    def _sources(
        self, coll: AssistantCollection | None, session_id: str | None
    ) -> list[tuple[DenseIndex, float]]:
        out: list[tuple[DenseIndex, float]] = []
        if coll is not None:
            if not self.available(coll):
                raise RequestError(
                    503, f"The {coll.label} documents are not available on this server.", "index"
                )
            out.append((self.index(coll), self.settings.gate.threshold(coll.name)))
        for idx in self.uploads.indexes(session_id):
            out.append((idx, self.settings.gate.upload_threshold))
        return out

    def _titles(self, session_id: str | None) -> dict[str, str]:
        titles = {}
        for d in self.uploads.documents(session_id):
            titles[str(d["doc_id"])] = str(d["name"])
        return titles

    def _summarizer(self, llm: StreamingChat) -> Callable[[str, list[Turn]], str]:
        hs = self.settings.history

        def summarize(previous: str, turns: list[Turn]) -> str:
            msgs: list[Message] = [
                {"role": "system", "content": SUMMARY_PROMPT},
                {
                    "role": "user",
                    "content": summary_request(previous, transcript(turns, hs.summary_input_chars)),
                },
            ]
            return complete(llm, msgs, hs.summary_max_tokens)

        return summarize

    def validate(self, req: ChatRequestData) -> None:
        lim = self.settings.limits
        if not req.turns or req.turns[-1].role != "user":
            raise RequestError(400, "the last message must be the user's")
        if len(req.turns) > lim.max_messages:
            raise RequestError(400, f"at most {lim.max_messages} messages per request")
        latest = req.turns[-1].content.strip()
        if not latest:
            raise RequestError(400, "please type a message")
        if len(latest) > lim.max_message_chars:
            raise RequestError(
                413, f"Messages are limited to {lim.max_message_chars} characters.", "too_long"
            )

    def chat_events(
        self, req: ChatRequestData, stop: threading.Event | None = None
    ) -> Iterator[Event]:
        """The events of one chat request (see the module docstring). Setting ``stop`` (or
        :meth:`stop` with the ``stream_id`` of the ``meta`` event) ends the reply after the
        current token."""
        stream_id = secrets.token_urlsafe(12)
        stop = stop or threading.Event()
        if not self._slots.acquire(blocking=False):
            yield event(
                "error",
                code="busy",
                message="Many people are chatting right now. Please try again in a minute.",
            )
            return
        with self._streams_lock:
            self._streams[stream_id] = stop
        try:
            backend = self.backend(req.backend)
            coll = self.collection(req.collection)
            yield event(
                "meta",
                stream_id=stream_id,
                backend=backend.name,
                model=backend.llm.model,
                collection=coll.name if coll else None,
                grounded_only=req.grounded_only,
                notice=coll.notice if coll else "",
            )
            if req.grounded_only:
                yield from self._grounded_events(req, backend, coll)
            else:
                yield from self._chat_events(req, backend, coll, stop)
        except RequestError as exc:
            yield event("error", code=exc.code, message=exc.message)
        except StreamFailure as exc:
            yield event("error", code=exc.code, message=exc.message, retry_after=exc.retry_after)
        except IndexUnavailable:
            yield event(
                "error",
                code="index",
                message="The documents of this collection are not available right now.",
            )
        finally:
            with self._streams_lock:
                self._streams.pop(stream_id, None)
            stop.set()
            self._slots.release()

    def _chat_events(
        self,
        req: ChatRequestData,
        backend: ChatBackend,
        coll: AssistantCollection | None,
        stop: threading.Event,
    ) -> Iterator[Event]:
        t0 = time.perf_counter()
        turns = req.turns
        latest = turns[-1].content.strip()
        previous = next((t.content for t in reversed(turns[:-1]) if t.role == "user"), None)
        decision: GateDecision = self.gate.decide(
            latest,
            previous,
            self._sources(coll, req.session_id),
            self.settings.context_budget(backend.name),
        )
        titles = self._titles(req.session_id)
        passages: list[dict[str, Any]] = []
        if decision.context is not None:
            for p in decision.context.passages:
                src = decision.source_of(p.chunk_id)
                passages.append(passage_dict(p, src, titles.get(p.doc_id, "")))
        yield event(
            "context",
            grounded=decision.grounded,
            passages=passages,
            checks=[c.to_dict() for c in decision.checks],
            gate_ms=round(decision.total_ms, 1),
        )
        if stop.is_set():
            return
        plan: HistoryPlan = plan_history(
            turns,
            backend.history_tokens,
            max_turns=self.settings.history.max_turns,
            summary=req.summary,
            summarize=self._summarizer(backend.llm),
            target_share=self.settings.history.summary_target_share,
        )
        messages: list[Message] = [
            {
                "role": "system",
                "content": system_prompt(
                    coll.notice if coll else "",
                    coll.label if coll else "",
                    documents=decision.grounded,
                ),
            }
        ]
        if plan.summary is not None:
            messages.append(summary_message(plan.summary))
        for t in plan.kept[:-1]:
            messages.append({"role": t.role, "content": t.content})
        user_text = plan.kept[-1].content.strip()
        if decision.grounded and decision.context is not None:
            user_text = grounded_user_message(user_text, decision.context)
        messages.append({"role": "user", "content": user_text})
        max_tokens = min(self.settings.limits.max_output_tokens, backend.max_tokens)
        t_req = time.perf_counter()
        stream = backend.llm.stream(
            messages, max_tokens=max_tokens, temperature=backend.temperature
        )
        parts: list[str] = []
        first: float | None = None
        stopped = False
        try:
            for tok in stream:
                if first is None:
                    first = time.perf_counter()
                parts.append(tok)
                yield event("token", t=tok)
                if stop.is_set():
                    stopped = True
                    break
        finally:
            stream.close()
        t_end = time.perf_counter()
        valid = decision.context.labels if decision.context is not None else []
        cres = clean_citations("".join(parts), valid)
        cited = [p for p in passages if p["label"] in cres.used]
        queued = float(getattr(stream, "queued_ms", 0.0) or 0.0)
        yield event(
            "done",
            answer=cres.text,
            citations=cres.used,
            removed_labels=cres.removed,
            passages=cited,
            grounded=cres.grounded,
            source_line=cres.source_line(),
            stopped=stopped,
            finish_reason=getattr(stream, "finish_reason", None),
            usage=dict(getattr(stream, "usage", {}) or {}),
            summary=plan.summary.to_dict() if plan.new_summary and plan.summary else None,
            history={
                "kept": len(plan.kept),
                "dropped": plan.dropped,
                "summarised": plan.summary is not None,
                "notes": plan.notes,
            },
            timings={
                "gate_ms": round(decision.total_ms, 1),
                "embed_ms": round(decision.embed_ms, 1),
                "queue_ms": round(queued, 1),
                "ttft_ms": None if first is None else round((first - t_req) * 1000.0 - queued, 1),
                "total_ms": round((t_end - t0) * 1000.0, 1),
                "stream_ms": round((t_end - t_req) * 1000.0 - queued, 1),
            },
            backend=backend.name,
            model=backend.llm.model,
        )

    # ------------------------------------------------------------------ grounded only

    def _search_fn(
        self, coll: AssistantCollection | None, session_id: str | None, previous: str | None
    ) -> Callable[[str], list[Hit]]:
        sources = [idx for idx, _t in self._sources(coll, session_id)]
        k = self.settings.gate.search_k * 2

        def search(query: str) -> list[Hit]:
            queries = gate_queries(query, previous, self.settings.gate.follow_up_max_words)
            groups, _e, _s = self.gate.search(queries, sources)
            return merge_hits(groups)[:k]

        return search

    def _grounded_events(
        self, req: ChatRequestData, backend: ChatBackend, coll: AssistantCollection | None
    ) -> Iterator[Event]:
        if coll is None and not self.uploads.indexes(req.session_id):
            raise RequestError(
                400,
                "Grounded only needs documents: choose a collection or upload a PDF.",
                "no_documents",
            )
        t0 = time.perf_counter()
        latest = req.turns[-1].content.strip()
        previous = next((t.content for t in reversed(req.turns[:-1]) if t.role == "user"), None)
        yield event("status", stage="grounded", message="Searching and checking the documents…")
        gen = coll.generation if coll else None
        budget = self.settings.context_budget(backend.name)
        if gen is not None:
            context = gen.context_for(backend.name)
        else:
            context = ContextSettings(
                budget_tokens=budget, revision_budget_tokens=int(budget * 1.5), min_passages=1
            )
        loop = gen.loop if gen else LoopSettings()
        max_out = int(gen.llm(backend.name).max_tokens) if gen else 1024
        window = gen.context_window_for(backend.name) if gen else None
        llm = self.json_llm(backend.name, coll)
        search = self._search_fn(coll, req.session_id, previous)
        try:
            with self._grounded_lock:
                result = grounded_answer(
                    latest,
                    search,
                    llm,
                    prompts=coll.prompts() if coll else UPLOAD_PROMPTS,
                    context=context,
                    loop=loop,
                    max_output_tokens=max_out,
                    context_window=window,
                    config_id=f"assistant|{coll.name if coll else 'uploads'}|dense|bge-small",
                )
        except (QuotaExceeded, BudgetWait, openai.APIError, BackendUnavailable) as exc:
            raise generator_failure(exc) from exc
        titles = self._titles(req.session_id)
        all_passages = {
            p.label: passage_dict(p, p.provenance, titles.get(p.doc_id, ""))
            for p in result.context.passages
        }
        answered = result.outcome == "answered"
        cited = [all_passages[lab] for lab in result.citations if lab in all_passages]
        inline = [lab for lab in labels_in(result.answer) if lab in all_passages]
        for lab in inline:
            if all(c["label"] != lab for c in cited):
                cited.append(all_passages[lab])
        message = coll.abstention_message if coll else ABSTENTION_MESSAGE
        yield event(
            "done",
            mode="grounded_only",
            outcome=result.outcome,
            answer=result.answer if answered else "",
            citations=[c["label"] for c in cited] if answered else [],
            passages=cited if answered else [],
            grounded=answered,
            source_line=source_line(len(cited)) if answered else message,
            abstention_message=message,
            reason=result.reason,
            detail=result.detail,
            closest=result.closest_sections() if not answered else [],
            sentences=[{**s, "status": "supported"} for s in result.sentences()]
            if answered
            else [],
            cost=result.cost(),
            stopped=False,
            timings={"total_ms": round((time.perf_counter() - t0) * 1000.0, 1)},
            backend=backend.name,
            model=str(getattr(llm, "model", "") or ""),
        )

    # ------------------------------------------------------------------ verify

    def resolve_chunk(
        self, chunk_id: str, coll: AssistantCollection | None, session_id: str | None
    ) -> tuple[Chunk, str] | None:
        if ":up_" in chunk_id:
            c = self.uploads.chunk(session_id, chunk_id)
            return (c, UPLOAD_SOURCE) if c is not None else None
        if coll is None or not self.available(coll):
            return None
        c2 = self.index(coll).by_id.get(chunk_id)
        return (c2, coll.name) if c2 is not None else None

    def verify(
        self,
        question: str,
        answer: str,
        refs: Sequence[tuple[str, str]],
        *,
        collection: str | None,
        session_id: str | None,
        backend: str | None,
    ) -> dict[str, Any]:
        coll = self.collection(collection)
        b = self.backend(backend)
        if not refs:
            raise RequestError(400, "This reply cites no documents, so there is nothing to verify.")
        passages: list[Passage] = []
        for i, (label, chunk_id) in enumerate(refs, start=1):
            found = self.resolve_chunk(chunk_id, coll, session_id)
            if found is None:
                raise RequestError(
                    409,
                    "The cited passages are no longer available (uploads are deleted after 2 hours "
                    "of inactivity or when the page is closed).",
                    "gone",
                )
            chunk, source = found
            item = RetrievedChunk(i, chunk, source, i, None)
            passages.append(passage_from(item, label, 1))
        llm = self.json_llm(b.name, coll)
        try:
            result = verify_answer(
                question, answer, passages, llm, coll.prompts() if coll else UPLOAD_PROMPTS
            )
        except (QuotaExceeded, BudgetWait, openai.APIError, BackendUnavailable) as exc:
            raise generator_failure(exc) from exc
        return {**result.to_api(), "backend": b.name}

    # ------------------------------------------------------------------ uploads

    def upload_events(self, data: bytes, name: str, session: UploadSession) -> Iterator[Event]:
        """Process one PDF (checked by the route, its slot reserved with
        ``uploads.reserve``) and report progress; the slot is released at the end."""
        progress: queue.Queue[Event | None] = queue.Queue()
        box: dict[str, Any] = {}

        def work() -> None:
            got = self._upload_slot.acquire(timeout=120)
            try:
                if not got:
                    raise UploadError(
                        503, "The server is busy processing other uploads. Please try again."
                    )

                def report(stage: str, done: int, total: int) -> None:
                    progress.put(event("progress", stage=stage, done=done, total=total))

                box["doc"] = ingest_pdf(
                    data, name, self.encoder, self.tokenizer(), self.settings.uploads, report
                )
            except UploadError as exc:
                box["error"] = exc
            except Exception:  # never echo parser internals (or file contents) to the client
                box["error"] = UploadError(422, "This PDF could not be processed.")
            finally:
                if got:
                    self._upload_slot.release()
                progress.put(None)

        worker = threading.Thread(target=work, name="upload", daemon=True)
        doc = None
        try:
            yield event("session", session_id=session.session_id)
            worker.start()
            while True:
                item = progress.get()
                if item is None:
                    break
                yield item
            worker.join()
            if "error" in box:
                err: UploadError = box["error"]
                yield event("error", code="upload", message=err.message, status=err.status)
                return
            doc = box["doc"]
            yield event(
                "done",
                session_id=session.session_id,
                document=doc.to_dict(),
                documents=[*self.uploads.documents(session.session_id), doc.to_dict()],
            )
        finally:
            if worker.is_alive():
                worker.join()
                doc = box.get("doc")
            self.uploads.release(session, doc)
