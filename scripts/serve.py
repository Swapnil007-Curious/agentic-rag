"""Start the web app: the chat assistant at "/" and the grounded research page at "/research".

Usage:
    python scripts/serve.py [--collection NAME] [--backend local|hosted] [--port 8000]
                            [--fake-llm] [--warm-research] [--no-warm] [--tunnel]

Locally the server binds to 127.0.0.1 (http://127.0.0.1:8000). It binds elsewhere only when the
HOST environment variable is set (HOST=0.0.0.0 in the hosted demo's container), which also
switches on public mode: the research routes are off, only hosted backends are offered and
the rate limits of configs/assistant.yaml protect the API key (see
src/ragsys/assistant/hosting.py for ALLOWED_HOSTS, FRAME_ANCESTORS, TRUST_PROXY, PORT).

--tunnel is for sharing the app through a tunnel client on this computer (e.g. `cloudflared
tunnel --url http://127.0.0.1:8000`): the server still binds to 127.0.0.1 but runs in public
mode, accepts any host name and takes each visitor's address for the rate limits from the
tunnel's CF-Connecting-IP header (else X-Forwarded-For). A banner warns that the app is
publicly reachable. Without --tunnel nothing changes.

The assistant (src/ragsys/assistant/) streams one LLM call per message and grounds it
automatically in the selected collection's app index and uploaded PDFs; "Verify" runs the
guards and the critic on demand and "Grounded only" runs the grounded pipeline. The research
page keeps the original modes (Grounded, Hybrid, Chat & Code) with the evaluated systems.

The default backend is hosted when LLM_API_KEY is set and the endpoint answers, else local
(--backend overrides); every answer names the model that produced it. Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import uvicorn

from ragsys.agents.config import GenerationSettings
from ragsys.agents.engine import AnswerEngine
from ragsys.agents.runtime import (
    backend_problem,
    build_engine,
    chat_history_tokens,
    load_settings,
    make_chat_llm,
    make_llm,
    serving_retrieval,
)
from ragsys.app.server import CollectionEntry, create_multi_app, example_questions, read_sources
from ragsys.assistant import setup as assistant_setup
from ragsys.assistant.hosting import Binding, resolve_binding
from ragsys.collections import Collection as Coll
from ragsys.collections import add_collection_arguments, load_collections
from ragsys.config import resolve_path
from ragsys.llm.client import ChatModel
from ragsys.pipeline_plan import render_summary
from ragsys.retrieval.config import RetrievalSettings
from ragsys.utils.console import emit
from ragsys.utils.memory import set_torch_threads
from ragsys.utils.seed import set_seed


def _factory(
    settings: GenerationSettings, rsettings: RetrievalSettings, backend: str, fake: bool
) -> Callable[[], AnswerEngine]:
    def build() -> AnswerEngine:
        llm = make_llm(settings, fake=fake, backend=backend)
        return build_engine(
            settings, rsettings, llm, allow_selection_fallback=True, backend=backend, fake=fake
        )

    return build


def _summary(coll: Coll) -> Callable[[], str | None]:
    def text() -> str | None:
        path = resolve_path(coll.summary_file)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return render_summary(coll)  # generated from the tables on request

    return text


def sources_csv(coll: Coll, settings: GenerationSettings, served_app: bool) -> Path:
    """The SOURCES.csv of the documents the UI serves: the app index's own file (cs_ai: the
    evaluation documents plus the Python Tutorial and Think Python) once that index is built,
    else the collection's (evaluation corpus) file."""
    return assistant_setup.sources_csv(coll, settings, served_app)


def make_entry(coll: Coll, backend: str, fake_llm: bool) -> CollectionEntry:
    settings, _rs = load_settings(resolve_path(coll.generation_config))
    rsettings, index_note = serving_retrieval(coll)
    chunks = resolve_path(rsettings.paths.processed_dir) / "chunks"
    available = chunks.exists() and any(chunks.glob("*.jsonl"))
    runner = (
        "scripts/run_collection_pipeline.py" if coll.evaluated else "scripts/run_app_collections.py"
    )
    note = "" if available else f"not built yet: run python {runner} --collection {coll.name}"
    ui = coll.spec.ui
    examples = list(ui.examples) or example_questions(
        resolve_path(coll.questions_file), ui.n_examples
    )

    def chat_llm(name: str, settings: GenerationSettings = settings) -> ChatModel:
        return make_chat_llm(settings, name, fake=fake_llm)

    return CollectionEntry(
        name=coll.name,
        settings=settings,
        engine_factory=_factory(settings, rsettings, backend, fake_llm),
        label=ui.label,
        texts={
            "eyebrow": ui.eyebrow,
            "heading": ui.heading,
            "lede": ui.lede,
            "placeholder": ui.placeholder,
            "disclaimer": ui.disclaimer,
            "abstention_message": ui.abstention_message,
            "safety_note": ui.safety_note,
            "domain": ui.domain,
            "code_assistant": "true" if ui.code_assistant else "",
        },
        sources=read_sources(sources_csv(coll, settings, served_app=not index_note)),
        examples=examples,
        feedback_path=resolve_path(settings.ui.feedback_file),
        available=available,
        note=note,
        evaluated=coll.evaluated,
        summary_text=_summary(coll) if coll.evaluated else None,
        index_note=index_note,
        chat_llm_factory=chat_llm,
        history_tokens=lambda name, settings=settings: chat_history_tokens(settings, name),
    )


def tunnel_banner(binding: Binding) -> list[str]:
    """The warning printed at start-up in tunnel mode."""
    rule = "=" * 78
    target = f"http://{binding.host}:{binding.port}"
    return [
        rule,
        "WARNING: TUNNEL MODE - this app is PUBLICLY REACHABLE through your tunnel.",
        "Anyone with the tunnel URL can chat, upload PDFs and spend your API quota.",
        f"Bound to {binding.host}:{binding.port}; point the tunnel at {target}",
        f"(e.g. cloudflared tunnel --url {target}). Public mode is on: per-visitor rate",
        "limits (CF-Connecting-IP, else X-Forwarded-For), message and reply caps, no message",
        "logging, /research off, hosted backends only, any host name accepted.",
        "Stop the tunnel (or this server, Ctrl+C) to end public access.",
        rule,
    ]


class BackendStatus:
    """Availability of each backend, re-checked at most once a minute (a fake is available)."""

    def __init__(self, settings: GenerationSettings, fake: bool, ttl_s: float = 60.0) -> None:
        self.settings = settings
        self.fake = fake
        self.ttl_s = ttl_s
        self._cache: dict[str, tuple[float, str | None]] = {}
        self._lock = threading.Lock()

    def __call__(self, name: str) -> str | None:
        if self.fake:
            return None
        with self._lock:
            hit = self._cache.get(name)
            if hit is not None and time.monotonic() - hit[0] < self.ttl_s:
                return hit[1]
        reason = backend_problem(self.settings, name)
        with self._lock:
            self._cache[name] = (time.monotonic(), reason)
        return reason


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", type=int, default=None, help="default: PORT, else ui.port (8000)")
    ap.add_argument("--backend", default=None, help="default generator (default: auto)")
    ap.add_argument("--no-warm", action="store_true", help="load indexes on the first message")
    ap.add_argument(
        "--warm-research",
        action="store_true",
        help="also load the research page's retriever at start-up",
    )
    ap.add_argument("--fake-llm", action="store_true", help="offline deterministic stand-in")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--tunnel",
        action="store_true",
        help="public mode behind a tunnel to 127.0.0.1 (any host name, client IP from the tunnel)",
    )
    add_collection_arguments(ap)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    reg = load_collections(args.collections_file)
    set_seed(42)
    set_torch_threads(args.threads)
    first_coll = Coll(reg.ui_default or reg.default, reg.collections[reg.ui_default or reg.default])
    base_settings, _ = load_settings(resolve_path(first_coll.generation_config))
    binding = resolve_binding(os.environ, args.port or base_settings.ui.port, tunnel=args.tunnel)
    if args.port:
        binding.port = args.port
    if args.tunnel and (os.environ.get("HOST") or "").strip():
        emit("NOTE: HOST is ignored in tunnel mode; binding to 127.0.0.1")
    backends = base_settings.backends()
    # the public demo offers hosted backends only (no local model behind a public address)
    offered = [n for n in backends.names if backends.spec(n).requires_key or not binding.public]
    if not offered:
        emit("public mode needs a hosted backend (requires_key: true) in configs/llm.yaml")
        return 2
    status = BackendStatus(base_settings, args.fake_llm)
    if args.backend:
        if args.backend not in offered:
            emit(f"unknown backend {args.backend!r}; offered: {', '.join(offered)}")
            return 2
        default_backend = args.backend
    else:  # hosted when its key is set and it answers, else the default (local) backend
        hosted = [n for n in offered if backends.spec(n).requires_key]
        default_backend = next(
            (n for n in hosted if status(n) is None),
            backends.default_backend if backends.default_backend in offered else offered[0],
        )
    for name in offered:
        reason = status(name)
        emit(
            f"backend {name}: {backends.spec(name).label} - {'ready' if reason is None else reason}"
        )
    emit(f"default backend: {default_backend}")

    entries: list[CollectionEntry] = []
    for name in reg.ordered():
        entry = make_entry(Coll(name, reg.collections[name]), default_backend, args.fake_llm)
        entries.append(entry)
        state = "ready" if entry.available else entry.note
        badge = "evaluated" if entry.evaluated else "not evaluated"
        emit(
            f"collection {name}: {entry.label} ({len(entry.sources)} documents, {badge}) - {state}"
        )
        if entry.index_note:
            emit(f"    NOTE: {entry.index_note}")
    wanted = args.collection or reg.ui_default or reg.default
    if wanted not in reg.collections:
        emit(f"unknown collection {wanted!r}; known: {', '.join(reg.collections)}")
        return 2
    first = next((e for e in entries if e.name == wanted), entries[0])
    if not first.available:
        fallback = next((e for e in entries if e.available), None)
        if fallback is not None:
            emit(f"WARNING: {first.name} is not built yet; showing {fallback.name} first")
            first = fallback
    problem = status(default_backend)
    if problem:
        emit(f"WARNING: {problem}; questions will fail until it is reachable")

    assistant = assistant_setup.build_assistant(
        args.collections_file,
        backends,
        offered,
        default_backend,
        fake_llm=args.fake_llm,
        public=binding.public,
        backend_status=status,
    )
    if not args.no_warm:
        emit("loading the assistant's indexes and the bge-small encoder ...")
        names = None if binding.public else [first.name]
        for name, secs in assistant.warm_up(names).items():
            emit(
                f"    {name}: {len(assistant.index(assistant.collections[name]))} chunks ({secs:.1f}s)"
            )
    if args.warm_research and not binding.public and first.available:
        system = first.settings.ui.default_system
        emit(f"loading the research retriever of {first.name} / {system} ...")
        engine = first.engine()
        engine.orchestrator(system)
        for w in engine.warnings_for(system):
            emit(f"NOTE: {w}")
    app = create_multi_app(
        entries,
        first.name,
        allowed_hosts=binding.allowed_hosts,
        backends=[backends.spec(n).describe(n) for n in offered],
        default_backend=default_backend,
        backend_status=status,
        assistant=assistant,
        public=binding.public,
        frame_ancestors=binding.frame_ancestors,
        trust_forwarded=binding.trust_forwarded,
        forwarded_hops=binding.forwarded_hops,
        client_ip_header=binding.client_ip_header,
    )
    if binding.tunnel:
        for line in tunnel_banner(binding):
            emit(line)
    mode = "local only"
    if binding.public:
        mode = "tunnel, public mode" if binding.tunnel else "public mode"
    emit(f"serving on http://{binding.host}:{binding.port}  ({mode}; Ctrl+C to stop)")
    # public modes: no access log either (it would record visitors' paths and addresses)
    run_kwargs: dict[str, object] = {"access_log": False} if binding.public else {}
    try:
        uvicorn.run(app, host=binding.host, port=binding.port, log_level="warning", **run_kwargs)
    finally:
        for e in entries:
            if e.built:
                e.engine().close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
