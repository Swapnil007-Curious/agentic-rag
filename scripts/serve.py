"""Start the local web UI on http://127.0.0.1:8000 (loopback only).

Usage:
    python scripts/serve.py [--collection NAME] [--port 8000] [--no-warm]
Serves every collection of configs/collections.yaml with a collection selector (shown first:
--collection, else ui_default = cs_ai). Per collection: its documents with titles, authors,
licence and source links (SOURCES.csv), five example questions from the dev split as clickable
chips, its disclaimer, its systems (consumer_law: G2, G4, G5, G6; cs_ai: G2, G4) and the
abstention message "Not found in these documents". Answers have clickable citation chips that
open the cited passage (heading path, section id, pages, text with the sentence's key terms
highlighted), the three closest sections when the system abstains, a collapsible step trace
with timings, a plain-language toggle and thumbs up/down stored in the collection's feedback
file. A collection whose chunks are not built yet is listed but disabled. The server always
binds to 127.0.0.1; it is meant for one local user. Needs Ollama running (configs/llm.yaml).
Stop it with Ctrl+C.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import uvicorn

from ragsys.agents.config import GenerationSettings
from ragsys.agents.engine import AnswerEngine
from ragsys.agents.runtime import build_engine, llm_unreachable_message, load_settings, make_llm
from ragsys.app.server import CollectionEntry, create_multi_app, example_questions, read_sources
from ragsys.collections import Collection as Coll
from ragsys.collections import add_collection_arguments, load_collections
from ragsys.config import resolve_path
from ragsys.llm.client import ChatModel
from ragsys.retrieval.config import RetrievalSettings
from ragsys.utils.console import emit
from ragsys.utils.memory import set_torch_threads
from ragsys.utils.seed import set_seed

HOST = "127.0.0.1"  # never exposed beyond this machine


def _factory(
    settings: GenerationSettings, rsettings: RetrievalSettings, llm: ChatModel
) -> Callable[[], AnswerEngine]:
    return lambda: build_engine(settings, rsettings, llm, allow_selection_fallback=True)


def make_entry(coll: Coll, fake_llm: bool) -> tuple[CollectionEntry, ChatModel]:
    settings, rsettings = load_settings(resolve_path(coll.generation_config))
    llm = make_llm(settings, fake=fake_llm)
    chunks = resolve_path(rsettings.paths.processed_dir) / "chunks"
    available = chunks.exists() and any(chunks.glob("*.jsonl"))
    note = (
        ""
        if available
        else f"not built yet: run python scripts/run_collection_pipeline.py --collection {coll.name}"
    )
    ui = coll.spec.ui
    entry = CollectionEntry(
        name=coll.name,
        settings=settings,
        engine_factory=_factory(settings, rsettings, llm),
        label=ui.label,
        texts={
            "eyebrow": ui.eyebrow,
            "heading": ui.heading,
            "lede": ui.lede,
            "placeholder": ui.placeholder,
            "disclaimer": ui.disclaimer,
            "abstention_message": ui.abstention_message,
        },
        sources=read_sources(resolve_path(settings.ui.sources_csv)),
        examples=example_questions(resolve_path(rsettings.paths.questions), ui.n_examples),
        feedback_path=resolve_path(settings.ui.feedback_file),
        available=available,
        note=note,
    )
    return entry, llm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", type=int, default=None, help="default: ui.port (8000)")
    ap.add_argument("--no-warm", action="store_true", help="load models on the first question")
    ap.add_argument("--fake-llm", action="store_true", help="offline deterministic stand-in")
    ap.add_argument("--threads", type=int, default=4)
    add_collection_arguments(ap)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    reg = load_collections(args.collections_file)
    set_seed(42)
    set_torch_threads(args.threads)
    entries: list[CollectionEntry] = []
    llms: dict[str, ChatModel] = {}
    for name, spec in reg.collections.items():
        entry, llm = make_entry(Coll(name, spec), args.fake_llm)
        entries.append(entry)
        llms[name] = llm
        state = "ready" if entry.available else entry.note
        emit(f"collection {name}: {entry.label} ({len(entry.sources)} documents) - {state}")
    wanted = args.collection or reg.ui_default or reg.default
    if wanted not in llms:
        emit(f"unknown collection {wanted!r}; known: {', '.join(llms)}")
        return 2
    first = next((e for e in entries if e.name == wanted), entries[0])
    if not first.available:
        fallback = next((e for e in entries if e.available), None)
        if fallback is not None:
            emit(f"WARNING: {first.name} is not built yet; showing {fallback.name} first")
            first = fallback
    problem = llm_unreachable_message(llms[first.name])
    if problem:
        emit(f"WARNING: {problem}; questions will fail until it is reachable")
    if not args.no_warm and first.available:
        system = first.settings.ui.default_system
        emit(f"loading the retriever of {first.name} / {system} ...")
        engine = first.engine()
        engine.orchestrator(system)
        for w in engine.warnings_for(system):
            emit(f"NOTE: {w}")
    port = args.port or first.settings.ui.port
    app = create_multi_app(
        entries, first.name, health=lambda: llm_unreachable_message(llms[first.name])
    )
    emit(f"serving on http://{HOST}:{port}  (Ctrl+C to stop)")
    try:
        uvicorn.run(app, host=HOST, port=port, log_level="warning")
    finally:
        for e in entries:
            if e.built:
                e.engine().close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
