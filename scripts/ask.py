"""Answer one question in grounded, hybrid or chat mode and show how the answer was produced.

Usage:
    python scripts/ask.py "Within how many days must a complaint be decided if no testing of goods is needed?" --system G4
    python scripts/ask.py "..." --system G5 --plain
    python scripts/ask.py --collection cs_ai "What problem does attention solve?" --system G2
    python scripts/ask.py --collection indian_law "What does Article 21 of the Constitution protect?" --mode grounded --backend hosted
    python scripts/ask.py --collection cs_ai "Write a Python function that ..." --mode chat --backend hosted
Modes: grounded (default; the synthesizer + guards + critic pipeline of the chosen system,
answers only from passages, otherwise "Not found in these documents" with the three closest
sections), hybrid (grounded first; after an abstention, a separate general-knowledge answer under
a banner) and chat (Markdown with fenced code; --use-documents adds passages to cite as [C1];
the coding assistant is used for cs_ai or with --code). Code is never executed.
--backend local|hosted picks the generator (default: configs/llm.yaml's default backend, local);
the hosted backend needs LLM_API_KEY. Nothing falls back to another backend: an unavailable one
stops with the reason. The collection's app index is used (cs_ai: its documents plus the Python
Tutorial and Think Python, once built; before that, the evaluated documents, with a note);
before a collection's embeddings are built, retrieval falls back to BM25 and says so.
Prints the outcome (answered, or abstained with the reason), the answer, every citation with its
section ids, pages and heading path, the step trace with timings and LLM usage, and which model
produced the answer. --fake-llm is a deterministic offline stand-in used by the tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from typing import Any

import openai

from ragsys.agents.modes import Assistant, ChatTurn, ModeResult, ModeSettings
from ragsys.agents.orchestrator import AnswerResult
from ragsys.agents.runtime import (
    build_engine,
    chat_history_tokens,
    llm_unreachable_message,
    load_settings,
    make_chat_llm,
    make_llm,
    serving_retrieval,
)
from ragsys.collections import add_collection_arguments, collection_from_args, config_path
from ragsys.config import resolve_path
from ragsys.llm.backends import BackendUnavailable
from ragsys.llm.client import QuotaExceeded
from ragsys.retrieval.config import load_retrieval_settings
from ragsys.utils.console import emit
from ragsys.utils.memory import fmt_mb, peak_rss_bytes, set_torch_threads
from ragsys.utils.seed import set_seed


def _pages(pages: str) -> str:
    return f"pages {pages}" if "-" in pages else f"page {pages}"


def _fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def step_summary(step: dict[str, Any]) -> str:
    d = step["detail"]
    name = step["name"]
    if name == "retrieval":
        score = d.get("abstention_score")
        return (
            f"{d.get('n_ranked')} ranked chunks; abstention score {score:.4f}"
            if score is not None
            else ""
        )
    if name in ("context", "revision_context"):
        added = f"; added {', '.join(d['added'])}" if d.get("added") else ""
        return f"{d['n_passages']} passages, {d['used_tokens']}/{d['budget_tokens']} tokens{added}"
    if name == "threshold":
        return f"score {d.get('score')} vs threshold {d.get('threshold')}"
    if name == "guards":
        return "; ".join(d.get("reasons", [])) or "all guards passed"
    if name == "critic":
        fb = d.get("feedback") or ""
        return (fb[:120] + ("..." if len(fb) > 120 else "")) or "-"
    if name == "synthesis":
        return d.get("error") or d.get("note") or ("revision" if d.get("revision") else "")
    return ""


def print_result(result: AnswerResult, description: str) -> None:
    rec = result.to_api()
    emit(f"system:    {result.system} ({description})")
    emit(f"retrieval: {rec['retrieval']['config_id']}")
    if result.backend:
        emit(f"generator: {result.backend} ({result.model})")
    emit(f"question:  {result.question}")
    for w in result.warnings:
        emit(f"WARNING:   {w}")
    emit("")
    if result.outcome == "answered":
        emit("outcome:   ANSWERED")
        emit("answer:")
        for line in textwrap.wrap(result.answer, width=96):
            emit(f"  {line}")
        emit("citations:")
        cited = {lab for s in rec["sentences"] for lab in s["citations"]} | set(result.citations)
        for lab, p in rec["passages"].items():
            if lab in cited:
                emit(
                    f"  [{lab}] {', '.join(p['section_ids'])} ({_pages(p['pages'])})  {p['heading_path']}"
                )
        missing = sorted(cited - set(rec["passages"]))
        if missing:
            emit(f"  cited but not in the context: {', '.join(missing)}")
    else:
        emit(f"outcome:   ABSTAINED ({result.reason})")
        emit("answer:    Not found in these documents.")
        emit(f"why:       {result.detail}")
        emit("closest sections:")
        for c in rec["closest_sections"]:
            emit(f"  {', '.join(c['section_ids'])} ({_pages(c['pages'])})  {c['heading_path']}")
    emit("")
    emit("step trace:")
    for s in rec["steps"]:
        calls = s["llm_calls"]
        usage = ""
        if calls:
            pt = sum(c["prompt_tokens"] for c in calls)
            ct = sum(c["completion_tokens"] for c in calls)
            cached = sum(1 for c in calls if c["cached"])
            usage = f" [{len(calls)} LLM call(s), {cached} cached, {pt}+{ct} tokens]"
        emit(
            f"  {s['name']:17s} r{s['round']}  {_fmt_ms(s['ms']):>8s}  {s['status']:11s}"
            f" {step_summary(s)}{usage}"
        )
    c = rec["cost"]
    emit("")
    emit(
        f"timings: end-to-end {_fmt_ms(c['e2e_ms'])} (wall {_fmt_ms(c['measured_ms'])}); "
        f"LLM {_fmt_ms(c['llm_ms'])} in {c['llm_calls']} call(s) "
        f"({c['synth_calls']} synthesizer, {c['critic_calls']} critic, {c['repair_calls']} repair, "
        f"{c['cached_calls']} cached); tokens {c['prompt_tokens']} prompt / "
        f"{c['completion_tokens']} completion"
    )


def print_general(general: dict[str, Any]) -> None:
    emit("")
    emit(f"=== {general['banner']}")
    emit(general["answer"])
    emit(f"(general-knowledge answer by {general['backend']} / {general['model']})")
    emit("related reading:")
    for c in general.get("related_reading", []):
        emit(f"  {', '.join(c['section_ids'])} ({_pages(c['pages'])})  {c['heading_path']}")


def print_chat(res: ModeResult) -> None:
    chat = res.chat or {}
    emit(f"mode:      chat{' (coding assistant)' if chat.get('code') else ''}")
    emit(f"generator: {res.backend} ({res.model})")
    if chat.get("use_documents"):
        emit(f"documents: {chat.get('retrieval')} ({len(chat.get('passages', {}))} passages)")
    for w in res.warnings:
        emit(f"WARNING:   {w}")
    if res.safety_note:
        emit(f"note:      {res.safety_note}")
    emit("")
    emit(chat.get("answer", ""))
    if chat.get("citations"):
        emit("")
        emit("cited passages:")
        for lab in chat["citations"]:
            p = chat["passages"][lab]
            emit(
                f"  [{lab}] {', '.join(p['section_ids'])} ({_pages(p['pages'])})  {p['heading_path']}"
            )
    usage = chat.get("usage") or {}
    emit("")
    emit(
        f"timings: {_fmt_ms(res.ms)} end-to-end; LLM {_fmt_ms(chat.get('ms', 0.0))}"
        f"{' (cached)' if chat.get('cached') else ''}; tokens "
        f"{usage.get('prompt_tokens', '?')} prompt / {usage.get('completion_tokens', '?')} completion"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("question", nargs="+")
    ap.add_argument("--system", default=None, help="G1..G6 (default G4)")
    ap.add_argument("--mode", choices=["grounded", "hybrid", "chat"], default="grounded")
    ap.add_argument("--backend", default=None, help="local | hosted (default: llm.yaml's default)")
    ap.add_argument("--use-documents", action="store_true", help="chat: cite retrieved passages")
    code = ap.add_mutually_exclusive_group()
    code.add_argument(
        "--code", dest="code", action="store_true", default=None, help="chat: coding assistant"
    )
    code.add_argument("--no-code", dest="code", action="store_false", help="chat: plain assistant")
    ap.add_argument("--plain", action="store_true", help="plain-language answer (still cited)")
    ap.add_argument("--config", default=None, help="default: the collection's generation config")
    add_collection_arguments(ap)
    ap.add_argument("--retrieval-config", default=None, help="override the served index")
    ap.add_argument("--json", action="store_true", help="print the full record as JSON")
    ap.add_argument("--fake-llm", action="store_true", help="offline deterministic stand-in")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (default 4)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    coll = collection_from_args(args)
    settings, config_rs = load_settings(config_path(args.config, coll.generation_config))
    index_note = ""
    if args.retrieval_config:
        rsettings = load_retrieval_settings(resolve_path_arg(args.retrieval_config))
    elif args.config:  # an explicit generation config brings its own retrieval settings
        rsettings = config_rs
    else:
        rsettings, index_note = serving_retrieval(coll)
    system = args.system or ("G4" if "G4" in settings.systems else settings.ui.default_system)
    if system not in settings.systems:
        emit(f"unknown system {system!r}; choose one of {', '.join(settings.systems)}")
        return 2
    backend = args.backend or settings.default_backend
    if backend not in settings.backends().names:
        emit(f"unknown backend {backend!r}; configured: {', '.join(settings.backends().names)}")
        return 2
    set_seed(settings.seed)
    set_torch_threads(args.threads)
    llm = make_llm(settings, fake=args.fake_llm, backend=backend)
    problem = llm_unreachable_message(llm)
    if problem:
        emit(f"ERROR: {problem}")
        return 1
    if index_note:
        emit(f"NOTE: {index_note}")
    t0 = time.perf_counter()
    engine = build_engine(
        settings, rsettings, llm, allow_selection_fallback=True, backend=backend, fake=args.fake_llm
    )
    ui = coll.spec.ui
    assistant = Assistant(
        engine,
        ModeSettings(
            collection=coll.name,
            domain=ui.domain,
            safety_note=ui.safety_note,
            code_assistant=ui.code_assistant,
            abstention_message=ui.abstention_message,
        ),
        lambda name: make_chat_llm(settings, name, fake=args.fake_llm),
        lambda name: chat_history_tokens(settings, name),
    )
    text = " ".join(args.question)
    try:
        return _run(args, assistant, engine, settings, system, backend, text, t0)
    except (QuotaExceeded, BackendUnavailable, openai.APIError) as exc:
        engine.close()
        emit(f"ERROR: the {backend} generator failed ({type(exc).__name__}): {exc}")
        emit("No other model was used; try again later or pass another --backend.")
        return 1


def _run(
    args: argparse.Namespace,
    assistant: Assistant,
    engine: Any,
    settings: Any,
    system: str,
    backend: str,
    text: str,
    t0: float,
) -> int:
    if args.mode == "chat":
        if args.use_documents:
            engine.orchestrator(system, backend)  # load the retriever before timing
        load_s = time.perf_counter() - t0
        res = assistant.chat(
            [ChatTurn("user", text)],
            system=system,
            backend=backend,
            use_documents=args.use_documents,
            code=args.code,
        )
        if args.json:
            emit(json.dumps(res.to_api(), indent=2, ensure_ascii=False))
        else:
            print_chat(res)
    else:
        engine.orchestrator(system, backend)  # load the retriever (and warm it up) before timing
        load_s = time.perf_counter() - t0
        res = assistant.ask(text, system, args.mode, backend=backend, plain=args.plain)
        if args.json:
            emit(json.dumps(res.to_api(), indent=2, ensure_ascii=False))
        else:
            assert res.grounded is not None
            print_result(res.grounded, settings.system(system).description)
            if res.general:
                print_general(res.general)
            if res.safety_note:
                emit(f"note: {res.safety_note}")
    engine.close()
    emit(f"(models loaded in {load_s:.1f} s; peak RSS {fmt_mb(peak_rss_bytes())})")
    return 0


def resolve_path_arg(p: str) -> Any:
    from pathlib import Path

    return resolve_path(Path(p))


if __name__ == "__main__":
    raise SystemExit(main())
