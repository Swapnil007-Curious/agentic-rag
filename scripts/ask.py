"""Answer one question with a configured system and show how the answer was produced.

Usage:
    python scripts/ask.py "Within how many days must a complaint be decided if no testing of goods is needed?" --system G4
    python scripts/ask.py "..." --system G5 --plain
    python scripts/ask.py --collection cs_ai "What problem does attention solve?" --system G2
A "selected:" system (cs_ai G2/G4) uses the dev-selected retrieval config; before the dev grid
and select_configs.py have run it falls back to the first built config of selection_fallback
and says which one it used.
Prints the outcome (answered, or abstained with the reason), the answer, every citation with its
section ids, pages and heading path, the step trace (retrieval, context, threshold, synthesis,
guards, critic, revision) with timings and LLM usage. Needs Ollama running (configs/llm.yaml)
unless --fake-llm is given (a deterministic offline stand-in used by the tests).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from typing import Any

from ragsys.agents.orchestrator import AnswerResult
from ragsys.agents.runtime import build_engine, llm_unreachable_message, load_settings, make_llm
from ragsys.collections import add_collection_arguments, collection_from_args, config_path
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("question", nargs="+")
    ap.add_argument("--system", default="G4", help="G1..G6 (default G4)")
    ap.add_argument("--plain", action="store_true", help="plain-language answer (still cited)")
    ap.add_argument("--config", default=None, help="default: the collection's generation config")
    add_collection_arguments(ap)
    ap.add_argument("--retrieval-config", default=None, help="override retrieval_config")
    ap.add_argument("--json", action="store_true", help="print the full record as JSON")
    ap.add_argument("--fake-llm", action="store_true", help="offline deterministic stand-in")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (default 4)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    coll = collection_from_args(args)
    settings, rsettings = load_settings(
        config_path(args.config, coll.generation_config), args.retrieval_config
    )
    if args.system not in settings.systems:
        emit(f"unknown system {args.system!r}; choose one of {', '.join(settings.systems)}")
        return 2
    set_seed(settings.seed)
    set_torch_threads(args.threads)
    llm = make_llm(settings, fake=args.fake_llm)
    problem = llm_unreachable_message(llm)
    if problem:
        emit(f"ERROR: {problem}")
        return 1
    t0 = time.perf_counter()
    engine = build_engine(settings, rsettings, llm, allow_selection_fallback=True)
    engine.orchestrator(args.system)  # load the retriever (and warm it up) before timing
    load_s = time.perf_counter() - t0
    result = engine.ask(" ".join(args.question), args.system, plain=args.plain)
    if args.json:
        emit(json.dumps(result.to_api(), indent=2, ensure_ascii=False))
    else:
        print_result(result, settings.system(args.system).description)
    engine.close()
    emit(f"(models loaded in {load_s:.1f} s; peak RSS {fmt_mb(peak_rss_bytes())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
