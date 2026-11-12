"""Measure the chat assistant's speed: time to first token, full answer time and the grounding
gate (embedding + search), on 10 prompts over the collections.

Usage:
    python scripts/bench_chat.py [--backend hosted] [--fake-llm] [--gate-runs 50]

Each prompt runs through the assistant exactly as the web app runs it (gate, history, one
streamed call), in this process. Reported per prompt: whether it was grounded, the gate time,
TTFT (from sending the request to the first token; time spent waiting for the client-side
per-minute token budget is excluded and shown separately), the total time, completion tokens
and a projection of the time a 300-token reply needs (TTFT + 300 tokens at the measured
streaming rate). The gate is also timed on its own (--gate-runs messages over all collections).
Requests are paced to the backend's tokens-per-minute budget, so a run on a hosted free tier
takes a few minutes and uses roughly 30k tokens. Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any

from ragsys.agents.config import load_generation_settings
from ragsys.assistant.config import load_assistant_settings
from ragsys.assistant.history import Turn
from ragsys.assistant.service import ChatRequestData
from ragsys.assistant.setup import build_assistant
from ragsys.collections import DEFAULT_COLLECTIONS_FILE, Collection, load_collections
from ragsys.config import resolve_path
from ragsys.llm.backends import key_present
from ragsys.utils.memory import set_torch_threads

PROMPTS: list[tuple[str | None, str]] = [
    ("cs_ai", "Explain big-O notation with two short Python examples. Answer in about 200 words."),
    ("cs_ai", "What does git rebase do, and when should I avoid it? Answer in about 200 words."),
    ("cs_ai", "How does the attention mechanism work in sequence models? About 200 words."),
    (
        None,
        "Write a Python function that checks whether a string is a palindrome and explain it briefly (about 200 words).",
    ),
    ("indian_law", "What does Article 21 of the Constitution protect? Answer in about 200 words."),
    (
        "indian_law",
        "Who is liable to be registered under the CGST Act, 2017? Answer in about 200 words.",
    ),
    ("human_biology", "How does a muscle fiber contract and relax? Answer in about 200 words."),
    (
        "physics_chemistry",
        "Derive the kinematic equations for constant acceleration (about 200 words, LaTeX).",
    ),
    ("physics_chemistry", "What does Archimedes' principle state? Answer in about 200 words."),
    ("consumer_law", "Within how many days must a consumer complaint be decided? About 200 words."),
]
GATE_PROMPTS = [
    "hello, how are you?",
    "What is the time complexity of quicksort?",
    "What is the punishment for theft?",
    "How does the heart pump blood?",
    "What is the ideal gas law?",
    "and why is that?",
]
TARGET_TTFT_MS = 1500.0
TARGET_300_MS = 5000.0
TARGET_GATE_MS = 100.0


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_prompt(assistant: Any, collection: str | None, prompt: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    first: float | None = None
    done: dict[str, Any] = {}
    error = ""
    grounded = False
    for ev in assistant.chat_events(ChatRequestData([Turn("user", prompt)], collection)):
        if ev["event"] == "context":
            grounded = bool(ev["data"]["grounded"])
        elif ev["event"] == "token" and first is None:
            first = time.perf_counter()
        elif ev["event"] == "done":
            done = ev["data"]
        elif ev["event"] == "error":
            error = ev["data"]["message"]
    t = done.get("timings", {})
    usage = done.get("usage", {})
    ttft = t.get("ttft_ms")
    stream_ms = t.get("stream_ms") or 0.0
    completion = int(usage.get("completion_tokens") or 0)
    rate = None
    if ttft is not None and completion > 1 and stream_ms > ttft:
        rate = completion / ((stream_ms - ttft) / 1000.0)  # tokens per second after the first
    projected = ttft + 300.0 / rate * 1000.0 if (ttft is not None and rate) else None
    return {
        "collection": collection or "-",
        "grounded": grounded,
        "cited": len(done.get("citations", [])),
        "gate_ms": t.get("gate_ms"),
        "queue_ms": t.get("queue_ms", 0.0),
        "ttft_ms": ttft,
        "e2e_first_ms": None
        if first is None
        else (first - t0) * 1000.0 - (t.get("queue_ms") or 0.0),
        "total_ms": None
        if not done
        else (time.perf_counter() - t0) * 1000.0 - (t.get("queue_ms") or 0.0),
        "completion_tokens": completion,
        "chars": len(done.get("answer", "")),
        "tok_per_s": rate,
        "projected_300_ms": projected,
        "error": error,
    }


def fmt(v: Any, nd: int = 0) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", default="hosted")
    ap.add_argument("--fake-llm", action="store_true", help="offline stand-in (no API calls)")
    ap.add_argument("--gate-runs", type=int, default=50)
    ap.add_argument("--collections-file", default=str(DEFAULT_COLLECTIONS_FILE))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    set_torch_threads(args.threads)
    reg = load_collections(args.collections_file)
    first = Collection(
        reg.ui_default or reg.default, reg.collections[reg.ui_default or reg.default]
    )
    llm_file = load_generation_settings(resolve_path(first.generation_config)).backends()
    if args.backend not in llm_file.names:
        print(f"unknown backend {args.backend!r}; configured: {', '.join(llm_file.names)}")
        return 2
    spec = llm_file.spec(args.backend)
    if not args.fake_llm and spec.requires_key and not key_present():
        print("LLM_API_KEY is not set: the hosted backend is not reachable; rerun with --fake-llm")
        return 2
    settings = load_assistant_settings()
    settings.limits.queue_wait_s = 120.0  # pace to the per-minute budget instead of failing
    assistant = build_assistant(
        args.collections_file,
        llm_file,
        [args.backend],
        args.backend,
        fake_llm=args.fake_llm,
        settings=settings,
    )
    model = (
        "offline stand-in" if args.fake_llm else f"{spec.model} ({spec.provider or args.backend})"
    )
    print(f"backend: {args.backend} - {model}")
    t0 = time.perf_counter()
    loaded = assistant.warm_up()
    print(
        f"start-up: {len(loaded)} indexes + bge-small loaded in {time.perf_counter() - t0:.1f}s "
        + ", ".join(f"{k} {len(assistant.index(assistant.collections[k]))} chunks" for k in loaded)
    )

    # ---- gate alone (embedding + search over each collection's index)
    gate_ms: list[float] = []
    names = [n for n in assistant.order if n in loaded]
    for i in range(args.gate_runs):
        coll = assistant.collections[names[i % len(names)]]
        msg = GATE_PROMPTS[i % len(GATE_PROMPTS)]
        d = assistant.gate.decide(
            msg,
            "What is a hash table?",
            [(assistant.index(coll), 0.62)],
            settings.context_budget(args.backend),
        )
        gate_ms.append(d.total_ms)
    print(
        f"gate (embedding + search), {len(gate_ms)} messages: p50 {pct(gate_ms, 50):.1f} ms, "
        f"p95 {pct(gate_ms, 95):.1f} ms, max {max(gate_ms):.1f} ms (target < {TARGET_GATE_MS:.0f} ms)"
    )

    # ---- 10 prompts end to end
    rows = []
    head = f"{'#':>2} {'collection':<18} {'grounded':<8} {'cited':>5} {'gate':>6} {'queue':>7} {'TTFT':>6} {'total':>7} {'tokens':>6} {'tok/s':>6} {'300tok':>7}"
    print("\n" + head + "\n" + "-" * len(head))
    for i, (coll, prompt) in enumerate(PROMPTS, start=1):
        r = run_prompt(assistant, coll, prompt)
        rows.append(r)
        print(
            f"{i:>2} {r['collection']:<18} {('yes' if r['grounded'] else 'no'):<8} {r['cited']:>5} "
            f"{fmt(r['gate_ms'], 1):>6} {fmt(r['queue_ms'], 0):>7} {fmt(r['ttft_ms']):>6} "
            f"{fmt(r['total_ms']):>7} {r['completion_tokens']:>6} {fmt(r['tok_per_s']):>6} "
            f"{fmt(r['projected_300_ms']):>7}" + (f"  ERROR: {r['error']}" if r["error"] else "")
        )
    ok = [r for r in rows if not r["error"] and r["ttft_ms"] is not None]
    ttft = [float(r["ttft_ms"]) for r in ok]
    e2e = [float(r["e2e_first_ms"]) for r in ok if r["e2e_first_ms"] is not None]
    total = [float(r["total_ms"]) for r in ok if r["total_ms"] is not None]
    proj = [float(r["projected_300_ms"]) for r in ok if r["projected_300_ms"] is not None]
    print("\nms (client-side pacing excluded)        p50      p95")
    print(
        f"TTFT, request -> first token        {pct(ttft, 50):7.0f}  {pct(ttft, 95):7.0f}   target p50 < {TARGET_TTFT_MS:.0f}"
    )
    print(f"first token incl. gate + prompt     {pct(e2e, 50):7.0f}  {pct(e2e, 95):7.0f}")
    print(f"full answer (as generated)          {pct(total, 50):7.0f}  {pct(total, 95):7.0f}")
    print(
        f"projected 300-token answer          {pct(proj, 50):7.0f}  {pct(proj, 95):7.0f}   target p50 < {TARGET_300_MS:.0f}"
    )
    print(
        f"answers: {len(ok)}/{len(rows)} ok, {sum(1 for r in ok if r['grounded'])} grounded, mean {statistics.mean([r['completion_tokens'] for r in ok]) if ok else 0:.0f} completion tokens"
    )
    if len(ok) < len(rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
