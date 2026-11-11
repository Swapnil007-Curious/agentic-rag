"""Run a collection's whole pipeline unattended (overnight), resumably.

Usage:
    python scripts/run_collection_pipeline.py --collection cs_ai [--dry-run] [--no-keep-awake]
    python scripts/run_collection_pipeline.py --collection cs_ai --summary-only
PowerShell wrapper: scripts/run_collection_pipeline.ps1 -Collection cs_ai

Steps, in order (each is skipped when already complete with unchanged inputs):
  ingest -> chunk -> build_index -> build_auto_eval -> validate_eval (stops on failure) ->
  ollama stop -> retrieval dev grid -> select_configs -> eval_retrieval test ->
  run_generation dev (the collection's dev systems; cs_ai: G2 G4) -> derive_systems dev (G3) ->
  eval_generation test (cs_ai: G1 G2 G3 G4; derives G3 on test) -> SUMMARY.md
A time estimate per step (from the corpus size and the speeds measured in this repository's
logs) is printed before starting. Everything is logged with timestamps, each step's exit code,
wall time and peak RSS to <results_dir>/pipeline_log.txt; step completion is recorded in
<results_dir>/pipeline_state.json. After a crash, a reboot or Ctrl+C, run the same command
again: finished steps are skipped and the interrupted step resumes from its own caches
(embeddings, LLM responses, per-config and per-question outputs). The run ends with
"PIPELINE COMPLETE" or names the failing step and the command to resume. While it runs, the
computer is kept from going to sleep (Windows; --no-keep-awake disables this).
"""

from __future__ import annotations

import argparse
import sys
import time

from ragsys.collections import add_collection_arguments, collection_from_args
from ragsys.config import resolve_path
from ragsys.pipeline_plan import build_steps, estimate_steps, plan_context, write_summary
from ragsys.pipeline_runner import Log, keep_awake, read_state, run_steps
from ragsys.utils.console import emit
from ragsys.utils.memory import fmt_duration


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_collection_arguments(ap)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and estimates only")
    ap.add_argument("--summary-only", action="store_true", help="only (re)write SUMMARY.md")
    ap.add_argument("--no-keep-awake", action="store_true", help="allow the computer to sleep")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    coll = collection_from_args(args)
    if args.summary_only:
        path = write_summary(coll)
        emit(f"wrote {path}")
        return 0

    ctx = plan_context(coll, args.collections_file)
    steps = build_steps(ctx)
    est, basis, info = estimate_steps(ctx)
    state = read_state(resolve_path(coll.pipeline_state))
    resume = f"python scripts/run_collection_pipeline.py --collection {coll.name}" + (
        f" --collections-file {args.collections_file}"
        if args.collections_file != "configs/collections.yaml"
        else ""
    )
    chunks = ", ".join(f"{k} {n} chunks (~{t:.0f} tokens)" for k, (n, t) in info["chunks"].items())
    emit(f"collection {coll.name} ({coll.title}): {info['pages']} pages kept; {chunks}")
    emit(f"questions planned: ~{info['n_dev']} dev + ~{info['n_test']} test")
    emit("estimated time per step (" + "; ".join(basis) + "):")
    remaining = 0.0
    for s in steps:
        done = s.name in state["steps"] and not s.always
        e = est.get(s.name, 0.0)
        s.estimate_s = e
        flag = "  (recorded as done; re-checked at run time)" if done else ""
        if not done:
            remaining += e
        emit(f"  {s.name:22s} {fmt_duration(e):>8s}{flag}")
    emit(f"  {'total (not yet done)':22s} {fmt_duration(remaining):>8s}")
    if args.dry_run:
        for s in steps:
            emit(f"  $ {s.command()}")
        return 0

    log = Log(resolve_path(coll.pipeline_log), echo=True)
    log.line(f"=== pipeline for collection {coll.name} started; estimate {fmt_duration(remaining)}")
    if not args.no_keep_awake:
        keep_awake(True)
    t0 = time.perf_counter()
    try:
        report = run_steps(steps, resolve_path(coll.pipeline_state), log, resume_hint=resume)
    except KeyboardInterrupt:
        log.line("INTERRUPTED (Ctrl+C); resume with: " + resume)
        return 130
    finally:
        if not args.no_keep_awake:
            keep_awake(False)
        log.line(f"=== total wall time {fmt_duration(time.perf_counter() - t0)}")
        log.close()
    if report.complete:
        emit("PIPELINE COMPLETE")
        emit(f"summary: {resolve_path(coll.summary_file)}")
        return 0
    emit(f"PIPELINE FAILED at step '{report.failed}'. See {resolve_path(coll.pipeline_log)}")
    emit(f"Resume with: {resume}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
