"""Build the app indexes and run the hosted generator-scale ablation, unattended and resumably.

Usage:
    python scripts/run_app_collections.py [--dry-run] [--no-ablation] [--wait-for-quota]
                                          [--no-keep-awake]
PowerShell wrapper: scripts/run_app_collections.ps1

Steps, in order (each is skipped when already complete with unchanged inputs):
  indian_law, human_biology, physics_chemistry: ingest -> chunk -> build_index (structural
    chunks, bge-small embeddings, resumable in batches with progress and ETA)
  cs_ai app index (evaluation documents + Python Tutorial + Think Python, in
    collections/cs_ai/processed_app; the frozen evaluation corpus is not touched):
    ingest -> chunk -> build_index (bge-small)
  consumer_law, cs_ai: eval_generation.py --split test --systems G2H G4H (the hosted model of
    configs/llm.yaml; needs LLM_API_KEY; never falls back to the local model)
  consumer_law, cs_ai: SUMMARY.md with the ablation section
A time estimate per step (embedding speeds from this repository's logs, the API's token and
request limits for the ablation) is printed before starting. Output, exit codes, wall times and
peak memory go to collections/app_pipeline_log.txt; completion is recorded in
collections/app_pipeline_state.json. The run ends with "ALL DONE" or names the failing step and
the command to resume (after a crash, a reboot or an exhausted API quota: run the same command
again; finished embeddings batches and LLM calls are reused).

Daily API quota: the ablation needs more tokens than a free API account's daily quota (see the
estimate printed first). Without --wait-for-quota an ablation step stops when the quota is used
up (exit code 75) and the same command continues it once the quota has renewed. With
--wait-for-quota the run waits instead: each call waits out the provider's Retry-After (up to 6
hours) and a step stopped by the quota is started again every 30 minutes for up to 5 days, so
the computer must stay on (the runner keeps it awake unless --no-keep-awake).
"""

from __future__ import annotations

import argparse
import sys
import time

from ragsys.app_plan import LOG_FILE, STATE_FILE, build_app_steps, estimate_app_steps
from ragsys.collections import DEFAULT_COLLECTIONS_FILE
from ragsys.config import resolve_path
from ragsys.llm.backends import key_present
from ragsys.llm.client import EXIT_QUOTA
from ragsys.pipeline_runner import Log, keep_awake, read_state, run_steps
from ragsys.utils.console import emit
from ragsys.utils.memory import fmt_duration


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--collections-file", default=str(DEFAULT_COLLECTIONS_FILE))
    ap.add_argument("--dry-run", action="store_true", help="print the plan and estimates only")
    ap.add_argument("--no-ablation", action="store_true", help="build the app indexes only")
    ap.add_argument(
        "--wait-for-quota",
        action="store_true",
        help="wait for the API's daily quota to renew instead of stopping (unattended, days)",
    )
    ap.add_argument("--no-keep-awake", action="store_true", help="allow the computer to sleep")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    steps = build_app_steps(
        args.collections_file, ablation=not args.no_ablation, wait_for_quota=args.wait_for_quota
    )
    est, basis, info = estimate_app_steps(steps, args.collections_file)
    state = read_state(resolve_path(STATE_FILE))
    resume = (
        "python scripts/run_app_collections.py"
        + (" --no-ablation" if args.no_ablation else "")
        + (" --wait-for-quota" if args.wait_for_quota else "")
    )
    for name, t in info["targets"].items():
        chunks = ", ".join(f"{k} {n} chunks (~{m:.0f} tokens)" for k, (n, m) in t["chunks"].items())
        emit(f"{name}: {t['pages']} pages kept; {chunks or 'chunks not built yet'}")
    if not args.no_ablation:
        for name, tok in info["ablation"].items():
            emit(
                f"{name} ablation: {tok['n_test']} test questions x G2H/G4H on {tok['model']}: "
                f"about {tok['tokens'] / 1e6:.2f} M API tokens in ~{tok['calls']} calls "
                f"(revision share {tok['revision_share']:.0%} as in the local G4 run); at "
                f"{tok['tpm']:.0f} tokens/min and {tok['rpm']:.0f} requests/min that is at least "
                f"{fmt_duration(tok['api_minutes'] * 60)} of API time"
            )
        tpd = next((t["tpd"] for t in info["ablation"].values() if t.get("tpd")), None)
        days = float(info.get("quota_days") or 0.0)
        if tpd and days > 1.0:
            emit(
                f"  daily quota: {tpd:,} tokens/day (configs/llm.yaml), so the ablation needs "
                f"about {days:.1f} days of quota."
            )
            if args.wait_for_quota:
                emit(
                    "  --wait-for-quota: the run waits for the quota to renew (keep the computer "
                    "on); the estimate below includes that waiting"
                )
            else:
                emit(
                    "  Without --wait-for-quota an ablation step stops when the day's quota is "
                    "used up (exit code 75); run the same command again the next day (about "
                    f"{int(days) + 1} runs in all; answered questions and cached calls are "
                    "reused), or add --wait-for-quota for one unattended multi-day run. The "
                    "estimate below assumes the quota renews while the run waits."
                )
        else:
            emit(
                "  a daily token or request quota on the API account can stop an ablation step "
                "early (exit code 75); the same command resumes it later (answered questions "
                "and cached calls are reused)"
            )
        if not key_present():
            emit("  WARNING: LLM_API_KEY is not set in this shell; the ablation steps will stop")
    emit("estimated time per step (" + "; ".join(basis) + "):")
    remaining = 0.0
    for s in steps:
        done = s.name in state["steps"] and not s.always
        e = est.get(s.name, 0.0)
        s.estimate_s = e
        if not done:
            remaining += e
        flag = "  (recorded as done; re-checked at run time)" if done else ""
        emit(f"  {s.name:28s} {fmt_duration(e):>8s}{flag}")
    emit(f"  {'total (not yet done)':28s} {fmt_duration(remaining):>8s}")
    if args.dry_run:
        for s in steps:
            emit(f"  $ {s.command()}")
        return 0

    log = Log(resolve_path(LOG_FILE), echo=True)
    log.line(f"=== app collections run started; estimate {fmt_duration(remaining)}")
    if not args.no_keep_awake:
        keep_awake(True)
    t0 = time.perf_counter()
    try:
        report = run_steps(steps, resolve_path(STATE_FILE), log, resume_hint=resume)
    except KeyboardInterrupt:
        log.line("INTERRUPTED (Ctrl+C); resume with: " + resume)
        return 130
    finally:
        if not args.no_keep_awake:
            keep_awake(False)
        log.line(f"=== total wall time {fmt_duration(time.perf_counter() - t0)}")
        log.close()
    if report.complete:
        emit("ALL DONE")
        return 0
    emit(f"FAILED at step '{report.failed}'. See {resolve_path(LOG_FILE)}")
    last = report.outcomes[-1] if report.outcomes else None
    if last is not None and last.returncode == EXIT_QUOTA:
        emit(
            "The API's quota is used up for now (nothing fell back to another model). Run the "
            "same command again once it has renewed (usually within a day), or add "
            "--wait-for-quota to wait unattended."
        )
    emit(f"Resume with: {resume}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
