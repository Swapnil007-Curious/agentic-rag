"""Evaluate the answer-generation systems on the TEST split (the only generation code path that
reads test questions).

Usage:
    python scripts/eval_generation.py --split test [--collection NAME] --systems G1 G2 G3 G4 G5 G6
--systems defaults to the collection's test systems (consumer_law: G1-G6; cs_ai: G1-G4). The
abstention threshold file is required only when a requested system uses the threshold (G5).
Run once, after the dev runs, derive_systems.py and tune_abstention.py. Refuses to start when
results/tables/abstention_threshold.json (tuned on dev) is missing, when the evaluation set has
fewer answerable questions than selection.min_answerable_for_test, or when a generation test
evaluation already completed (pass --allow-repeat to run again; it is logged).
Each start appends an entry to results/test_access_log.json (UTC time, git commit, dirty flag,
systems, number of questions). Resumable: per-question records go to
results/generation/test/<system>/per_question.jsonl as they are produced; rerun the same command
after an interruption (the access is logged again, marked as a resume).
Runs the runnable systems needed (G1, G2, G4, G6; the bases of G3 and G5), derives G3 from G2 and
G5 from G4 plus the threshold, releases the retrieval models, scores every answer with the NLI
model and writes results/tables/generation_test.csv, generation_test_by_type.csv and
generation_test_bootstrap.csv (paired bootstrap vs G1). Prints an ETA per question and the peak
RSS. Needs Ollama running (configs/llm.yaml).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Any

from ragsys.agents.abstention import load_threshold
from ragsys.agents.config import SelectionMissing
from ragsys.agents.runtime import build_engine, llm_unreachable_message, load_settings, make_llm
from ragsys.collections import add_collection_arguments, collection_from_args, config_path
from ragsys.config import resolve_path
from ragsys.eval.generation import (
    derive_system,
    evaluate_systems,
    ordered_systems,
    questions_digest,
    read_jsonl,
    run_system,
    system_fingerprint,
)
from ragsys.eval.split import grant_test_access, questions_for_split, split_of
from ragsys.eval.validate import validate_file
from ragsys.utils.console import emit
from ragsys.utils.memory import fmt_mb, peak_rss_bytes, rss_bytes, set_torch_threads
from ragsys.utils.seed import set_seed

ALL_SYSTEMS = ["G1", "G2", "G3", "G4", "G5", "G6"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["test"], required=True)
    ap.add_argument("--systems", nargs="+", default=None, help="default: the collection's")
    ap.add_argument("--config", default=None, help="default: the collection's generation config")
    add_collection_arguments(ap)
    ap.add_argument("--retrieval-config", default=None)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--allow-repeat", action="store_true", help="run although it completed")
    ap.add_argument("--min-answerable", type=int, default=None)
    ap.add_argument("--fake-llm", action="store_true", help="offline deterministic stand-in")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (default 4)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    t_start = time.perf_counter()
    coll = collection_from_args(args)
    settings, rsettings = load_settings(
        config_path(args.config, coll.generation_config), args.retrieval_config
    )
    systems = ordered_systems(settings, args.systems or coll.spec.pipeline.test_systems)
    needs_threshold = any(settings.system(s).threshold for s in systems)
    try:
        info = load_threshold(settings.threshold_path())
    except ValueError as exc:
        emit(f"refusing to evaluate the test split: {exc}")
        return 2
    if info is None and needs_threshold:
        emit(
            f"refusing to evaluate the test split: {settings.threshold_path()} is missing. "
            "Tune the abstention threshold on dev first (python scripts/tune_abstention.py); "
            "the test split stays unopened."
        )
        return 2
    threshold = float(info["threshold"]) if info is not None else None
    try:
        for s in systems:
            settings.retrieval_cfg(s)
    except SelectionMissing as exc:
        emit(f"refusing to evaluate the test split: {exc}; the test split stays unopened.")
        return 2
    qpath = resolve_path(Path(args.questions) if args.questions else rsettings.paths.questions)
    processed = resolve_path(rsettings.paths.processed_dir)
    rep = validate_file(qpath, processed / "structure" / "nodes.jsonl")
    if not rep.ok:
        for e in rep.errors:
            emit(f"ERROR: {e}")
        return 1
    minimum = args.min_answerable or rsettings.selection.min_answerable_for_test
    n_answerable = sum(1 for q in rep.questions if q.answerable)
    if n_answerable < minimum:
        emit(
            f"refusing to evaluate the test split: only {n_answerable} answerable questions "
            f"(minimum {minimum}); the test split stays unopened."
        )
        return 2
    root = resolve_path(settings.results_dir)
    out_dir = root / "generation" / "test"
    marker = out_dir / "_complete.json"
    if marker.exists() and not args.allow_repeat:
        emit(
            f"a generation test evaluation already completed (see {marker}); "
            "pass --allow-repeat to evaluate again (it will be logged)"
        )
        return 3
    llm = make_llm(settings, fake=args.fake_llm)
    problem = llm_unreachable_message(llm)
    if problem:
        emit(f"ERROR: {problem}")
        return 1
    set_seed(settings.seed)
    set_torch_threads(args.threads)

    runnable = [s for s in settings.systems if not settings.is_derived(s)]
    needed = [
        s
        for s in runnable
        if s in systems
        or any(settings.is_derived(d) and settings.runnable_base(d) == s for d in systems)
    ]
    resume = any(read_jsonl(out_dir / s / "per_question.jsonl") for s in needed)
    n_test = sum(1 for q in rep.questions if split_of(q.id) == "test")
    note = (
        f"generation{' (resume)' if resume else ''}{' (repeat)' if marker.exists() else ''}: "
        f"systems {' '.join(systems)}; "
        + (
            f"abstention threshold {threshold:.4f} from {settings.threshold_path()}"
            if threshold is not None
            else "no abstention threshold (no threshold system)"
        )
    )
    log_path = root / "test_access_log.json"
    access = grant_test_access(log_path, systems, n_test, note=note)
    emit(f"TEST split opened ({n_test} questions); access logged to {log_path}")
    questions = questions_for_split(rep.questions, "test", access)
    llm_cfg = settings.llm()
    engine = build_engine(settings, rsettings, llm, require_threshold=needs_threshold)
    records: dict[str, list[dict[str, Any]]] = {}
    for system in needed:
        fp = system_fingerprint(settings, system, llm_cfg)
        records[system] = run_system(engine, system, questions, "test", out_dir, fp, log=emit)
    for system in systems:
        if settings.is_derived(system):
            base = settings.runnable_base(system)
            thr = threshold if settings.system(system).threshold else None
            records[system] = derive_system(settings, system, records[base], out_dir, thr)
            n_abst = sum(1 for r in records[system] if r["outcome"] == "abstained")
            emit(f"{system} <- {base}: {n_abst}/{len(questions)} abstained")
    released = engine.resources.release_encoders()
    if released:
        emit(
            f"released {', '.join(released)} before the NLI evaluation | RSS {fmt_mb(rss_bytes())}"
        )
    evaluated = {s: records[s] for s in systems}
    paths = evaluate_systems(
        settings,
        engine.resources,
        evaluated,
        questions,
        "test",
        out_dir,
        root / "tables",
        bootstrap=True,
        log=emit,
        extra_meta={"abstention_threshold": info},
    )
    engine.close()
    marker.write_text(
        json.dumps(
            {
                "completed_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                "systems": systems,
                "n_questions": len(questions),
                "questions_digest": questions_digest(questions),
                "threshold": threshold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for p in paths:
        emit(f"wrote {p}")
    emit(f"peak RSS {fmt_mb(peak_rss_bytes())}")
    emit(f"total wall time {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
