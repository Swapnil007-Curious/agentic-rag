"""Compute the derived systems on the DEV split from their base systems' outputs (no LLM calls).

Usage:
    python scripts/derive_systems.py --split dev [--collection NAME] [--systems G3 G5]
G3 = G2 + deterministic guards (a guard failure becomes an abstention), from
results/generation/dev/G2/per_question.jsonl. G5 = G4 + the retrieval-score threshold of
results/tables/abstention_threshold.json (questions below it are abstained with their LLM calls
removed from the cost), from results/generation/dev/G4/. Writes
results/generation/dev/<system>/per_question.jsonl, then re-evaluates every system with complete
dev outputs and rewrites results/tables/generation_dev.csv (+ _by_type, _bootstrap). NLI scores
of derived answers come from the cache (they are the base system's answers). The test split is
handled only by scripts/eval_generation.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ragsys.agents.abstention import load_threshold
from ragsys.agents.derive import derivation_kind
from ragsys.agents.runtime import load_settings
from ragsys.collections import (
    add_collection_arguments,
    collection_from_args,
    config_path,
    eval_index_refusal,
)
from ragsys.config import resolve_path
from ragsys.eval.generation import (
    derive_system,
    evaluate_systems,
    load_system_records,
    ordered_systems,
)
from ragsys.eval.split import questions_for_split
from ragsys.eval.validate import validate_file
from ragsys.retrieval.pipeline import RetrievalResources
from ragsys.utils.console import emit
from ragsys.utils.memory import fmt_mb, peak_rss_bytes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["dev"], default="dev", help="only dev is allowed")
    ap.add_argument("--systems", nargs="+", default=None, help="default: every derived system")
    ap.add_argument("--config", default=None, help="default: the collection's generation config")
    add_collection_arguments(ap)
    ap.add_argument("--retrieval-config", default=None)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--no-eval", action="store_true", help="skip the evaluation and tables")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    t_start = time.perf_counter()
    coll = collection_from_args(args)
    settings, rsettings = load_settings(
        config_path(args.config, coll.generation_config), args.retrieval_config
    )
    refusal = eval_index_refusal(coll, rsettings)
    if refusal:
        emit(f"refusing to run: {refusal}")
        return 2
    names = args.systems or [s for s in settings.systems if settings.is_derived(s)]
    systems = ordered_systems(settings, names)
    not_derived = [s for s in systems if not settings.is_derived(s)]
    if not_derived:
        emit(f"{', '.join(not_derived)} are not derived systems (run them with run_generation.py)")
        return 2
    qpath = resolve_path(Path(args.questions) if args.questions else rsettings.paths.questions)
    processed = resolve_path(rsettings.paths.processed_dir)
    rep = validate_file(qpath, processed / "structure" / "nodes.jsonl")
    if not rep.ok:
        for e in rep.errors:
            emit(f"ERROR: {e}")
        return 1
    questions = questions_for_split(rep.questions, "dev")
    out_dir = resolve_path(settings.results_dir) / "generation" / "dev"
    info = load_threshold(settings.threshold_path())
    for system in systems:
        base = settings.runnable_base(system)
        base_recs = load_system_records(out_dir, base, questions)
        if base_recs is None:
            emit(
                f"{system}: {base} has no complete dev outputs; run run_generation.py --systems {base}"
            )
            return 1
        threshold = None
        if derivation_kind(settings, system) == "threshold":
            if info is None:
                emit(
                    f"{system}: {settings.threshold_path()} is missing; run tune_abstention.py first"
                )
                return 1
            threshold = float(info["threshold"])
        recs = derive_system(settings, system, base_recs, out_dir, threshold)
        n_abst = sum(1 for r in recs if r["outcome"] == "abstained")
        extra = f" (threshold {threshold:.4f})" if threshold is not None else ""
        emit(f"{system} <- {base}{extra}: {n_abst}/{len(recs)} abstained")
    if not args.no_eval:
        res = RetrievalResources(rsettings)
        records = {}
        for s in settings.systems:
            recs = load_system_records(out_dir, s, questions)
            if recs is not None:
                records[s] = recs
        emit(f"evaluating {', '.join(records)}")
        paths = evaluate_systems(
            settings,
            res,
            records,
            questions,
            "dev",
            out_dir,
            resolve_path(settings.results_dir) / "tables",
            bootstrap=True,
            log=emit,
            extra_meta={"abstention_threshold": info},
        )
        for p in paths:
            emit(f"wrote {p}")
        res.close()
    emit(f"peak RSS {fmt_mb(peak_rss_bytes())}")
    emit(f"total wall time {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
