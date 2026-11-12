"""Evaluate the selected configs on the TEST split (the only code path that reads test).

Usage:
    python scripts/eval_retrieval.py --split test [--collection NAME]
                                     [--configs-from <results_dir>/tables/selected_configs.json]
Run once, after the dev grid and select_configs.py. Writes results/retrieval/test/<config>/,
results/tables/retrieval_test.csv, retrieval_test_by_type.csv (also _by_document, _by_origin),
retrieval_test_bootstrap.csv (each selected config vs the naive RAG baseline: paired bootstrap
of Recall@5 and coverage@1024) and appends an entry to results/test_access_log.json (UTC time,
git commit, dirty flag, configs, number of questions). Refuses to run when the evaluation set
has fewer answerable questions than selection.min_answerable_for_test, or when the test split
was already accessed (pass --allow-repeat to run again; the repeat is logged too).
Configs run grouped by dense model with one encoder in memory at a time, the rerank score cache
is used (--no-rerank-cache disables it; stored latencies are reported on hits) and RSS, an ETA
per config and the peak RSS are printed, as in run_retrieval_grid.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ragsys.collections import (
    add_collection_arguments,
    collection_from_args,
    config_path,
    eval_index_refusal,
)
from ragsys.config import resolve_path
from ragsys.eval.harness import Evaluator, run_configs, write_tables
from ragsys.eval.split import grant_test_access, read_access_log, split_of
from ragsys.eval.validate import validate_file
from ragsys.retrieval.config import RetrievalConfig, load_retrieval_settings, parse_config_id
from ragsys.retrieval.pipeline import RetrievalResources
from ragsys.utils.console import emit
from ragsys.utils.memory import fmt_mb, peak_rss_bytes, set_torch_threads
from ragsys.utils.seed import set_seed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["test"], required=True)
    ap.add_argument("--configs-from", default=None, help="default: <results_dir>/tables/...")
    ap.add_argument("--config", default=None, help="default: the collection's retrieval config")
    add_collection_arguments(ap)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--allow-repeat", action="store_true", help="run although test was accessed")
    ap.add_argument("--min-answerable", type=int, default=None)
    ap.add_argument(
        "--no-rerank-cache", action="store_true", help="always rescore with the reranker"
    )
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (default 4)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    t_start = time.perf_counter()
    coll = collection_from_args(args)
    settings = load_retrieval_settings(
        resolve_path(config_path(args.config, coll.retrieval_config))
    )
    refusal = eval_index_refusal(coll, settings)
    if refusal:
        emit(f"refusing to run: {refusal}")
        return 2
    set_seed(settings.seed)
    set_torch_threads(args.threads)
    res = RetrievalResources(settings, rerank_cache=not args.no_rerank_cache)
    qpath = resolve_path(Path(args.questions) if args.questions else settings.paths.questions)
    rep = validate_file(qpath, res.processed_dir / "structure" / "nodes.jsonl")
    if not rep.ok:
        for e in rep.errors:
            emit(f"ERROR: {e}")
        return 1
    minimum = args.min_answerable or settings.selection.min_answerable_for_test
    n_answerable = sum(1 for q in rep.questions if q.answerable)
    if n_answerable < minimum:
        emit(
            f"refusing to evaluate the test split: only {n_answerable} answerable questions "
            f"(minimum {minimum}). Write more questions first; the test split stays unopened."
        )
        return 2
    root = resolve_path(settings.paths.results_dir)
    log_path = root / "test_access_log.json"
    previous = read_access_log(log_path)
    if previous and not args.allow_repeat:
        emit(
            f"the test split was already accessed {len(previous)} time(s) (see {log_path}); "
            "pass --allow-repeat to evaluate again (it will be logged)"
        )
        return 3

    sel_path = (
        resolve_path(Path(args.configs_from))
        if args.configs_from
        else root / "tables" / "selected_configs.json"
    )
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    configs = [RetrievalConfig.model_validate(s["config"]) for s in selection["selected"]]
    baseline = selection.get("reference_baseline_id") or settings.selection.baselines["naive_rag"]
    if baseline not in {c.config_id() for c in configs}:
        configs.append(parse_config_id(baseline))
    ids = [c.config_id() for c in configs]
    n_test = sum(1 for q in rep.questions if split_of(q.id) == "test")

    access = grant_test_access(log_path, ids, n_test, note=f"selection: {sel_path}")
    emit(f"TEST split opened ({n_test} questions); access logged to {log_path}")
    evaluator = Evaluator(settings, res)
    summaries, records = run_configs(
        evaluator,
        configs,
        rep.questions,
        "test",
        root / "retrieval" / "test",
        access=access,
        overwrite=True,
        log=emit,
    )
    paths = write_tables(
        root / "tables",
        "retrieval_test",
        summaries,
        records,
        settings,
        baseline_id=baseline,
        extra_meta={"split": "test", "selection": str(sel_path)},
    )
    res.close()
    for p in paths:
        emit(f"wrote {p}")
    emit(f"peak RSS {fmt_mb(peak_rss_bytes())}")
    emit(f"total wall time {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
