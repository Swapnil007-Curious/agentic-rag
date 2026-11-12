"""Run the retrieval grid of configs/retrieval.yaml on the DEV split.

Usage:
    python scripts/run_retrieval_grid.py --split dev [--collection NAME] [--overwrite]
                                         [--only TEXT ...] [--smoke] [--no-rerank-cache] [--threads 4]
Writes results/retrieval/dev/<config>/per_query.jsonl and summary.json for every config, and
results/tables/retrieval_dev.csv (one row per config), retrieval_dev_by_type.csv,
_by_document.csv, _by_origin.csv, _bootstrap.csv (vs the naive baseline) and .meta.json.
Configs whose outputs exist for the same dev question set are skipped (resumable) unless
--overwrite. --smoke writes everything under results/smoke/ and labels every row
"smoke - not results". The test split cannot be selected here (see eval_retrieval.py).
Memory/speed: configs run grouped by dense model (BM25-only first, then one encoder at a time,
released between groups); the reranker is loaded once; rerank scores are cached in
data/processed/rerank_cache/<model>.sqlite (latencies stored on first scoring are reported on
cache hits; --no-rerank-cache disables it); query embeddings are cached in memory for the run.
RSS is printed at group boundaries, an ETA after each config, and the peak RSS at the end.
Build the indexes first: python scripts/build_index.py --chunker all --models bge-small bge-base bge-large
"""

from __future__ import annotations

import argparse
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
from ragsys.eval.harness import (
    SMOKE_LABEL,
    Evaluator,
    load_existing,
    run_configs,
    split_counts,
    write_tables,
)
from ragsys.eval.validate import validate_file
from ragsys.retrieval.config import load_retrieval_settings
from ragsys.retrieval.pipeline import RetrievalResources
from ragsys.utils.console import emit
from ragsys.utils.memory import fmt_mb, peak_rss_bytes, set_torch_threads
from ragsys.utils.seed import set_seed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["dev"], default="dev", help="only dev is allowed")
    ap.add_argument("--config", default=None, help="default: the collection's retrieval config")
    ap.add_argument("--questions", default=None, help="override paths.questions")
    add_collection_arguments(ap)
    ap.add_argument("--overwrite", action="store_true", help="recompute existing configs")
    ap.add_argument("--only", nargs="+", default=None, help="run configs whose id contains TEXT")
    ap.add_argument("--smoke", action="store_true", help='write to results/smoke, label "smoke"')
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
    threads = set_torch_threads(args.threads)
    qpath = resolve_path(Path(args.questions) if args.questions else settings.paths.questions)
    res = RetrievalResources(settings, rerank_cache=not args.no_rerank_cache)
    rep = validate_file(qpath, res.processed_dir / "structure" / "nodes.jsonl")
    if not rep.ok:
        for e in rep.errors:
            emit(f"ERROR: {e}")
        emit("fix the evaluation set first (python scripts/validate_eval.py)")
        return 1
    counts = split_counts(rep.questions)
    root = resolve_path(settings.paths.results_dir)
    label = ""
    if args.smoke:
        root = root / "smoke"
        label = SMOKE_LABEL
        emit(f"*** {SMOKE_LABEL.upper()} ***")
    emit(
        f"dev split: {counts['dev']['answerable']} answerable + "
        f"{counts['dev']['unanswerable']} unanswerable question(s)"
    )
    if counts["dev"]["answerable"] == 0:
        emit("WARNING: no answerable dev questions; retrieval metrics will be empty (n/a)")

    grid = settings.grid_configs()
    configs = grid
    if args.only:
        configs = [c for c in grid if any(s in c.config_id() for s in args.only)]
    emit(
        f"{len(configs)} config(s); torch threads {threads}; rerank cache "
        + ("off" if res.rerank_cache_dir is None else f"{res.rerank_cache_dir}")
    )
    evaluator = Evaluator(settings, res)
    out_root = root / "retrieval" / "dev"
    run_configs(
        evaluator,
        configs,
        rep.questions,
        "dev",
        out_root,
        overwrite=args.overwrite,
        label=label,
        log=emit,
    )
    # tables always cover every grid config with up-to-date outputs, not only this run's subset
    summaries, records = load_existing(grid, rep.questions, "dev", out_root, label=label)
    emit(f"tables cover {len(summaries)}/{len(grid)} grid config(s)")
    baseline = settings.selection.baselines[settings.selection.reference_baseline]
    paths = write_tables(
        root / "tables",
        "retrieval_dev",
        summaries,
        records,
        settings,
        baseline_id=baseline,
        label=label,
        extra_meta={"split": "dev", "split_counts": counts},
    )
    res.close()
    for p in paths:
        emit(f"wrote {p}")
    emit(f"peak RSS {fmt_mb(peak_rss_bytes())}")
    emit(f"total wall time {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
