"""Run answer-generation systems on the DEV split and evaluate them.

Usage:
    python scripts/run_generation.py --split dev --systems G1 G2 G4 G6
Resumable per question: results/generation/dev/<system>/per_question.jsonl gets one record per
answered question; rerunning the command continues where it stopped (records whose question or
system definition changed are recomputed; unchanged LLM calls come from the response cache).
After the runs the retrieval models are released and the NLI model scores every answer
(scores cached in data/processed/nli_cache/). Tables: results/tables/generation_dev.csv, one
row per system with every available system (also the derived G3/G5 when scripts/derive_systems.py
has produced them), generation_dev_by_type.csv and generation_dev_bootstrap.csv (vs G1).
Derived systems (G3, G5) are not run here: use scripts/derive_systems.py. The test split cannot
be selected (see scripts/eval_generation.py). Progress lines show an ETA; the peak RSS is printed
at the end. Needs Ollama running (configs/llm.yaml).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ragsys.agents.runtime import build_engine, llm_unreachable_message, load_settings, make_llm
from ragsys.config import resolve_path
from ragsys.eval.generation import (
    evaluate_systems,
    load_system_records,
    ordered_systems,
    run_system,
    system_fingerprint,
)
from ragsys.eval.split import questions_for_split
from ragsys.eval.validate import validate_file
from ragsys.utils.console import emit
from ragsys.utils.memory import fmt_mb, peak_rss_bytes, rss_bytes, set_torch_threads
from ragsys.utils.seed import set_seed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["dev"], default="dev", help="only dev is allowed")
    ap.add_argument("--systems", nargs="+", default=["G1", "G2", "G4", "G6"])
    ap.add_argument("--config", default="configs/generation.yaml")
    ap.add_argument("--retrieval-config", default=None, help="override retrieval_config")
    ap.add_argument("--questions", default=None, help="override the question file")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing records")
    ap.add_argument("--no-eval", action="store_true", help="skip the NLI evaluation and tables")
    ap.add_argument("--fake-llm", action="store_true", help="offline deterministic stand-in")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (default 4)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    t_start = time.perf_counter()
    settings, rsettings = load_settings(args.config, args.retrieval_config)
    systems = ordered_systems(settings, args.systems)
    derived = [s for s in systems if settings.is_derived(s)]
    if derived:
        emit(f"derived system(s) {', '.join(derived)} are computed by scripts/derive_systems.py")
        return 2
    set_seed(settings.seed)
    set_torch_threads(args.threads)
    qpath = resolve_path(Path(args.questions) if args.questions else rsettings.paths.questions)
    rep = validate_file(
        qpath, resolve_path(rsettings.paths.processed_dir) / "structure" / "nodes.jsonl"
    )
    if not rep.ok:
        for e in rep.errors:
            emit(f"ERROR: {e}")
        return 1
    questions = questions_for_split(rep.questions, "dev")
    n_ans = sum(1 for q in questions if q.answerable)
    emit(f"dev split: {n_ans} answerable + {len(questions) - n_ans} unanswerable question(s)")
    llm = make_llm(settings, fake=args.fake_llm)
    problem = llm_unreachable_message(llm)
    if problem:
        emit(f"ERROR: {problem}")
        return 1
    llm_cfg = settings.llm()
    emit(
        f"LLM {llm.model} (temperature {llm_cfg.temperature}, seed {llm_cfg.seed}); systems: {', '.join(systems)}"
    )
    engine = build_engine(settings, rsettings, llm)
    out_dir = resolve_path(settings.results_dir) / "generation" / "dev"
    for system in systems:
        fp = system_fingerprint(settings, system, llm_cfg)
        run_system(
            engine, system, questions, "dev", out_dir, fp, overwrite=args.overwrite, log=emit
        )
    released = engine.resources.release_encoders()
    if released:
        emit(
            f"released {', '.join(released)} before the NLI evaluation | RSS {fmt_mb(rss_bytes())}"
        )
    if not args.no_eval:
        records = {}
        for s in settings.systems:  # every system with complete dev outputs, in config order
            recs = load_system_records(out_dir, s, questions)
            if recs is not None:
                records[s] = recs
        emit(f"evaluating {', '.join(records)} with {settings.nli.model}")
        paths = evaluate_systems(
            settings,
            engine.resources,
            records,
            questions,
            "dev",
            out_dir,
            resolve_path(settings.results_dir) / "tables",
            bootstrap=True,
            log=emit,
        )
        for p in paths:
            emit(f"wrote {p}")
    engine.close()
    emit(f"peak RSS {fmt_mb(peak_rss_bytes())}")
    emit(f"total wall time {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
