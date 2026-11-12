"""Tune the retrieval-score abstention threshold on the DEV split (no LLM calls).

Usage:
    python scripts/tune_abstention.py
The score is the abstention score of the tuned system's retriever (abstention.tuned_system in
configs/generation.yaml, G5 -> the structural|dense|bge-base retriever of G4): the top-1 dense
cosine of the first stage, or the top-1 rerank score when reranking is on. Candidate thresholds
are the distinct dev scores; a question is abstained when its score is below the threshold. The
threshold with the highest abstention F1 (positive class = unanswerable) wins, ties go to the
lower threshold. Writes results/tables/abstention_threshold.json with the threshold, the dev
precision / recall / F1, the whole curve and the per-question dev scores. Dev only: the test
split is never read here.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

from ragsys.agents.abstention import tune_threshold
from ragsys.agents.orchestrator import abstention_score
from ragsys.agents.runtime import load_settings
from ragsys.collections import (
    add_collection_arguments,
    collection_from_args,
    config_path,
    eval_index_refusal,
)
from ragsys.config import resolve_path
from ragsys.eval.generation import questions_digest
from ragsys.eval.split import questions_for_split
from ragsys.eval.validate import validate_file
from ragsys.retrieval.pipeline import RetrievalResources, Retriever
from ragsys.utils.console import emit
from ragsys.utils.memory import set_torch_threads
from ragsys.utils.runinfo import collect_run_info
from ragsys.utils.seed import set_seed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="default: the collection's generation config")
    add_collection_arguments(ap)
    ap.add_argument("--retrieval-config", default=None)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--threads", type=int, default=4)
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
    set_seed(settings.seed)
    set_torch_threads(args.threads)
    system = settings.abstention.tuned_system
    if system is None:
        emit(f"{coll.name}: no threshold system is configured (abstention.tuned_system: null)")
        return 2
    cfg = settings.retrieval_cfg(system)
    qpath = resolve_path(Path(args.questions) if args.questions else rsettings.paths.questions)
    processed = resolve_path(rsettings.paths.processed_dir)
    rep = validate_file(qpath, processed / "structure" / "nodes.jsonl")
    if not rep.ok:
        for e in rep.errors:
            emit(f"ERROR: {e}")
        return 1
    questions = questions_for_split(rep.questions, "dev")
    if not any(not q.answerable for q in questions) or all(not q.answerable for q in questions):
        emit("dev needs answerable and unanswerable questions to tune a threshold")
        return 1
    res = RetrievalResources(rsettings, rerank_cache=True)
    retriever = Retriever(cfg, res)
    retriever.warm_up()
    per_q = []
    for q in questions:
        score = abstention_score(retriever.search(q.question), cfg)
        if score is None:
            emit(f"ERROR: no abstention score for {q.id}")
            return 1
        per_q.append({"qid": q.id, "answerable": q.answerable, "score": round(float(score), 6)})
    res.close()
    tuned = tune_threshold([r["score"] for r in per_q], [not r["answerable"] for r in per_q])
    out = {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "split": "dev",
        "system": system,
        "retrieval_config_id": cfg.config_id(),
        "score": "top1_rerank_score" if cfg.rerank else "top1_dense_cosine",
        "rule": "abstain when score < threshold; threshold = distinct dev score maximising "
        "abstention F1 (positive = unanswerable); ties -> lower threshold",
        "threshold": tuned["threshold"],
        "dev": tuned["dev"],
        "n": tuned["n"],
        "n_unanswerable": tuned["n_unanswerable"],
        "questions_digest": questions_digest(questions),
        "curve": tuned["curve"],
        "dev_scores": per_q,
        "run_info": collect_run_info(),
    }
    path = settings.threshold_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    d = tuned["dev"]
    emit(
        f"{system} ({cfg.config_id()}): threshold {tuned['threshold']:.4f} on {tuned['n']} dev "
        f"questions ({tuned['n_unanswerable']} unanswerable): precision {d['precision']:.3f} "
        f"recall {d['recall']:.3f} F1 {d['f1']:.3f} ({d['n_abstained']} abstained)"
    )
    emit(f"wrote {path}")
    emit(f"total wall time {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
