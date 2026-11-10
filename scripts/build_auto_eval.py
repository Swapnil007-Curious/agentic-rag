"""Build the evaluation set automatically with a local LLM and automatic quality filters.

Usage:
    python scripts/build_auto_eval.py [--rebuild] [--seed 0]
        [--n-candidates 200] [--target-answerable 90]
        [--n-unanswerable-candidates 90] [--target-unanswerable 30]
        [--llm-config configs/llm.yaml] [--config configs/retrieval.yaml]
        [--questions data/eval/questions.jsonl] [--report results/tables/eval_construction.json]

Answerable: 200 candidates from the drafting code (same type mix, seed 0), then the filters
format -> provision_number -> copy (one paraphrase attempt) -> numbers -> support (LLM) ->
answerable_from_source (LLM + deterministic comparison) -> near_duplicate (bge-small), then up
to 90 survivors stratified by type and document. Unanswerable: 45 out-of-scope + 45
perturbation candidates, filtered (format, no section/rule number of this corpus,
near-duplicates, LLM check of the top-5 hybrid_rrf/bge-small structural chunks), 30 selected
balanced over the two methods. See src/ragsys/eval/autogen.py for details.

The selected items are APPENDED to questions.jsonl (ids continue from the existing ones; origin
"llm_auto_filtered", author "auto (<model>) - filtered", with a gen_meta record) and
results/tables/eval_construction.json is written. The script refuses to run when
questions.jsonl already holds llm_auto_filtered items, unless --rebuild, which replaces only
those items (human-written and human-reviewed items are never touched).

Long run (1-3 hours with qwen2.5:3b on a laptop CPU). Every LLM call is cached on disk
(configs/llm.yaml cache_dir), so an interrupted run resumes where it stopped: just run the same
command again. The bge-small encoder is released before the script ends; do not run the
retrieval grid at the same time.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ragsys.config import load_ingest_config, resolve_path
from ragsys.eval.autogen import AutoEvalBuilder
from ragsys.eval.schema import AUTO_ORIGIN, EvalQuestion
from ragsys.eval.split import split_of
from ragsys.eval.validate import load_questions
from ragsys.ingest.pipeline import load_nodes, load_xrefs
from ragsys.llm.client import ChatModel, LLMClient, load_llm_config
from ragsys.retrieval.config import load_retrieval_settings
from ragsys.retrieval.pipeline import RetrievalResources
from ragsys.utils.console import emit
from ragsys.utils.memory import fmt_duration, fmt_mb, peak_rss_bytes, rss_bytes, set_torch_threads
from ragsys.utils.runinfo import collect_run_info

EXIT_EXISTING_AUTO = 3


def _origin_of(line: str) -> str:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return "human"
    return str(data.get("origin", "human")) if isinstance(data, dict) else "human"


def split_lines(path: Path) -> tuple[list[str], list[str]]:
    """(lines of non-automatic items, kept byte-for-byte; lines of llm_auto_filtered items)."""
    if not path.exists():
        return [], []
    keep, auto = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        (auto if _origin_of(line) == AUTO_ORIGIN else keep).append(line)
    return keep, auto


def write_questions(path: Path, keep_lines: list[str], new: list[EvalQuestion]) -> None:
    lines = keep_lines + [
        json.dumps(q.model_dump(mode="json", exclude_none=True), ensure_ascii=False) for q in new
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")
    tmp.replace(path)


def final_counts(questions: list[EvalQuestion]) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {
        sp: {"answerable": 0, "unanswerable": 0} for sp in ("dev", "test")
    }
    by_split_type: dict[str, Counter[str]] = {"dev": Counter(), "test": Counter()}
    for q in questions:
        sp = split_of(q.id)
        by_split[sp]["answerable" if q.answerable else "unanswerable"] += 1
        by_split_type[sp][q.type] += 1
    docs = Counter(
        q.gold_node_ids[0].split(":", 1)[0] if q.gold_node_ids else "(none)" for q in questions
    )
    return {
        "total": len(questions),
        "answerable": sum(1 for q in questions if q.answerable),
        "unanswerable": sum(1 for q in questions if not q.answerable),
        "by_type": dict(sorted(Counter(q.type for q in questions).items())),
        "by_document": dict(sorted(docs.items())),
        "by_difficulty": dict(sorted(Counter(q.difficulty for q in questions).items())),
        "by_split": by_split,
        "by_split_type": {sp: dict(sorted(c.items())) for sp, c in by_split_type.items()},
    }


def main(argv: list[str] | None = None, llm: ChatModel | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rebuild", action="store_true", help="replace existing automatic items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-candidates", type=int, default=200)
    ap.add_argument("--target-answerable", type=int, default=90)
    ap.add_argument("--n-unanswerable-candidates", type=int, default=90)
    ap.add_argument("--target-unanswerable", type=int, default=30)
    ap.add_argument("--dense-model", default="bge-small", help="retrieval check / duplicates")
    ap.add_argument("--llm-config", default="configs/llm.yaml")
    ap.add_argument("--config", default="configs/retrieval.yaml", help="retrieval settings")
    ap.add_argument("--ingest-config", default="configs/ingest.yaml", help="document titles")
    ap.add_argument("--questions", default=None, help="override paths.questions")
    ap.add_argument(
        "--report", default=None, help="default: <results_dir>/tables/eval_construction.json"
    )
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (default 4)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    t_start = time.perf_counter()
    started = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    settings = load_retrieval_settings(resolve_path(Path(args.config)))
    set_torch_threads(args.threads)
    qpath = resolve_path(Path(args.questions) if args.questions else settings.paths.questions)
    report_path = (
        resolve_path(Path(args.report))
        if args.report
        else resolve_path(settings.paths.results_dir) / "tables" / "eval_construction.json"
    )
    questions, errors = load_questions(qpath) if qpath.exists() else ([], [])
    if errors:
        for e in errors:
            emit(f"ERROR: {e}")
        emit(f"fix {qpath} first (python scripts/validate_eval.py)")
        return 1
    n_auto = sum(1 for q in questions if q.origin == AUTO_ORIGIN)
    if n_auto and not args.rebuild:
        emit(
            f"{qpath} already contains {n_auto} {AUTO_ORIGIN} item(s); nothing was changed.\n"
            "  Pass --rebuild to replace them (human-written and reviewed items are kept)."
        )
        return EXIT_EXISTING_AUTO

    res = RetrievalResources(settings)
    nodes_path = res.processed_dir / "structure" / "nodes.jsonl"
    if not nodes_path.exists():
        emit(f"{nodes_path} not found; run scripts/ingest.py and scripts/chunk.py first")
        return 1
    nodes = load_nodes(nodes_path)
    xrefs = load_xrefs(res.processed_dir / "structure" / "xrefs.jsonl")
    titles = {n.doc_id: n.doc_title for n in nodes}
    ing = resolve_path(Path(args.ingest_config))
    if ing.exists():
        titles.update({d.doc_id: d.display_title for d in load_ingest_config(ing).documents})

    base_url: str | None = None
    if llm is None:
        cfg = load_llm_config(resolve_path(Path(args.llm_config)))
        client = LLMClient(cfg)
        if not client.is_reachable():
            emit(
                f"LLM endpoint {cfg.base_url} is not reachable. Start Ollama and run "
                f"`ollama pull {cfg.model}` (or set base_url/model in {args.llm_config})."
            )
            return 2
        llm, base_url = client, cfg.base_url
        emit(f"using {cfg.model} at {cfg.base_url}; LLM cache: {client.cache_dir}")
    else:
        emit(f"using injected model {llm.model}")
    kept_human = [q for q in questions if q.origin != AUTO_ORIGIN]
    emit(
        f"{qpath}: {len(kept_human)} existing non-automatic item(s) kept"
        + (f"; {n_auto} automatic item(s) will be replaced (--rebuild)" if n_auto else "")
    )

    builder = AutoEvalBuilder(
        llm,
        res,
        nodes,
        xrefs,
        titles,
        kept_human,
        seed=args.seed,
        dense_model=args.dense_model,
        out=emit,
    )
    try:
        result = builder.build(
            n_candidates=args.n_candidates,
            target_answerable=args.target_answerable,
            n_unanswerable_candidates=args.n_unanswerable_candidates,
            target_unanswerable=args.target_unanswerable,
        )
    finally:
        released = res.loaded_encoders()
        res.close()
        emit(f"released encoder(s) {', '.join(released) or '-'} | RSS {fmt_mb(rss_bytes())}")

    keep_lines, auto_lines = split_lines(qpath)
    if auto_lines and not args.rebuild:
        emit(f"{qpath} gained {AUTO_ORIGIN} items during the run; nothing written (use --rebuild)")
        return EXIT_EXISTING_AUTO
    node_ids = {n.node_id for n in nodes}
    for q in result.questions:
        problems = q.consistency_errors() + [
            f"unknown gold node {g}" for g in q.gold_node_ids if g not in node_ids
        ]
        if problems:
            emit(f"ERROR: generated item {q.id}: {'; '.join(problems)}; nothing written")
            return 1
    write_questions(qpath, keep_lines, result.questions)

    report = result.report
    new_q = result.questions
    report.update(
        {
            "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "started_utc": started,
            "base_url": base_url,
            "questions_file": str(qpath),
            "questions_sha256": hashlib.sha256(qpath.read_bytes()).hexdigest(),
            "rebuild": bool(args.rebuild),
            "replaced_auto_items": len(auto_lines),
            "kept_existing_items": len(keep_lines),
            "ids": {"first": new_q[0].id, "last": new_q[-1].id} if new_q else None,
            "final_counts": {
                "new_items": final_counts(new_q),
                "whole_file": final_counts(load_questions(qpath)[0]),
            },
            "wall_time_s": round(time.perf_counter() - t_start, 1),
            "peak_rss_mb": round(peak_rss_bytes() / 1024**2),
            "run_info": collect_run_info(),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    emit("== filters (in order): pass / in")
    for part in ("answerable", "unanswerable"):
        for f in report[part]["filters"]:
            reasons = ", ".join(f"{k}={v}" for k, v in f["fail_reasons"].items())
            emit(f"  {part[0]}:{f['name']:24s} {f['pass']:4d} / {f['in']:4d}  {reasons}")
    fc = report["final_counts"]["new_items"]
    emit(
        f"appended {len(new_q)} item(s) to {qpath}: {fc['answerable']} answerable "
        f"(target {args.target_answerable}) + {fc['unanswerable']} unanswerable "
        f"(target {args.target_unanswerable})"
    )
    emit("  by type: " + ", ".join(f"{k}={v}" for k, v in fc["by_type"].items()))
    emit(
        "  by split: "
        + ", ".join(
            f"{sp} {v['answerable']}+{v['unanswerable']}" for sp, v in fc["by_split"].items()
        )
    )
    calls = report["llm_calls"]
    emit(
        f"LLM calls: {calls['calls']} ({calls['cached']} from cache), "
        f"{fmt_duration(calls['latency_s'])} of model time"
    )
    emit(f"wrote {report_path}")
    emit(f"total wall time {fmt_duration(time.perf_counter() - t_start)}")
    emit(
        "Next: python scripts/validate_eval.py   (optional: python scripts/spot_check.py --n 20 --seed 0)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
