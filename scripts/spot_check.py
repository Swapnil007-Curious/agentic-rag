"""Optional spot check of the automatically generated evaluation items.

Usage:
    python scripts/spot_check.py --n 20 --seed 0       judge a random sample (interactive)
    python scripts/spot_check.py --apply-removals      remove the items marked wrong

Session: shows --n random llm_auto_filtered items from questions.jsonl (stratified answerable /
unanswerable, seeded). Answerable items are shown with the text of their gold nodes; unanswerable
items with the top-5 passages retrieved during construction (hybrid_rrf, bm25 + bge-small,
structural chunks). Keys: y = correct, n = wrong, s = skip, q = quit (what was judged is kept).
Writes results/tables/eval_spot_check.json: counts, error rate among judged items with a 95%
Wilson interval (overall and per stratum) and the ids marked wrong. questions.jsonl is not
changed.

--apply-removals (separate, explicit step): reads eval_spot_check.json and removes the ids
marked wrong from questions.jsonl. Only llm_auto_filtered items are ever removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path

from ragsys.collections import add_collection_arguments, collection_from_args, config_path
from ragsys.config import resolve_path
from ragsys.eval.schema import EvalQuestion
from ragsys.eval.spot_check import remove_auto_items, run_session, sample_items, summarise
from ragsys.eval.validate import load_questions
from ragsys.ingest.pipeline import load_nodes
from ragsys.retrieval.bm25 import BM25Index
from ragsys.retrieval.config import load_retrieval_settings
from ragsys.retrieval.corpus import load_chunks
from ragsys.utils.console import ascii_safe, emit

SOURCE_CHARS = 2500
PASSAGE_CHARS = 700


def _wrap(text: str, indent: str = "    ") -> list[str]:
    out: list[str] = []
    for para in text.splitlines():
        if para.strip():
            out.extend(
                textwrap.wrap(para, width=100, initial_indent=indent, subsequent_indent=indent)
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help="default: the collection's retrieval config")
    ap.add_argument("--questions", default=None, help="override paths.questions")
    add_collection_arguments(ap)
    ap.add_argument(
        "--out", default=None, help="default: <results_dir>/tables/eval_spot_check.json"
    )
    ap.add_argument(
        "--apply-removals",
        action="store_true",
        help="remove the items marked wrong in the saved spot check (no session)",
    )
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    coll = collection_from_args(args)
    settings = load_retrieval_settings(
        resolve_path(config_path(args.config, coll.retrieval_config))
    )
    qpath = resolve_path(Path(args.questions) if args.questions else settings.paths.questions)
    out_path = (
        resolve_path(Path(args.out))
        if args.out
        else resolve_path(settings.paths.results_dir) / "tables" / "eval_spot_check.json"
    )

    if args.apply_removals:
        if not out_path.exists():
            emit(f"{out_path} not found; run the spot check first (without --apply-removals)")
            return 1
        result = json.loads(out_path.read_text(encoding="utf-8"))
        wrong = list(result.get("ids_wrong", []))
        texts = {w["id"]: w.get("question") for w in result.get("wrong_items", [])}
        removed = remove_auto_items(qpath, {i: texts.get(i) for i in wrong}) if wrong else []
        result.update(removals_applied=True, removed_ids=removed)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        emit(
            f"removed {len(removed)} item(s) marked wrong from {qpath}: {', '.join(removed) or '-'}"
        )
        not_removed = sorted(set(wrong) - set(removed))
        if not_removed:
            emit(
                "not removed (not found, not automatic, or the question text changed since the "
                f"spot check): {', '.join(not_removed)}"
            )
        emit("Run python scripts/validate_eval.py to check the set.")
        return 0

    questions, errors = load_questions(qpath)
    if errors:
        for e in errors:
            emit(f"ERROR: {e}")
        return 1
    items = sample_items(questions, args.n, args.seed)
    if not items:
        emit(f"no llm_auto_filtered items in {qpath}; run scripts/build_auto_eval.py first")
        return 1
    processed = resolve_path(settings.paths.processed_dir)
    nodes = {n.node_id: n for n in load_nodes(processed / "structure" / "nodes.jsonl")}
    chunks = load_chunks(processed, "structural")
    bm25: BM25Index | None = None

    def render(q: EvalQuestion) -> list[str]:
        nonlocal bm25
        meta = q.gen_meta or {}
        lines = [f"question:    {q.question}"]
        if q.answerable:
            lines.append(f"gold answer: {q.gold_answer}")
            lines.append(f"gold nodes:  {', '.join(q.gold_node_ids)}")
            if meta.get("original_question"):
                lines.append(f"(paraphrased; original: {meta['original_question']})")
            lines.append("source text:")
            for g in q.gold_node_ids:
                node = nodes.get(g)
                lines.append(f"  [{g}]")
                lines += _wrap(node.text[:SOURCE_CHARS] if node else "(node not found)")
            lines.append(
                "CHECK: is the question answerable from this text, and is the gold answer correct?"
            )
            return lines
        lines.append(f"reason:      {q.gold_answer}")
        extra = meta.get("target") or meta.get("topic") or ""
        lines.append(f"method:      {meta.get('method', '?')} {('- ' + extra) if extra else ''}")
        ids = [c for c in meta.get("retrieved_chunk_ids", []) if c in chunks.by_id]
        label = f"top-{len(ids)} passages retrieved at construction ({meta.get('retrieval_config', '')})"
        if not ids:
            if bm25 is None:
                bm25 = BM25Index(chunks.ids, chunks.texts)
            ids = [cid for cid, _s in bm25.search(q.question, 5)]
            label = "BM25 top-5 passages (no retrieval stored for this item)"
        lines.append(label + ":")
        for i, cid in enumerate(ids, start=1):
            c = chunks.by_id[cid]
            lines.append(f"  {i}. {cid}  {c.heading_path}")
            lines += _wrap(c.display_text[:PASSAGE_CHARS], "       ")
        lines.append("CHECK: is it true that the corpus does NOT answer this question?")
        return lines

    def _input(prompt: str) -> str:
        return input(ascii_safe(prompt))

    judgments = run_session(items, render, input_fn=_input, out=emit)
    result = summarise(
        items,
        judgments,
        questions_file=str(qpath),
        questions_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest(),
        seed=args.seed,
        n_requested=args.n,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    c = result["counts"]
    ci = result["wilson_95"]
    rate = "n/a" if result["error_rate"] is None else f"{result['error_rate']:.1%}"
    ci_s = "n/a" if ci is None else f"[{ci[0]:.1%}, {ci[1]:.1%}]"
    emit(
        f"judged {result['judged']}: correct {c['correct']}, wrong {c['wrong']}, skipped "
        f"{c['skipped']}, not judged {c['not_judged']}; error rate {rate} (95% Wilson {ci_s})"
    )
    emit(f"wrote {out_path}")
    if result["ids_wrong"]:
        emit(
            f"marked wrong: {', '.join(result['ids_wrong'])}. To remove them from {qpath.name}: "
            "python scripts/spot_check.py --apply-removals"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
