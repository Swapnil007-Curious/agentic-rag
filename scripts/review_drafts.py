"""Review drafted questions in the terminal, or add unanswerable questions by hand.

Usage:
    python scripts/review_drafts.py                       review pending drafts
    python scripts/review_drafts.py --add-unanswerable    type unanswerable questions
For each pending draft in data/eval/drafts.jsonl it shows the question, type, difficulty,
gold answer, gold node IDs, the lexical-overlap flag and the source text. Keys:
    a = accept   e = edit (blank keeps the current value; gold IDs are checked against nodes.jsonl)
    r = reject   s = skip   q = quit
Progress is saved after every action. Accepted drafts are appended to data/eval/questions.jsonl
with the next free id, origin "llm_draft_human_reviewed" and author "Swapnil".
--add-unanswerable asks for a question and the reason the corpus cannot answer it, shows the
top 5 BM25 hits so you can confirm that, then appends it (answerable false, origin "human").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragsys.config import resolve_path
from ragsys.eval.review import add_unanswerable, review_drafts
from ragsys.ingest.pipeline import load_nodes
from ragsys.retrieval.bm25 import BM25Index
from ragsys.retrieval.corpus import load_chunks
from ragsys.utils.console import ascii_safe, emit


def _out(line: str) -> None:
    emit(line)


def _input(prompt: str) -> str:
    return input(ascii_safe(prompt))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--drafts", default="data/eval/drafts.jsonl")
    ap.add_argument("--questions", default="data/eval/questions.jsonl")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--add-unanswerable", action="store_true")
    ap.add_argument("--chunker", default="structural", help="chunks searched by BM25")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    processed = resolve_path(Path(args.processed_dir))
    qpath = resolve_path(Path(args.questions))
    if args.add_unanswerable:
        corpus = load_chunks(processed, args.chunker)
        index = BM25Index(corpus.ids, corpus.texts)

        def search(query: str) -> list[tuple[str, str, str]]:
            hits = index.search(query, 5)
            return [
                (cid, corpus.by_id[cid].heading_path, corpus.by_id[cid].display_text[:240])
                for cid, _s in hits
            ]

        add_unanswerable(qpath, search, input_fn=_input, out=_out)
        return 0

    nodes_path = processed / "structure" / "nodes.jsonl"
    if not nodes_path.exists():
        emit(f"{nodes_path} not found; run scripts/ingest.py first")
        return 1
    node_ids = [n.node_id for n in load_nodes(nodes_path)]
    review_drafts(resolve_path(Path(args.drafts)), qpath, node_ids, input_fn=_input, out=_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
