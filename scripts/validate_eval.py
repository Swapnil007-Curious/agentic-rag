"""Validate data/eval/questions.jsonl against data/processed/structure/nodes.jsonl.

Usage:
    python scripts/validate_eval.py [--questions data/eval/questions.jsonl]
                                    [--nodes data/processed/structure/nodes.jsonl]
Checks the schema, unique ids, that every gold_node_id exists (suggesting the closest IDs
when not) and answerable/unanswerable consistency; prints counts by type, difficulty and
document plus progress toward 90 answerable + 30 unanswerable. Exit code 1 on errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragsys.config import resolve_path
from ragsys.eval.validate import format_report, validate_file
from ragsys.utils.console import emit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--questions", default="data/eval/questions.jsonl")
    ap.add_argument("--nodes", default="data/processed/structure/nodes.jsonl")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    qpath = resolve_path(Path(args.questions))
    npath = resolve_path(Path(args.nodes))
    for p, hint in ((qpath, ""), (npath, " (run scripts/ingest.py first)")):
        if not p.exists():
            emit(f"ERROR: {p} not found{hint}")
            return 1
    rep = validate_file(qpath, npath)
    emit(format_report(rep))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
