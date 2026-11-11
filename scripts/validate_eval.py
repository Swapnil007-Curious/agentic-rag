"""Validate data/eval/questions.jsonl against data/processed/structure/nodes.jsonl.

Usage:
    python scripts/validate_eval.py [--collection NAME] [--questions FILE] [--nodes FILE]
Checks the schema, unique ids, that every gold_node_id exists (suggesting the closest IDs
when not) and answerable/unanswerable consistency; prints counts by type, difficulty and
document plus progress toward the collection's targets (consumer_law: 90 answerable + 30
unanswerable; cs_ai: 60 + 20). Exit code 1 on errors.
"""

from __future__ import annotations

import argparse
import sys

from ragsys.collections import add_collection_arguments, collection_from_args, config_path
from ragsys.config import resolve_path
from ragsys.eval.validate import format_report, validate_file
from ragsys.profiles import get_profile
from ragsys.utils.console import emit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--questions", default=None, help="default: <eval_dir>/questions.jsonl")
    ap.add_argument("--nodes", default=None, help="default: <processed_dir>/structure/nodes.jsonl")
    add_collection_arguments(ap)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    coll = collection_from_args(args)
    qpath = resolve_path(config_path(args.questions, coll.questions_file))
    npath = resolve_path(config_path(args.nodes, coll.nodes_file))
    for p, hint in ((qpath, ""), (npath, " (run scripts/ingest.py first)")):
        if not p.exists():
            emit(f"ERROR: {p} not found{hint}")
            return 1
    rep = validate_file(qpath, npath, get_profile(coll.profile).targets)
    emit(format_report(rep))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
