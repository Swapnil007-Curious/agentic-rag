"""Ingest the configured PDFs: extract, clean, parse structure, extract cross-references.

Usage:
    python scripts/ingest.py [--collection NAME] [--config FILE] [--raw-dir DIR]
                             [--processed-dir DIR] [--write-sources]
Outputs (under processed_dir):
    clean/<doc_id>.txt, clean/<doc_id>.pages.json
    structure/nodes.jsonl, xrefs.jsonl, footnotes.jsonl, structure_report.json
Textbook-format collections also (re)write <raw_dir>/SOURCES.csv (paths.sources_file; the cs_ai
app index writes SOURCES_app.csv, so the evaluation corpus's file is untouched) with title, authors, edition,
licence (read from the PDF's licence page when it states one), source URL, sha256, pages and
the chapters kept; each document line shows the pages kept, the resolved chapter ranges, the
nodes per level and the licence.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from ragsys.collections import add_collection_arguments, collection_from_args, config_path
from ragsys.config import load_ingest_config, resolve_path
from ragsys.ingest.pipeline import run_ingest
from ragsys.ingest.sources import write_sources_csv, write_textbook_sources_csv
from ragsys.utils.console import emit
from ragsys.utils.seed import set_seed


def _xref_cell(d: dict[str, Any]) -> str:
    x = d.get("xrefs", {})
    return (
        f"{x.get('resolved_to_node', 0)}/{x.get('document', 0)}/{x.get('external', 0)}/"
        f"{x.get('unresolved', 0)}"
    )


def print_textbook(doc_id: str, d: dict[str, Any]) -> None:
    levels = d["nodes_per_level"]
    emit(
        f"{doc_id:8s} pages kept {d['pages_kept']:3d} of {d['pages']:4d}  nodes={d['nodes_total']:4d} "
        f"(chapters {levels.get('chapter', 0)}, sections {levels.get('section', 0)}, "
        f"subsections {levels.get('subsection', 0)}, headings {levels.get('heading', 0)}, "
        f"code {levels.get('code', 0)})  xrefs node/doc/ext/unres={_xref_cell(d)}"
    )
    for r in d["chapter_ranges"]:
        where = (
            f"pp. {r['start']}-{r['end']} ({r['n_pages']} pages, {r['method']})"
            if r["found"]
            else "NOT FOUND"
        )
        emit(f"    chapter {r['title']!r}: {where}")
    lic = d["licence"]
    emit(f"    licence: {lic['licence']}  [{lic['source'][:140]}]")
    dropped = d.get("sections_dropped") or []
    if dropped:
        emit(f"    dropped sections: {len(dropped)} (e.g. {', '.join(dropped[:3])})")
    for w in d["warnings"]:
        emit(f"    WARNING: {w}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="default: the collection's ingest config")
    ap.add_argument("--raw-dir", default=None, help="override paths.raw_dir")
    ap.add_argument("--processed-dir", default=None, help="override paths.processed_dir")
    ap.add_argument("--write-sources", action="store_true", help="(re)write <raw_dir>/SOURCES.csv")
    add_collection_arguments(ap)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    coll = collection_from_args(args)
    cfg = load_ingest_config(resolve_path(config_path(args.config, coll.ingest_config)))
    if args.raw_dir:
        cfg.paths.raw_dir = Path(args.raw_dir)
    if args.processed_dir:
        cfg.paths.processed_dir = Path(args.processed_dir)
    raw_dir = resolve_path(cfg.paths.raw_dir)
    processed_dir = resolve_path(cfg.paths.processed_dir)
    set_seed(42)

    t0 = time.perf_counter()
    result = run_ingest(cfg, raw_dir, processed_dir, write=True)
    elapsed = time.perf_counter() - t0
    sources = raw_dir / cfg.paths.sources_file
    if cfg.sources_format == "textbook":
        rows = write_textbook_sources_csv(cfg, raw_dir, sources, result.report)
        emit(f"wrote {sources} ({len(rows)} rows)")
    elif args.write_sources:
        rows = write_sources_csv(cfg, raw_dir, sources)
        emit(f"wrote {sources} ({len(rows)} rows)")

    rep = result.report
    for s in rep["skipped"]:
        emit(f"SKIPPED {s['doc_id']} ({s['file']}): {s['reason']}")
    total_kept = 0
    for doc_id, d in rep["documents"].items():
        if d.get("doc_type") == "textbook":
            print_textbook(doc_id, d)
            total_kept += int(d["pages_kept"])
            continue
        num = d.get("numbering") or {}
        emit(
            f"{doc_id:17s} pages={d['pages']:3d} dropped_deva={len(d['pages_dropped_devanagari']):2d} "
            f"chapters={d['chapters']:2d} {num.get('unit', 'units')}s={d['sections_or_rules']:3d} "
            f"(max no. {num.get('max_number')}) definitions={d['definitions']:3d} "
            f"nodes={d['nodes_total']:4d} xrefs node/doc/ext/unres={_xref_cell(d)}"
        )
        ps = d.get("parser_stats") or {}
        if d.get("doc_type") == "constitution":
            emit(
                f"    parts={ps.get('parts')} chapters inside parts={ps.get('chapters_in_parts')} "
                f"articles={ps.get('articles')} (lettered {ps.get('lettered_articles')}) "
                f"schedules={d['schedules']} pages kept={d.get('pages_kept', '?')}"
            )
        elif d.get("pages_kept"):
            emit(f"    pages kept={d['pages_kept']} schedules={d['schedules']}")
        if d.get("pages_kept") and cfg.sources_format == "textbook":
            total_kept += int(d["pages_kept"])
        for w in d["warnings"]:
            if w.startswith(("duplicate label", "embedded table")):
                continue  # counted below
            emit(f"    WARNING: {w}")
        dup = sum(1 for w in d["warnings"] if w.startswith("duplicate label"))
        tab = sum(1 for w in d["warnings"] if w.startswith("embedded table"))
        if dup or tab:
            emit(
                f"    note: {dup} repeated label(s) stored with ~2 suffixes, {tab} embedded "
                "table(s) kept as plain text (details in structure_report.json)"
            )
    if total_kept:
        emit(f"pages kept in total: {total_kept}")
    emit(
        f"total nodes={rep['totals']['nodes']} xrefs={rep['totals']['xrefs']['total']} "
        f"footnotes={rep['totals']['footnotes']} in {elapsed:.1f}s"
    )
    emit(f"outputs in {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
