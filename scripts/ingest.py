"""Ingest the configured PDFs: extract, clean, parse structure, extract cross-references.

Usage:
    python scripts/ingest.py [--config configs/ingest.yaml] [--raw-dir DIR] [--processed-dir DIR]
                             [--write-sources]
Outputs (under processed_dir):
    clean/<doc_id>.txt, clean/<doc_id>.pages.json
    structure/nodes.jsonl, xrefs.jsonl, footnotes.jsonl, structure_report.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ragsys.config import load_ingest_config, resolve_path
from ragsys.ingest.pipeline import run_ingest
from ragsys.ingest.sources import write_sources_csv
from ragsys.utils.seed import set_seed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/ingest.yaml")
    ap.add_argument("--raw-dir", default=None, help="override paths.raw_dir")
    ap.add_argument("--processed-dir", default=None, help="override paths.processed_dir")
    ap.add_argument("--write-sources", action="store_true", help="(re)write data/raw/SOURCES.csv")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    cfg = load_ingest_config(resolve_path(Path(args.config)))
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
    if args.write_sources:
        rows = write_sources_csv(cfg, raw_dir, raw_dir / "SOURCES.csv")
        print(f"wrote {raw_dir / 'SOURCES.csv'} ({len(rows)} rows)")

    rep = result.report
    for s in rep["skipped"]:
        print(f"SKIPPED {s['doc_id']} ({s['file']}): {s['reason']}")
    for doc_id, d in rep["documents"].items():
        num = d.get("numbering") or {}
        x = d.get("xrefs", {})
        print(
            f"{doc_id:17s} pages={d['pages']:3d} dropped_deva={len(d['pages_dropped_devanagari']):2d} "
            f"chapters={d['chapters']:2d} {num.get('unit', 'units')}s={d['sections_or_rules']:3d} "
            f"(max no. {num.get('max_number')}) definitions={d['definitions']:3d} "
            f"nodes={d['nodes_total']:4d} xrefs node/doc/ext/unres="
            f"{x.get('resolved_to_node', 0)}/{x.get('document', 0)}/{x.get('external', 0)}/{x.get('unresolved', 0)}"
        )
        for w in d["warnings"]:
            print(f"    WARNING: {w}")
    print(
        f"total nodes={rep['totals']['nodes']} xrefs={rep['totals']['xrefs']['total']} "
        f"footnotes={rep['totals']['footnotes']} in {elapsed:.1f}s"
    )
    print(f"outputs in {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
