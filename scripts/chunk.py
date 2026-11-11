"""Chunk the ingested corpus.

Usage:
    python scripts/chunk.py --chunker {fixed_256,fixed_512,semantic,structural,all}
                            [--collection NAME] [--config FILE] [--processed-dir DIR]
"all" runs the chunkers of the collection's chunking config (cs_ai: fixed_512, structural).
Writes processed_dir/chunks/<chunker>.jsonl and processed_dir/chunks/chunk_report.json.
Run scripts/ingest.py first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragsys.chunking.pipeline import chunker_names, load_corpus, run_chunker, update_report
from ragsys.collections import add_collection_arguments, collection_from_args, config_path
from ragsys.config import load_chunking_config, resolve_path
from ragsys.utils.seed import set_seed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--chunker", required=True, help="fixed_256 | fixed_512 | semantic | structural | all"
    )
    ap.add_argument("--config", default=None, help="default: the collection's chunking config")
    add_collection_arguments(ap)
    ap.add_argument("--processed-dir", default=None, help="override paths.processed_dir")
    ap.add_argument("--cache-dir", default=None, help="override paths.cache_dir")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    coll = collection_from_args(args)
    cfg = load_chunking_config(resolve_path(config_path(args.config, coll.chunking_config)))
    if args.processed_dir:
        cfg.paths.processed_dir = Path(args.processed_dir)
    if args.cache_dir:
        cfg.paths.cache_dir = Path(args.cache_dir)
    names = chunker_names(cfg)
    selected = names if args.chunker == "all" else [args.chunker]
    unknown = [n for n in selected if n not in names]
    if unknown:
        ap.error(f"unknown chunker(s) {unknown}; choose from {[*names, 'all']}")
    set_seed(cfg.seed)

    processed = resolve_path(cfg.paths.processed_dir)
    if not (processed / "structure" / "nodes.jsonl").exists():
        print(f"no nodes.jsonl under {processed}; run scripts/ingest.py first", file=sys.stderr)
        return 2
    docs = load_corpus(processed)
    out_dir = processed / "chunks"
    results = {}
    for name in selected:
        stats = run_chunker(name, cfg, docs, out_dir)
        results[name] = stats
        t = stats["tokens"]
        print(
            f"{name:11s} chunks={stats['count']:4d} tokens min/med/p95/max="
            f"{t['min']}/{t['median']:.0f}/{t['p95']:.0f}/{t['max']} "
            f"crossing_sections={stats['chunks_crossing_section_boundaries']:3d} "
            f"sections_split={stats['fraction_sections_split']:.3f} time={stats['wall_time_s']}s"
        )
    update_report(out_dir / "chunk_report.json", results, cfg)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
