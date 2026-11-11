"""Interactive search over the chunked corpus.

Usage:
    python scripts/search.py "Who can file a complaint about defective goods?"
    python scripts/search.py "..." --config-id "structural|hybrid_rrf|bge-base|xref|rerank-minilm"
    python scripts/search.py "..." --chunker fixed_512 --retriever dense --model bge-base --no-xref --no-rerank
Without flags the `defaults` of configs/retrieval.yaml are used. Prints, per result: rank,
chunk_id, section_ids, heading_path, per-component ranks/scores, rerank score, provenance
(first stage or "xref_from:<chunk>"), and the first 300 characters of display_text.
Quote config ids in PowerShell (they contain '|').
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any

from ragsys.collections import add_collection_arguments, collection_from_args, config_path
from ragsys.config import resolve_path
from ragsys.retrieval.config import (
    RETRIEVER_TYPES,
    RetrievalConfig,
    load_retrieval_settings,
    parse_config_id,
)
from ragsys.retrieval.pipeline import RetrievalResources, Retriever
from ragsys.utils.console import emit
from ragsys.utils.seed import set_seed


def build_config(args: argparse.Namespace, defaults: RetrievalConfig) -> RetrievalConfig:
    if args.config_id:
        # the id fixes chunker / retriever / model / xref / rerank; depths etc. keep defaults
        parsed = parse_config_id(args.config_id)
        data = {**defaults.model_dump(), "xref_expand": False, "rerank": False}
        data.update(parsed.model_dump(include=set(parsed.model_fields_set)))
        cfg = RetrievalConfig.model_validate(data)
    else:
        cfg = defaults.model_copy()
    over: dict[str, Any] = {}
    for flag, field in (
        ("chunker", "chunker"),
        ("retriever", "retriever"),
        ("model", "dense_model"),
        ("rrf_k", "rrf_k"),
        ("alpha", "alpha"),
        ("first_stage_k", "first_stage_k"),
        ("k", "final_k"),
        ("rerank_model", "rerank_model"),
        ("rerank_n", "rerank_n"),
    ):
        v = getattr(args, flag)
        if v is not None:
            over[field] = v
    if args.xref is not None:
        over["xref_expand"] = args.xref
    if args.rerank is not None:
        over["rerank"] = args.rerank
    return RetrievalConfig.model_validate({**cfg.model_dump(), **over})


def fmt_components(comps: dict[str, Any]) -> str:
    parts = []
    for name in ("bm25", "dense"):
        if name in comps:
            parts.append(f"{name} #{comps[name].rank} ({comps[name].score:.4f})")
    return "  ".join(parts) if parts else "-"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("query", nargs="+")
    ap.add_argument("--config", default=None, help="default: the collection's retrieval config")
    add_collection_arguments(ap)
    ap.add_argument("--config-id", default=None, help='e.g. "fixed_512|dense|bge-base"')
    ap.add_argument("--chunker", default=None)
    ap.add_argument("--retriever", choices=RETRIEVER_TYPES, default=None)
    ap.add_argument("--model", default=None, help="dense model key, e.g. bge-base")
    ap.add_argument("--rrf-k", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--first-stage-k", type=int, default=None)
    ap.add_argument("-k", "--k", type=int, default=None, help="number of results (final_k)")
    ap.add_argument("--xref", dest="xref", action="store_true", default=None)
    ap.add_argument("--no-xref", dest="xref", action="store_false")
    ap.add_argument("--rerank", dest="rerank", action="store_true", default=None)
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--rerank-model", default=None)
    ap.add_argument("--rerank-n", type=int, default=None)
    ap.add_argument("--processed-dir", default=None)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    coll = collection_from_args(args)
    settings = load_retrieval_settings(
        resolve_path(config_path(args.config, coll.retrieval_config))
    )
    set_seed(settings.seed)
    cfg = build_config(args, settings.defaults)
    res = RetrievalResources(
        settings, processed_dir=Path(args.processed_dir) if args.processed_dir else None
    )
    retriever = Retriever(cfg, res)
    retriever.warm_up()
    query = " ".join(args.query)
    result = retriever.search(query)

    emit(f"config: {cfg.config_id()}")
    emit(f"query:  {query}")
    emit(
        "timings (ms): "
        + " | ".join(f"{k} {v:.1f}" for k, v in result.timings_ms.items())
        + f"   ({len(result.expansions)} chunk(s) added by xref expansion)"
    )
    for r in result.results:
        c = r.chunk
        emit("-" * 100)
        fs = (
            f"first-stage #{r.first_stage_rank} ({r.first_stage_score:.4f})"
            if r.first_stage_rank is not None and r.first_stage_score is not None
            else "first-stage: not retrieved"
        )
        rr = f"rerank={r.rerank_score:.3f}  " if r.rerank_score is not None else ""
        emit(f"#{r.rank:<2d} {c.chunk_id}   {rr}{fs}   {fmt_components(r.components)}")
        prov = r.provenance + (f" -> {r.xref_target}" if r.xref_target else "")
        emit(f"    provenance: {prov}   tokens: {c.n_tokens}   pages {c.page_start}-{c.page_end}")
        emit(f"    sections:   {', '.join(c.section_ids)}")
        emit(f"    path:       {c.heading_path}")
        snippet = " ".join(c.display_text[:300].split())
        emit(textwrap.fill(snippet, width=100, initial_indent="    ", subsequent_indent="    "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
