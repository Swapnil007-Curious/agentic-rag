"""Run chunkers over the ingested corpus and summarise them in chunk_report.json."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ragsys.chunking.base import Chunk, Chunker, DocInput
from ragsys.chunking.fixed import FixedChunker
from ragsys.chunking.semantic import Embedder, SemanticChunker
from ragsys.chunking.structural import StructuralChunker
from ragsys.chunking.tokenizer import get_tokenizer
from ragsys.config import ChunkingConfig, resolve_path
from ragsys.ingest.pipeline import load_nodes, load_page_spans, load_xrefs
from ragsys.utils.runinfo import collect_run_info


def load_corpus(processed_dir: Path) -> list[DocInput]:
    structure = processed_dir / "structure"
    clean = processed_dir / "clean"
    nodes = load_nodes(structure / "nodes.jsonl")
    xrefs = load_xrefs(structure / "xrefs.jsonl")
    by_doc: dict[str, list[Any]] = defaultdict(list)
    for n in nodes:
        by_doc[n.doc_id].append(n)
    x_by_doc: dict[str, list[Any]] = defaultdict(list)
    for x in xrefs:
        x_by_doc[x.source_doc].append(x)
    docs = []
    for doc_id, doc_nodes in by_doc.items():
        first = doc_nodes[0]
        docs.append(
            DocInput(
                doc_id=doc_id,
                title=first.doc_title,
                doc_type=first.doc_type,
                text=(clean / f"{doc_id}.txt").read_text(encoding="utf-8"),
                page_spans=load_page_spans(clean / f"{doc_id}.pages.json"),
                nodes=doc_nodes,
                xrefs=x_by_doc.get(doc_id, []),
            )
        )
    return docs


def all_chunker_names(cfg: ChunkingConfig) -> list[str]:
    return [f.name for f in cfg.fixed] + ["semantic", "structural"]


def chunker_names(cfg: ChunkingConfig) -> list[str]:
    """Chunkers of this configuration (``chunkers`` when set, else every chunker)."""
    names = all_chunker_names(cfg)
    if cfg.chunkers is None:
        return names
    unknown = [c for c in cfg.chunkers if c not in names]
    if unknown:
        raise ValueError(f"unknown chunker(s) {unknown} in chunkers; known: {names}")
    return list(cfg.chunkers)


def build_chunker(name: str, cfg: ChunkingConfig) -> Chunker:
    tok = get_tokenizer(cfg.tokenizer_model)
    for f in cfg.fixed:
        if f.name == name:
            return FixedChunker(f.name, f.size, f.overlap, tok)
    if name == "semantic":
        s = cfg.semantic
        embedder = Embedder(
            s.embedding_model, s.device, s.batch_size, resolve_path(cfg.paths.cache_dir)
        )
        return SemanticChunker(tok, embedder, s.breakpoint_percentile, s.min_tokens, s.max_tokens)
    if name == "structural":
        st = cfg.structural
        return StructuralChunker(
            tok, st.max_tokens, st.min_tokens, st.contextual_header, orphan_text=st.orphan_text
        )
    raise ValueError(f"unknown chunker {name!r}; choose from {chunker_names(cfg)}")


def chunk_stats(chunks: list[Chunk], docs: list[DocInput]) -> dict[str, Any]:
    tokens = np.array([c.n_tokens for c in chunks]) if chunks else np.zeros(0)
    unit_ids = {n.node_id for d in docs for n in d.nodes if n.is_unit}
    seen: Counter[str] = Counter()
    for c in chunks:
        seen.update(set(c.section_ids))
    split_units = sum(1 for u in unit_ids if seen[u] > 1)

    def q(p: float) -> float | None:
        return float(np.percentile(tokens, p)) if tokens.size else None

    return {
        "count": len(chunks),
        "per_document": dict(Counter(c.doc_id for c in chunks)),
        "tokens": {
            "min": int(tokens.min()) if tokens.size else None,
            "median": q(50),
            "p95": q(95),
            "max": int(tokens.max()) if tokens.size else None,
            "mean": float(tokens.mean()) if tokens.size else None,
        },
        "chunks_crossing_section_boundaries": sum(1 for c in chunks if len(c.section_ids) > 1),
        "section_units_total": len(unit_ids),
        "section_units_split_across_chunks": split_units,
        "fraction_sections_split": (split_units / len(unit_ids)) if unit_ids else 0.0,
        "section_units_not_covered": sorted(u for u in unit_ids if seen[u] == 0)[:20],
    }


def run_chunker(
    name: str, cfg: ChunkingConfig, docs: list[DocInput], out_dir: Path
) -> dict[str, Any]:
    chunker = build_chunker(name, cfg)
    t0 = time.perf_counter()
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunker.chunk_document(doc))
    wall = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")
    stats = chunk_stats(chunks, docs)
    stats["wall_time_s"] = round(wall, 2)
    if isinstance(chunker, SemanticChunker) and isinstance(chunker.embed, Embedder):
        stats["embeddings"] = dict(chunker.embed.stats)
    return stats


def update_report(
    report_path: Path, results: dict[str, dict[str, Any]], cfg: ChunkingConfig
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
    report.setdefault("chunkers", {}).update(results)
    report["config"] = cfg.model_dump(mode="json")
    report["run_info"] = collect_run_info()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
