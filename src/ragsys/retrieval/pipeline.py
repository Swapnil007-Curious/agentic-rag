"""The retriever: first stage (BM25 / dense / hybrid) -> optional cross-reference expansion ->
optional cross-encoder reranking, with full provenance and per-stage timings.

Ranking assembly
----------------
* ``window`` = the first ``rerank_n`` first-stage hits when reranking, else the first
  ``final_k``.
* xref expansion follows the top ``xref_top_m`` first-stage hits and adds up to
  ``xref_max_added`` chunks that are not already in the window.
* rerank on: window + expanded chunks are scored by the cross-encoder and sorted.
  rerank off: each expanded chunk is inserted directly after the chunk that referenced it.
* The remaining first-stage hits follow in first-stage order. ``results`` is the first
  ``final_k`` of this ranking; ``ranked`` keeps all of it (token-budget metrics use it).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ragsys.chunking.base import Chunk
from ragsys.config import resolve_path
from ragsys.ingest.pipeline import load_nodes, load_xrefs
from ragsys.retrieval.bm25 import BM25Index
from ragsys.retrieval.config import RetrievalConfig, RetrievalSettings
from ragsys.retrieval.corpus import ChunkCorpus, NodeGraph, load_chunks
from ragsys.retrieval.embed import CacheInfo, Encoder, SentenceTransformerEncoder, corpus_embeddings
from ragsys.retrieval.fakes import OverlapCrossEncoder, fake_encoder
from ragsys.retrieval.fusion import ComponentHit, FusedHit, hybrid_linear, rrf, single
from ragsys.retrieval.rerank import Reranker
from ragsys.retrieval.store import NumpyStore
from ragsys.retrieval.xref_expand import Expansion, XrefExpander
from ragsys.structure.xrefs import XRef

STAGES = ("bm25", "dense_encode", "dense_search", "fusion", "xref", "rerank", "total")


@dataclass
class RetrievedChunk:
    rank: int
    chunk: Chunk
    provenance: str  # "first_stage" or "xref_from:<chunk_id>"
    first_stage_rank: int | None = None
    first_stage_score: float | None = None
    components: dict[str, ComponentHit] = field(default_factory=dict)
    rerank_score: float | None = None
    xref_target: str | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk.chunk_id,
            "provenance": self.provenance,
            "xref_target": self.xref_target,
            "first_stage_rank": self.first_stage_rank,
            "first_stage_score": self.first_stage_score,
            "components": {
                k: {"rank": v.rank, "score": v.score} for k, v in self.components.items()
            },
            "rerank_score": self.rerank_score,
            "n_tokens": self.chunk.n_tokens,
            "section_ids": self.chunk.section_ids,
            "heading_path": self.chunk.heading_path,
        }


@dataclass
class RetrievalResult:
    query: str
    config_id: str
    ranked: list[RetrievedChunk]
    final_k: int
    timings_ms: dict[str, float]
    expansions: list[Expansion] = field(default_factory=list)

    @property
    def results(self) -> list[RetrievedChunk]:
        return self.ranked[: self.final_k]

    @property
    def top1_first_stage_score(self) -> float | None:
        fs = [r for r in self.ranked if r.first_stage_rank == 1]
        return fs[0].first_stage_score if fs else None

    @property
    def top1_rerank_score(self) -> float | None:
        return self.ranked[0].rerank_score if self.ranked else None


class RetrievalResources:
    """Lazily built, shared building blocks (chunks, node graph, BM25 indexes, encoders,
    corpus embeddings, rerankers), so a grid of retrievers reuses them."""

    def __init__(
        self,
        settings: RetrievalSettings,
        processed_dir: Path | None = None,
        emb_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.processed_dir = resolve_path(processed_dir or settings.paths.processed_dir)
        self.emb_dir = resolve_path(emb_dir or settings.paths.emb_dir)
        self._corpora: dict[str, ChunkCorpus] = {}
        self._graph: NodeGraph | None = None
        self._xrefs: list[XRef] | None = None
        self._bm25: dict[tuple[str, float, float, bool], BM25Index] = {}
        self._encoders: dict[tuple[str, bool], Encoder] = {}
        self._emb: dict[tuple[str, str], tuple[np.ndarray, CacheInfo]] = {}
        self._stores: dict[tuple[str, str], NumpyStore] = {}
        self._rerankers: dict[str, Reranker] = {}
        self._expanders: dict[str, XrefExpander] = {}

    def corpus(self, chunker: str) -> ChunkCorpus:
        if chunker not in self._corpora:
            self._corpora[chunker] = load_chunks(self.processed_dir, chunker)
        return self._corpora[chunker]

    def graph(self) -> NodeGraph:
        if self._graph is None:
            self._graph = NodeGraph(load_nodes(self.processed_dir / "structure" / "nodes.jsonl"))
        return self._graph

    def xrefs(self) -> list[XRef]:
        if self._xrefs is None:
            self._xrefs = load_xrefs(self.processed_dir / "structure" / "xrefs.jsonl")
        return self._xrefs

    def bm25(self, chunker: str, k1: float = 1.5, b: float = 0.75, stem: bool = False) -> BM25Index:
        key = (chunker, k1, b, stem)
        if key not in self._bm25:
            corpus = self.corpus(chunker)
            self._bm25[key] = BM25Index(corpus.ids, corpus.texts, k1=k1, b=b, stem=stem)
        return self._bm25[key]

    def encoder(self, model: str, query_instruction: bool = True) -> Encoder:
        name = self.settings.dense_model_name(model)
        key = (name, query_instruction)
        if key not in self._encoders:
            if name.startswith("fake:"):
                self._encoders[key] = fake_encoder(name)
            else:
                # one loaded model per name: the instruction flag only changes query text
                twin = next((e for (n, _q), e in self._encoders.items() if n == name), None)
                enc = SentenceTransformerEncoder(
                    name,
                    self.settings.device,
                    self.settings.embed_batch_size,
                    self.settings.query_instruction if query_instruction else None,
                )
                if isinstance(twin, SentenceTransformerEncoder):
                    enc._model = twin._model
                self._encoders[key] = enc
        return self._encoders[key]

    def embeddings(
        self, model: str, chunker: str, force: bool = False
    ) -> tuple[np.ndarray, CacheInfo]:
        name = self.settings.dense_model_name(model)
        key = (name, chunker)
        if key not in self._emb or force:
            self._emb[key] = corpus_embeddings(
                self.encoder(model), self.corpus(chunker), self.emb_dir, force=force
            )
        return self._emb[key]

    def vector_store(self, model: str, chunker: str) -> NumpyStore:
        name = self.settings.dense_model_name(model)
        key = (name, chunker)
        if key not in self._stores:
            emb, _info = self.embeddings(model, chunker)
            self._stores[key] = NumpyStore(self.corpus(chunker).ids, emb)
        return self._stores[key]

    def reranker(self, model: str) -> Reranker:
        name = self.settings.reranker_name(model)
        if name not in self._rerankers:
            scorer = OverlapCrossEncoder() if name == "fake:overlap" else None
            self._rerankers[name] = Reranker(
                name, self.settings.device, self.settings.rerank_batch_size, model=scorer
            )
        return self._rerankers[name]

    def expander(self, chunker: str) -> XrefExpander:
        if chunker not in self._expanders:
            self._expanders[chunker] = XrefExpander(
                self.corpus(chunker), self.graph(), self.xrefs()
            )
        return self._expanders[chunker]


class Retriever:
    def __init__(self, cfg: RetrievalConfig, resources: RetrievalResources) -> None:
        self.cfg = cfg
        self.config_id = cfg.config_id()
        self.corpus = resources.corpus(cfg.chunker)
        self.bm25 = (
            resources.bm25(cfg.chunker, cfg.bm25_k1, cfg.bm25_b, cfg.bm25_stem)
            if cfg.uses_bm25
            else None
        )
        if cfg.uses_dense:
            self.encoder: Encoder | None = resources.encoder(cfg.dense_model, cfg.query_instruction)
            self.store: NumpyStore | None = resources.vector_store(cfg.dense_model, cfg.chunker)
        else:
            self.encoder, self.store = None, None
        self.reranker = resources.reranker(cfg.rerank_model) if cfg.rerank else None
        self.expander = resources.expander(cfg.chunker) if cfg.xref_expand else None

    def warm_up(self) -> None:
        """Load lazily initialised models so the first timed query is not penalised."""
        self.search("warm up query")

    def first_stage(self, query: str, timings: dict[str, float]) -> list[FusedHit]:
        cfg = self.cfg
        k = cfg.first_stage_k
        lexical: list[tuple[str, float]] = []
        dense: list[tuple[str, float]] = []
        if self.bm25 is not None:
            t0 = time.perf_counter()
            lexical = self.bm25.search(query, k)
            timings["bm25"] = _ms(t0)
        if self.encoder is not None and self.store is not None:
            t0 = time.perf_counter()
            qvec = self.encoder.encode([query], is_query=True)[0]
            timings["dense_encode"] = _ms(t0)
            t0 = time.perf_counter()
            dense = self.store.search(qvec, k)
            timings["dense_search"] = _ms(t0)
        t0 = time.perf_counter()
        if cfg.retriever == "bm25":
            hits = single("bm25", lexical)
        elif cfg.retriever == "dense":
            hits = single("dense", dense)
        elif cfg.retriever == "hybrid_rrf":
            hits = rrf({"bm25": lexical, "dense": dense}, k=cfg.rrf_k)
        else:
            hits = hybrid_linear(lexical, dense, cfg.alpha)
        timings["fusion"] = _ms(t0)
        return hits[:k]

    def search(self, query: str) -> RetrievalResult:
        cfg = self.cfg
        timings: dict[str, float] = {}
        t_start = time.perf_counter()
        first = self.first_stage(query, timings)
        by_id = self.corpus.by_id

        def from_hit(h: FusedHit, rank: int) -> RetrievedChunk:
            return RetrievedChunk(0, by_id[h.chunk_id], "first_stage", rank, h.score, h.components)

        first_items = [from_hit(h, i) for i, h in enumerate(first, start=1)]
        window_n = cfg.rerank_n if cfg.rerank else cfg.final_k
        window = first_items[:window_n]
        window_ids = {r.chunk_id for r in window}

        expansions: list[Expansion] = []
        if self.expander is not None:
            t0 = time.perf_counter()
            expansions = self.expander.expand(
                [h.chunk_id for h in first],
                top_m=cfg.xref_top_m,
                max_added=cfg.xref_max_added,
                per_target=cfg.xref_per_target,
                exclude=window_ids,
            )
            timings["xref"] = _ms(t0)
        fs_lookup = {r.chunk_id: r for r in first_items}
        expanded_items = []
        for e in expansions:
            fs = fs_lookup.get(e.chunk_id)
            expanded_items.append(
                RetrievedChunk(
                    0,
                    by_id[e.chunk_id],
                    e.provenance,
                    fs.first_stage_rank if fs else None,
                    fs.first_stage_score if fs else None,
                    dict(fs.components) if fs else {},
                    xref_target=e.target_node,
                )
            )

        if self.reranker is not None:
            pool = window + expanded_items
            t0 = time.perf_counter()
            scores = self.reranker.score(query, [r.chunk.text for r in pool])
            timings["rerank"] = _ms(t0)
            for r, s in zip(pool, scores, strict=True):
                r.rerank_score = s
            head = [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-scores[i], i))]
        else:
            head = []
            pending = list(expanded_items)
            for r in window:
                head.append(r)
                head.extend(x for x in pending if x.provenance == f"xref_from:{r.chunk_id}")
                pending = [x for x in pending if x.provenance != f"xref_from:{r.chunk_id}"]
            head.extend(pending)
        head_ids = {r.chunk_id for r in head}
        ranked = head + [r for r in first_items[window_n:] if r.chunk_id not in head_ids]
        for i, r in enumerate(ranked, start=1):
            r.rank = i
        timings["total"] = _ms(t_start)
        return RetrievalResult(query, self.config_id, ranked, cfg.final_k, timings, expansions)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)
