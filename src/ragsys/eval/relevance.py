"""Chunk relevance with respect to gold nodes.

A chunk *covers* a node when it contains part of that node's own text (the text of the node
that is not inside one of its children; whitespace ignored). Chapters are never covered.
A chunk is relevant to a gold node ``g`` if some node it covers is

* ``g`` itself,
* a descendant of ``g`` (the chunk holds part of ``g``, e.g. clause (a) of gold sub-section (1)),
* or an ancestor of ``g`` below chapter level (the chunk holds the lead-in of the provision that
  contains ``g``).

Ancestry comes from the parent links of nodes.jsonl, never from ID prefixes (IDs contain
``~2``, ``-prov1``, ``-expl1``). Using own text rather than plain overlap keeps a chunk that
merely sits inside the same section (say definition (15) for a question about definition (16))
from counting as relevant. Unanswerable questions have no relevant chunks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ragsys.retrieval.corpus import ChunkCorpus, NodeGraph


class RelevanceJudge:
    def __init__(self, graph: NodeGraph, corpus: ChunkCorpus) -> None:
        self.graph = graph
        self.corpus = corpus
        self._covered: dict[str, frozenset[str]] = {}
        self._gold_cache: dict[str, frozenset[str]] = {}

    def covered_nodes(self, chunk_id: str) -> frozenset[str]:
        hit = self._covered.get(chunk_id)
        if hit is not None:
            return hit
        c = self.corpus.by_id[chunk_id]
        out: set[str] = set()
        for nid in c.node_ids:
            if nid not in self.graph or self.graph.level(nid) == "chapter":
                continue
            node = self.graph.nodes[nid]
            if node.doc_id != c.doc_id:
                continue
            if any(a < c.char_end and b > c.char_start for a, b in self.graph.own_spans(nid)):
                out.add(nid)
        res = frozenset(out)
        self._covered[chunk_id] = res
        return res

    def matching_nodes(self, gold: str) -> frozenset[str]:
        """Nodes whose coverage makes a chunk relevant to ``gold``."""
        hit = self._gold_cache.get(gold)
        if hit is not None:
            return hit
        if gold not in self.graph:
            res: frozenset[str] = frozenset()
        else:
            ups = {a for a in self.graph.ancestors(gold) if self.graph.level(a) != "chapter"}
            res = frozenset({gold} | self.graph.descendants(gold) | ups)
        self._gold_cache[gold] = res
        return res

    def relevant_golds(self, chunk_id: str, golds: Iterable[str]) -> set[str]:
        """The gold nodes this chunk is relevant to."""
        cov = self.covered_nodes(chunk_id)
        return {g for g in golds if cov & self.matching_nodes(g)}

    def is_relevant(self, chunk_id: str, golds: Iterable[str]) -> bool:
        return bool(self.relevant_golds(chunk_id, golds))

    def profile(self, ranked_ids: Sequence[str], golds: Sequence[str]) -> list[set[str]]:
        """Per ranked position, the set of gold nodes that chunk is relevant to."""
        return [self.relevant_golds(cid, golds) for cid in ranked_ids]

    def n_relevant_chunks(self, golds: Sequence[str]) -> int:
        """Number of chunks in the whole corpus relevant to any gold node (for ideal DCG)."""
        if not golds:
            return 0
        return sum(1 for c in self.corpus.chunks if self.is_relevant(c.chunk_id, golds))
