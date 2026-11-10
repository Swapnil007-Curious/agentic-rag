"""Structure-aware cross-reference expansion.

After first-stage retrieval, the top ``M`` chunks are followed along their *resolved*
cross-references (``exact``/``partial`` edges of xrefs.jsonl, i.e. the chunk's
``xref_targets``). For every target node, the chunk(s) that best cover it are added to the
candidate pool before reranking, tagged with provenance ``xref_from:<chunk_id>``. Nothing else
is added: there is no implicit expansion to definitions or neighbouring chunks.

Target nodes are mapped to chunks through ``Chunk.node_ids`` (a chunk lists every node it
overlaps), so the same code works for every chunker. Targets that are chapters, or that the
source chunk already overlaps, are skipped.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ragsys.retrieval.corpus import ChunkCorpus, NodeGraph
from ragsys.structure.xrefs import XRef


@dataclass(frozen=True)
class Expansion:
    chunk_id: str  # the added chunk
    source_chunk_id: str  # the first-stage chunk whose reference was followed
    target_node: str  # the referenced node the added chunk covers

    @property
    def provenance(self) -> str:
        return f"xref_from:{self.source_chunk_id}"


class XrefExpander:
    def __init__(self, corpus: ChunkCorpus, graph: NodeGraph, xrefs: Iterable[XRef] = ()) -> None:
        self.corpus = corpus
        self.graph = graph
        self._node_chunks: dict[str, list[str]] = defaultdict(list)
        for c in corpus.chunks:
            for nid in c.node_ids:
                self._node_chunks[nid].append(c.chunk_id)
        # order of first mention of each target inside each chunk (xrefs carry char offsets)
        by_doc: dict[str, list[XRef]] = defaultdict(list)
        for x in xrefs:
            if x.target_node and x.resolution in ("exact", "partial"):
                by_doc[x.source_doc].append(x)
        for refs in by_doc.values():
            refs.sort(key=lambda x: x.char_start)
        self._targets: dict[str, list[str]] = {}
        for c in corpus.chunks:
            allowed = set(c.xref_targets)
            ordered: list[str] = []
            for x in by_doc.get(c.doc_id, []):
                if x.char_start >= c.char_end:
                    break
                if (
                    x.char_start >= c.char_start
                    and x.target_node in allowed
                    and x.target_node not in ordered
                ):
                    ordered.append(x.target_node)
            ordered.extend(t for t in c.xref_targets if t not in ordered)
            self._targets[c.chunk_id] = ordered

    def targets_of(self, chunk_id: str) -> list[str]:
        """Resolved targets worth following, in order of first mention in the chunk."""
        chunk = self.corpus.by_id[chunk_id]
        own = set(chunk.node_ids)
        out = []
        for t in self._targets.get(chunk_id, []):
            if t not in self.graph or self.graph.level(t) == "chapter" or t in own:
                continue
            out.append(t)
        return out

    def chunks_for_target(self, node_id: str, limit: int = 1) -> list[str]:
        """Chunks covering ``node_id``, best first: the largest share of the smaller of the two
        spans, then the chunk containing the start of the node, then document order."""
        if node_id not in self.graph:
            return []
        node = self.graph.nodes[node_id]
        ts, te = node.char_start, node.char_end
        scored = []
        for cid in self._node_chunks.get(node_id, []):
            c = self.corpus.by_id[cid]
            if c.doc_id != node.doc_id:
                continue
            overlap = min(c.char_end, te) - max(c.char_start, ts)
            if overlap <= 0:
                continue
            share = overlap / max(1, min(te - ts, c.char_end - c.char_start))
            contains_start = c.char_start <= ts < c.char_end
            scored.append((-share, not contains_start, self.corpus.position[cid], cid))
        scored.sort()
        return [cid for *_rest, cid in scored[:limit]]

    def expand(
        self,
        ranked_ids: Sequence[str],
        top_m: int = 5,
        max_added: int = 10,
        per_target: int = 1,
        exclude: Iterable[str] = (),
    ) -> list[Expansion]:
        """Follow the references of the first ``top_m`` chunks of ``ranked_ids``.

        Chunks in ``exclude`` (already in the candidate pool) and duplicates are not added;
        at most ``max_added`` chunks are returned, in source-rank then mention order.
        """
        seen = set(exclude) | set(ranked_ids[:top_m])
        out: list[Expansion] = []
        for src in ranked_ids[:top_m]:
            if src not in self.corpus.by_id:
                continue
            for target in self.targets_of(src):
                for cid in self.chunks_for_target(target, per_target):
                    if len(out) >= max_added:
                        return out
                    if cid in seen:
                        continue
                    seen.add(cid)
                    out.append(Expansion(cid, src, target))
        return out
