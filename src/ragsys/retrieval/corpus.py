"""Loading chunks and the node tree (parent/child links from nodes.jsonl) for retrieval."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ragsys.chunking.base import Chunk
from ragsys.structure.base import Node


@dataclass
class ChunkCorpus:
    """All chunks of one chunker, in file order (the order defines embedding rows)."""

    chunker: str
    chunks: list[Chunk]
    by_id: dict[str, Chunk] = field(init=False)
    position: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.position = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        if len(self.by_id) != len(self.chunks):
            raise ValueError(f"duplicate chunk ids in chunker {self.chunker!r}")

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def texts_hash(self) -> str:
        return texts_hash(self.texts)


def texts_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def load_chunks(processed_dir: Path, chunker: str) -> ChunkCorpus:
    path = processed_dir / "chunks" / f"{chunker}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/chunk.py --chunker {chunker}")
    with path.open(encoding="utf-8") as fh:
        chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]
    return ChunkCorpus(chunker, chunks)


class NodeGraph:
    """Parent/child links of all nodes, plus each node's *own* text spans (the parts of the
    node that are not inside any child), trimmed of whitespace."""

    def __init__(self, nodes: list[Node]) -> None:
        self.nodes = {n.node_id: n for n in nodes}
        self._children: dict[str, list[str]] = defaultdict(list)
        for n in sorted(nodes, key=lambda x: (x.doc_id, x.order)):
            if n.parent_id and n.parent_id in self.nodes:
                self._children[n.parent_id].append(n.node_id)
        self._own: dict[str, list[tuple[int, int]]] = {}

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.nodes

    def level(self, node_id: str) -> str:
        return self.nodes[node_id].level

    def parent(self, node_id: str) -> str | None:
        p = self.nodes[node_id].parent_id
        return p if p in self.nodes else None

    def children(self, node_id: str) -> list[str]:
        return self._children.get(node_id, [])

    def ancestors(self, node_id: str) -> list[str]:
        """Parent, grandparent, ... (nearest first; excludes the node itself)."""
        out: list[str] = []
        cur = self.parent(node_id)
        while cur is not None and cur not in out:
            out.append(cur)
            cur = self.parent(cur)
        return out

    def descendants(self, node_id: str) -> set[str]:
        out: set[str] = set()
        stack = list(self.children(node_id))
        while stack:
            c = stack.pop()
            if c not in out:
                out.add(c)
                stack.extend(self.children(c))
        return out

    def own_spans(self, node_id: str) -> list[tuple[int, int]]:
        """Character spans (document offsets) of the node's own, non-whitespace text."""
        cached = self._own.get(node_id)
        if cached is not None:
            return cached
        n = self.nodes[node_id]
        kids = sorted((self.nodes[c] for c in self.children(node_id)), key=lambda k: k.char_start)
        raw: list[tuple[int, int]] = []
        cursor = n.char_start
        for k in kids:
            raw.append((cursor, max(cursor, k.char_start)))
            cursor = max(cursor, k.char_end)
        raw.append((cursor, n.char_end))
        spans: list[tuple[int, int]] = []
        for a, b in raw:
            seg = n.text[a - n.char_start : b - n.char_start]
            stripped = seg.strip()
            if stripped:
                lead = len(seg) - len(seg.lstrip())
                spans.append((a + lead, a + lead + len(stripped)))
        self._own[node_id] = spans
        return spans
