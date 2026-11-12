"""Chunk model, Chunker protocol and shared provenance helpers."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from ragsys.structure.base import Node
from ragsys.structure.xrefs import XRef


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    chunker: str
    doc_id: str
    text: str  # what is embedded
    display_text: str  # what is shown to users == clean_text[char_start:char_end]
    node_ids: list[str]  # every non-chapter node whose text overlaps the chunk
    primary_node_id: str | None
    section_ids: list[str]  # section/rule-level units (incl. schedules, preamble) overlapped
    heading_path: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    n_tokens: int  # tokens of ``text`` (bge-small tokenizer, no special tokens)
    xref_targets: list[str]


@dataclass
class DocInput:
    """Everything a chunker needs about one document."""

    doc_id: str
    title: str
    doc_type: str
    text: str
    page_spans: list[tuple[int, int, int]]
    nodes: list[Node]  # document order
    xrefs: list[XRef] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.nodes = sorted(self.nodes, key=lambda n: n.order)
        self.by_id = {n.node_id: n for n in self.nodes}
        self._span_starts = [s for s, _e, _p in self.page_spans]
        self._xrefs = sorted(
            (x for x in self.xrefs if x.target_node and x.resolution in ("exact", "partial")),
            key=lambda x: x.char_start,
        )
        self._xref_starts = [x.char_start for x in self._xrefs]

    @property
    def short_title(self) -> str:
        t = self.title.strip()
        return t[4:] if t.lower().startswith("the ") else t

    def page_at(self, offset: int) -> int:
        if not self.page_spans:
            return 1
        i = max(bisect.bisect_right(self._span_starts, offset) - 1, 0)
        return self.page_spans[i][2]

    def pages_for(self, start: int, end: int) -> tuple[int, int]:
        return self.page_at(start), self.page_at(max(start, end - 1))

    def overlapping(self, start: int, end: int) -> list[Node]:
        return [n for n in self.nodes if n.char_start < end and n.char_end > start]

    def xref_targets(self, start: int, end: int) -> list[str]:
        i = bisect.bisect_left(self._xref_starts, start)
        out: set[str] = set()
        while i < len(self._xrefs) and self._xrefs[i].char_start < end:
            x = self._xrefs[i]
            if x.char_end <= end and x.target_node:
                out.add(x.target_node)
            i += 1
        return sorted(out)

    def ancestors(self, node: Node) -> list[Node]:
        out = [node]
        while out[-1].parent_id and out[-1].parent_id in self.by_id:
            out.append(self.by_id[out[-1].parent_id])
        return out

    def unit_of(self, node: Node) -> Node | None:
        return next((a for a in self.ancestors(node) if a.is_unit), None)


class Chunker(Protocol):
    name: str

    def chunk_document(self, doc: DocInput) -> list[Chunk]: ...


# --------------------------------------------------------------------------- headings

_SUB_LABEL = {
    "subsection": ("Sub-section", "Sub-rule"),
    "clause": ("Clause", "Clause"),
    "subclause": ("Sub-clause", "Sub-clause"),
}


def textbook_label(node: Node) -> str:
    """'Chapter 5 — Multilayer Perceptrons', '5.1.2 Activation Functions', 'Git Basics'."""
    if node.level == "code":
        return "Code"
    if node.level == "chapter":
        if node.number:
            return (
                f"Chapter {node.number} — {node.heading}"
                if node.heading
                else f"Chapter {node.number}"
            )
        return node.heading or node.node_id
    if node.number:
        return f"{node.number} {node.heading}" if node.heading else node.number
    return node.heading or node.node_id


def node_label(node: Node, doc_type: str) -> str:
    """Human-readable label for one node, e.g. 'Section 35 — Manner in which ...'."""
    if doc_type == "textbook":
        return textbook_label(node)
    lvl = node.level
    if doc_type == "constitution":
        if lvl == "section":
            base = f"Article {node.number}"
            return f"{base} — {node.heading}" if node.heading else base
        if lvl == "chapter":
            word = "Chapter" if "-ch" in node.node_id.rsplit(":", 1)[-1] else "Part"
            base = f"{word} {node.number}"
            return f"{base} — {node.heading}" if node.heading else base
        if lvl in ("subsection", "clause", "subclause"):
            return f"Clause ({node.number})" if lvl == "subsection" else f"({node.number})"
    if lvl in ("section", "rule"):
        base = f"{'Section' if lvl == 'section' else 'Rule'} {node.number}"
        return f"{base} — {node.heading}" if node.heading else base
    if lvl == "chapter":
        return f"Chapter {node.number}"
    if lvl == "schedule":
        return f"Schedule {node.number}"
    if lvl in _SUB_LABEL:
        word = _SUB_LABEL[lvl][1 if doc_type == "rules" else 0]
        label = f"{word} ({node.number})"
        return f'{label} — "{node.defined_term}"' if node.defined_term else label
    if lvl == "proviso":
        return f"Proviso {node.number}"
    if lvl == "explanation":
        return f"Explanation {node.number}" if node.number and node.number != "1" else "Explanation"
    return node.heading or node.node_id


def heading_path(doc: DocInput, node: Node | None, include_sub: bool = True) -> str:
    """'<short title> > Chapter IV > Section 35 — <heading>[ > Sub-section (1) ...]'."""
    parts = [doc.short_title]
    if node is None:
        return parts[0]
    chain = list(reversed(doc.ancestors(node)))
    if doc.doc_type == "textbook":  # code blocks are not part of a heading path
        chain = [n for n in chain if n.level != "code"]
    unit_seen = False
    for n in chain:
        if n.level == "chapter":
            parts.append(node_label(n, doc.doc_type))
        elif n.is_unit or (n.level == "heading" and not unit_seen):
            parts.append(node_label(n, doc.doc_type))
            unit_seen = True
        elif include_sub:
            parts.append(node_label(n, doc.doc_type))
    return " > ".join(parts)


def deepest_containing(doc: DocInput, start: int, end: int) -> Node | None:
    best: Node | None = None
    for n in doc.nodes:
        if (
            n.char_start <= start
            and n.char_end >= end
            and (best is None or (n.char_start, n.order) >= (best.char_start, best.order))
        ):
            best = n
    return best


def make_chunk(
    doc: DocInput,
    chunker: str,
    index: int,
    start: int,
    end: int,
    n_tokens: int,
    primary: Node | None = None,
    embed_text: str | None = None,
    path: str | None = None,
) -> Chunk:
    """Build a chunk with provenance derived from character offsets."""
    overlap = doc.overlapping(start, end)
    units = [n for n in overlap if n.is_unit]
    if primary is None:
        primary = deepest_containing(doc, start, end)
        if (primary is None or primary.level == "chapter") and units:
            # crosses units: take the unit with the largest overlap (earliest on ties)
            primary = max(
                units, key=lambda n: (min(n.char_end, end) - max(n.char_start, start), -n.order)
            )
    display = doc.text[start:end]
    p0, p1 = doc.pages_for(start, end)
    return Chunk(
        chunk_id=f"{chunker}:{doc.doc_id}:{index:05d}",
        chunker=chunker,
        doc_id=doc.doc_id,
        text=embed_text if embed_text is not None else display,
        display_text=display,
        node_ids=[n.node_id for n in overlap if n.level != "chapter"],
        primary_node_id=primary.node_id if primary else None,
        section_ids=[n.node_id for n in units],
        heading_path=path if path is not None else heading_path(doc, primary, include_sub=False),
        page_start=p0,
        page_end=p1,
        char_start=start,
        char_end=end,
        n_tokens=n_tokens,
        xref_targets=doc.xref_targets(start, end),
    )
