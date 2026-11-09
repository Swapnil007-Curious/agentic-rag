"""GenericParser: heading detection by font size/weight and numbering patterns.

Used for any document that is not configured as an act/rules document. Produces a tree of
``heading`` nodes; body paragraphs belong to the nearest preceding heading.
Node IDs: ``<doc>:preamble`` for text before the first heading, then hierarchical counters
``<doc>:h1``, ``<doc>:h1.2``, ``<doc>:h1.2.1`` in document order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragsys.config import DocumentConfig
from ragsys.ingest.clean import CleanDocument, Paragraph
from ragsys.structure.base import Node, ParseResult

NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z].{0,120})$"
)


@dataclass(eq=False)
class _H:
    node_id: str
    level: int
    heading: str | None
    start: int
    end: int
    number: str | None = None
    parent: _H | None = None
    children: list[_H] = field(default_factory=list)


class GenericParser:
    name = "generic"

    def __init__(self, size_delta: float = 1.0, max_heading_chars: int = 150) -> None:
        self.size_delta = size_delta
        self.max_heading_chars = max_heading_chars

    def _is_heading(self, p: Paragraph, body: float) -> bool:
        t = p.text
        if len(t) > self.max_heading_chars or p.n_lines > 2:
            return False
        bigger = p.size >= body + self.size_delta
        boldish = p.bold_ratio >= 0.8
        numbered = bool(NUMBERED_HEADING_RE.match(t))
        if t.endswith((".", ":", ";", ",")) and not (bigger and len(t) < 60):
            return False
        return bigger or (boldish and (numbered or len(t.split()) <= 12))

    def parse(self, doc: CleanDocument, cfg: DocumentConfig) -> ParseResult:
        body = doc.body_font_size
        heads = [p for p in doc.paragraphs if self._is_heading(p, body)]
        sizes = sorted({round(p.size, 1) for p in heads}, reverse=True)
        size_rank = {s: i + 1 for i, s in enumerate(sizes)}

        def level_of(p: Paragraph) -> int:
            m = NUMBERED_HEADING_RE.match(p.text)
            if m:
                return m.group("num").count(".") + 1
            return size_rank.get(round(p.size, 1), len(sizes) + 1)

        all_nodes: list[_H] = []
        stack: list[_H] = []
        counters: dict[str, int] = {}
        head_ids = {id(p) for p in heads}
        current: _H | None = None

        def extend(n: _H | None, end: int) -> None:
            while n is not None:
                n.end = max(n.end, end)
                n = n.parent

        for p in doc.paragraphs:
            if id(p) in head_ids:
                lvl = level_of(p)
                while stack and stack[-1].level >= lvl:
                    stack.pop()
                parent = stack[-1] if stack else None
                key = parent.node_id if parent else ""
                counters[key] = counters.get(key, 0) + 1
                path = (
                    f"{parent.node_id.split(':h', 1)[1]}.{counters[key]}"
                    if parent
                    else str(counters[key])
                )
                m = NUMBERED_HEADING_RE.match(p.text)
                node = _H(
                    node_id=f"{cfg.doc_id}:h{path}",
                    level=lvl,
                    heading=p.text,
                    start=p.char_start,
                    end=p.char_end,
                    number=m.group("num") if m else None,
                    parent=parent,
                )
                if parent:
                    parent.children.append(node)
                all_nodes.append(node)
                stack.append(node)
                current = node
                extend(node, p.char_end)
            else:
                if current is None:
                    current = _H(f"{cfg.doc_id}:preamble", 0, "Preamble", p.char_start, p.char_end)
                    all_nodes.append(current)
                extend(current, p.char_end)

        nodes: list[Node] = []
        for order, h in enumerate(all_nodes):
            end = h.end
            while end > h.start and doc.text[end - 1].isspace():
                end -= 1
            p0, p1 = doc.pages_for(h.start, end)
            nodes.append(
                Node(
                    doc_id=cfg.doc_id,
                    doc_title=cfg.title,
                    doc_type=cfg.doc_type,
                    node_id=h.node_id,
                    level="heading",
                    number=h.number,
                    heading=h.heading,
                    text=doc.text[h.start : end],
                    parent_id=h.parent.node_id if h.parent else None,
                    children=[c.node_id for c in h.children],
                    page_start=p0,
                    page_end=p1,
                    char_start=h.start,
                    char_end=end,
                    order=order,
                )
            )
        warnings = [] if heads else ["no headings detected; whole document is one preamble node"]
        return ParseResult(nodes=nodes, warnings=warnings, stats={"headings": len(heads)})
