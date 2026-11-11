"""GenericParser: heading detection by font size/weight and numbering patterns.

Used for any document that is not configured as an act/rules document. Produces a tree of
``heading`` nodes; body paragraphs belong to the nearest preceding heading.
Node IDs: ``<doc>:preamble`` for text before the first heading, then hierarchical counters
``<doc>:h1``, ``<doc>:h1.2``, ``<doc>:h1.2.1`` in document order.

Textbooks (``doc_type: textbook``) use the headings, levels and code blocks found by the
textbook cleaner (:mod:`ragsys.ingest.textbook`) instead:

* levels: 1 -> ``chapter``, 2 -> ``section`` (the retrieval unit), 3 -> ``subsection``,
  deeper -> ``heading``; a heading's parent is the nearest preceding heading of a lower level;
* IDs: numbered headings ``<doc>:<number>`` (``d2l:5``, ``d2l:5.1``, ``d2l:5.1.2``);
  unnumbered headings ``<doc>:h-<slug>``; a slug used by several headings of the document is
  qualified with its parent's number or slug (``progit:h-git-basics-summary``); remaining
  clashes get ``~2``, ``~3``;
* code blocks become ``code`` nodes inside the heading they follow: ``<parent id>/code-<k>``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from ragsys.config import DocumentConfig
from ragsys.ingest.clean import CleanDocument, Paragraph
from ragsys.structure.base import Node, ParseResult, slugify

TextbookLevel = Literal["chapter", "section", "subsection", "heading", "code"]
TEXTBOOK_LEVELS: dict[int, TextbookLevel] = {1: "chapter", 2: "section", 3: "subsection"}

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
        if cfg.doc_type == "textbook":
            return _parse_textbook(doc, cfg)
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


# --------------------------------------------------------------------------- textbooks


@dataclass(eq=False)
class _T:
    node_id: str
    depth: int  # heading level; 0 = preamble, 99 = code
    level: TextbookLevel
    heading: str | None
    start: int
    end: int
    number: str | None = None
    slug: str = ""
    parent: _T | None = None
    children: list[_T] = field(default_factory=list)


def _extend(n: _T | None, end: int) -> None:
    while n is not None:
        n.end = max(n.end, end)
        n = n.parent


def _parse_textbook(doc: CleanDocument, cfg: DocumentConfig) -> ParseResult:
    doc_id = cfg.doc_id
    heads = [p for p in doc.paragraphs if p.kind == "heading"]
    slug_counts = Counter(slugify(p.title or p.text) for p in heads if not p.number)
    used: set[str] = set()
    all_nodes: list[_T] = []
    stack: list[_T] = []
    preamble: _T | None = None
    code_counts: Counter[str] = Counter()

    def unique(candidate: str) -> str:
        nid, k = candidate, 1
        while nid in used:
            k += 1
            nid = f"{candidate}~{k}"
        used.add(nid)
        return nid

    def owner(p: Paragraph) -> _T:
        nonlocal preamble
        if stack:
            return stack[-1]
        if preamble is None:
            preamble = _T(
                unique(f"{doc_id}:preamble"), 0, "heading", "Preamble", p.char_start, p.char_end
            )
            all_nodes.append(preamble)
        return preamble

    for p in doc.paragraphs:
        if p.kind == "heading":
            depth = p.level or 6
            while stack and stack[-1].depth >= depth:
                stack.pop()
            parent = stack[-1] if stack else None
            title = (p.title or p.text).strip()
            slug = slugify(title)
            if p.number:
                nid = unique(f"{doc_id}:{p.number}")
            elif slug_counts[slug] > 1 and parent is not None:
                qual = parent.number or parent.slug or "x"
                nid = unique(f"{doc_id}:h-{qual}-{slug}")
            else:
                nid = unique(f"{doc_id}:h-{slug}")
            node = _T(
                node_id=nid,
                depth=depth,
                level=TEXTBOOK_LEVELS.get(depth, "heading"),
                heading=title,
                start=p.char_start,
                end=p.char_end,
                number=p.number,
                slug=slug,
                parent=parent,
            )
            if parent is not None:
                parent.children.append(node)
            all_nodes.append(node)
            stack.append(node)
            _extend(node, p.char_end)
        elif p.kind == "code":
            own = owner(p)
            code_counts[own.node_id] += 1
            code = _T(
                node_id=unique(f"{own.node_id}/code-{code_counts[own.node_id]}"),
                depth=99,
                level="code",
                heading=None,
                start=p.char_start,
                end=p.char_end,
                parent=own,
            )
            own.children.append(code)
            all_nodes.append(code)
            _extend(code, p.char_end)
        else:
            _extend(owner(p), p.char_end)

    nodes: list[Node] = []
    for order, t in enumerate(all_nodes):
        end = t.end
        while end > t.start and doc.text[end - 1].isspace():
            end -= 1
        p0, p1 = doc.pages_for(t.start, end)
        nodes.append(
            Node(
                doc_id=doc_id,
                doc_title=cfg.title,
                doc_type=cfg.doc_type,
                node_id=t.node_id,
                level=t.level,
                number=t.number,
                heading=t.heading,
                text=doc.text[t.start : end],
                parent_id=t.parent.node_id if t.parent else None,
                children=[c.node_id for c in t.children],
                page_start=p0,
                page_end=p1,
                char_start=t.start,
                char_end=end,
                order=order,
            )
        )
    levels = Counter(n.level for n in nodes)
    warnings = [] if heads else ["no headings detected; whole document is one preamble node"]
    return ParseResult(
        nodes=nodes,
        warnings=warnings,
        stats={
            "headings": len(heads),
            "numbered_headings": sum(1 for p in heads if p.number),
            "code_blocks": levels.get("code", 0),
            "nodes_per_level": dict(sorted(levels.items())),
        },
    )
