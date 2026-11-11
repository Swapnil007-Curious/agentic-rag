"""Cross-references in textbooks: "Section 4.2", "Sections 4.2 and 4.3", "Chapter 3",
"Fig. 5.1.1", "Figure 2.1", "Table 3.2".

A section reference resolves to ``<doc>:<number>`` when that node exists (``exact``), else to its
nearest existing numbered ancestor (``5.1.7`` -> ``5.1``: ``partial``). A reference whose chapter
was not ingested (chapter selection) is ``external``; one into an ingested chapter that names no
existing node is ``unresolved``. Figures and tables have no nodes and stay ``unresolved`` with
their raw text. Citations of other books ("[46, Section 2.2.2]") are skipped, as are references
inside code blocks.
"""

from __future__ import annotations

import re

from ragsys.structure.base import Node
from ragsys.structure.xrefs import Resolution, XRef

_NUM = r"\d{1,2}(?:\.\d{1,2}){1,4}"
SECTION_RE = re.compile(
    rf"\b(?:Sections?|Secs?\.|§)[^\S\n]*(?P<nums>{_NUM}(?:[^\S\n]*(?:,|and|or|to|–|-)[^\S\n]*{_NUM})*)"
)
CHAPTER_RE = re.compile(
    r"\b(?:Chapters?|Chs?\.)[^\S\n]+(?P<nums>\d{1,2}(?:[^\S\n]*(?:,|and|or|to|–|-)[^\S\n]*\d{1,2})*)\b"
)
FIGURE_RE = re.compile(r"\b(?:Figs?\.|Figures?)[^\S\n]*(?P<nums>\d{1,2}(?:\.\d{1,2}){0,3})\b")
TABLE_RE = re.compile(r"\bTables?[^\S\n]+(?P<nums>\d{1,2}(?:\.\d{1,2}){0,3})\b")
_CITATION_BEFORE = re.compile(r"\[\d{1,4}(?:,[^\]\n]{0,20})?,\s*$")


def _split(nums: str, pattern: str) -> list[str]:
    return re.findall(pattern, nums)


def extract_textbook_xrefs(doc_id: str, text: str, nodes: list[Node]) -> list[XRef]:
    doc_nodes = sorted((n for n in nodes if n.doc_id == doc_id), key=lambda n: n.order)
    if not doc_nodes or not text:
        return []
    owner: list[int] = [-1] * len(text)
    for idx, n in enumerate(doc_nodes):  # parents precede children -> deepest wins
        for i in range(n.char_start, min(n.char_end, len(text))):
            owner[i] = idx
    ids = {n.node_id for n in doc_nodes}
    chapters = {n.number for n in doc_nodes if n.level == "chapter" and n.number}
    out: list[XRef] = []

    def source(pos: int) -> Node | None:
        i = owner[pos] if 0 <= pos < len(owner) else -1
        return doc_nodes[i] if i >= 0 else None

    def resolve(number: str) -> tuple[str | None, Resolution]:
        if f"{doc_id}:{number}" in ids:
            return f"{doc_id}:{number}", "exact"
        if number.split(".")[0] not in chapters:
            return None, "external"
        parts = number.split(".")
        while len(parts) > 1:
            parts = parts[:-1]
            cand = f"{doc_id}:{'.'.join(parts)}"
            if cand in ids:
                return cand, "partial"
        return None, "unresolved"

    def add(m: re.Match[str], kind: str, numbers: list[str], resolvable: bool) -> None:
        src = source(m.start())
        if src is None or src.level == "code":
            return
        if _CITATION_BEFORE.search(text[max(0, m.start() - 30) : m.start()]):
            return
        for num in numbers:
            target, res = resolve(num) if resolvable else (None, "unresolved")
            out.append(
                XRef(
                    source_node=src.node_id,
                    source_doc=doc_id,
                    target_node=target,
                    target_doc=doc_id if target else None,
                    raw_text=m.group(0),
                    kind=kind,  # type: ignore[arg-type]
                    resolution=res,
                    char_start=m.start(),
                    char_end=m.end(),
                )
            )

    for m in SECTION_RE.finditer(text):
        add(m, "section", _split(m.group("nums"), _NUM), True)
    for m in CHAPTER_RE.finditer(text):
        add(m, "chapter", _split(m.group("nums"), r"\d{1,2}"), True)
    for m in FIGURE_RE.finditer(text):
        add(m, "figure", [m.group("nums")], False)
    for m in TABLE_RE.finditer(text):
        add(m, "table", [m.group("nums")], False)
    out.sort(key=lambda x: (x.char_start, x.char_end, x.target_node or ""))
    return out
