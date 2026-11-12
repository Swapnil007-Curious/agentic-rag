"""IndianStatuteParser: chapters, sections/rules, sub-sections, clauses, sub-clauses,
Explanations, provisos, schedules and definition clauses of Indian Acts and Rules.

Node ID convention (stable; documented in README):
    act section        cpa2019:s35            rule            ecom2020:r4
    sub-section        cpa2019:s35(1)         sub-rule        ecom2020:r4(3)
    clause             cpa2019:s2(7)          sub-clause      cpa2019:s2(7)(i)
    proviso            cpa2019:s58(1)(a)(i)-prov1    (numbered among siblings)
    Explanation        cpa2019:s2(7)-expl1
    chapter            cpa2019:ch4            schedule        commissions2020:sch1
    preamble           cpa2019:preamble       other headings  cpa2019:h-<slug>
A repeated label inside one parent (a drafting error in the source) gets "~2", "~3".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ragsys.config import DocumentConfig, StructureConfig
from ragsys.ingest.clean import CleanDocument, Paragraph, is_all_caps_heading
from ragsys.structure.base import Node, ParseResult, slugify

# --------------------------------------------------------------------------- patterns

_LEAD = r"(?:\[\s*)*"  # amendment brackets: "[ [(1A) ..."
# Heading separators seen in the corpus: ".—", "—", ":—", "-(1)", ": -" (end), ".-The ..."
SECTION_RE = re.compile(
    _LEAD
    + r"(?P<num>\d{1,3}[A-Z]{0,2})\s*\.\s+(?P<head>[A-Z\[(][^—\n]{1,300}?)"
    + r"\s*(?:[.:]?\s*—|[.:]\s*-(?=\s*[A-Z(\[]|\s*$)|-(?=\s*\())\s*(?P<rest>.*)$",
    re.DOTALL,
)
TABLE_HEADER_RE = re.compile(r"^(?:\(\s*\d{1,2}\s*\)\s*){2,}")
_SEQUENCE_STARTS = frozenset({"1", "a", "i", "A", "I"})
ITEM_RE = re.compile(_LEAD + r"\(\s*(?P<label>\d{1,3}[A-Z]{0,2}|[a-z]{1,3}|[A-Z]{1,2})\s*\)\s*")
CHAPTER_RE = re.compile(r"^CHAPTER\s+(?P<num>[IVXLC]+[A-Z]?)\b\.?\s*(?:—\s*)?(?P<rest>.*)$")
SCHEDULE_RE = re.compile(
    r"^(?:THE\s+)?(?:(?P<ord>FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)\s+)?"
    r"SCHEDULE\b(?:\s*[-—]?\s*(?P<num>[IVXLC]+|\d+)\b)?"
)
EXPL_RE = re.compile(
    r"^Explanations?\b(?:\s*(?P<n>[IVX]+|\d+))?\s*[.:]?\s*—?\s*(?P<rest>.*)$", re.DOTALL
)
PROVISO_RE = re.compile(r"^Provided\b")
DEF_RE = re.compile(
    r"""^\s*["']\s*(?P<term>[^"'\n]{1,120}?)\s*["']\s*,?\s*
        (?:\([^)]{0,80}\)\s*,?\s*)?
        (?:in\s+relation\s+to\s+[^,]{1,120},\s*)?
        (?:means|mean|includes|include|shall\s+mean|shall\s+include|shall\s+have|has\s+the\s+meaning|
           have\s+the\s+meaning|is|are)\b""",
    re.VERBOSE,
)
_SCOPE_RE = re.compile(
    r"\bthis\s+(sub-clause|clause|sub-section|sub-rule|section|rule|proviso|Chapter|Act)\b",
    re.IGNORECASE,
)
_ORDINALS = {
    "FIRST": 1,
    "SECOND": 2,
    "THIRD": 3,
    "FOURTH": 4,
    "FIFTH": 5,
    "SIXTH": 6,
    "SEVENTH": 7,
    "EIGHTH": 8,
    "NINTH": 9,
    "TENTH": 10,
}
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
LIST_KINDS = frozenset({"ALPHA", "ROMAN", "UPPER"})
ITEM_KINDS = frozenset({"NUM", "ALPHA", "ROMAN", "UPPER"})


def roman_to_int(s: str) -> int | None:
    s = s.lower()
    if not s or any(c not in _ROMAN_VALUES for c in s):
        return None
    total = 0
    for i, c in enumerate(s):
        v = _ROMAN_VALUES[c]
        if i + 1 < len(s) and _ROMAN_VALUES[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


def int_to_roman(n: int) -> str:
    table = [
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ]
    out = ""
    for v, sym in table:
        while n >= v:
            out += sym
            n -= v
    return out


def next_alpha(label: str) -> str:
    """a -> b, ..., z -> za, za -> zb (Indian drafting convention for long lists)."""
    if len(label) == 1:
        return "za" if label == "z" else chr(ord(label) + 1)
    head, last = label[:-1], label[-1]
    return head + ("a" if last == "z" else chr(ord(last) + 1))


def is_roman_candidate(label: str) -> bool:
    return bool(re.fullmatch(r"[ivxlc]+", label)) and roman_to_int(label) is not None


# --------------------------------------------------------------------------- builder


@dataclass(eq=False)
class _B:
    node_id: str
    level: str
    kind: str  # UNIT, CHAPTER, NUM, ALPHA, ROMAN, UPPER, EXPL, PROV
    start: int
    end: int
    x0: float = 0.0
    number: str | None = None
    heading: str | None = None
    label: str | None = None
    parent: _B | None = None
    children: list[_B] = field(default_factory=list)
    defined_term: str | None = None
    is_definitions: bool = False


class IndianStatuteParser:
    """StructureParser for Indian Acts and Rules (see module docstring for IDs)."""

    name = "statute"

    def __init__(self, cfg: StructureConfig | None = None) -> None:
        self.cfg = cfg or StructureConfig()
        self.def_heading_re = re.compile(self.cfg.definitions_heading_pattern, re.IGNORECASE)

    # ------------------------------------------------------------------ public

    def parse(self, doc: CleanDocument, cfg: DocumentConfig) -> ParseResult:
        state = _State(doc, cfg, self.def_heading_re)
        state.run()
        nodes = state.to_nodes()
        warnings = list(state.warnings)
        numbering = numbering_report(nodes)
        if numbering["missing"]:
            warnings.append(
                f"gaps in {numbering['unit']} numbering: missing {numbering['missing']}"
            )
        if numbering["duplicates"]:
            warnings.append(f"duplicate {numbering['unit']} numbers: {numbering['duplicates']}")
        stats: dict[str, Any] = {"numbering": numbering, "headings_bold": state.headings_bold}
        return ParseResult(nodes=nodes, warnings=warnings, stats=stats)


class _State:
    def __init__(self, doc: CleanDocument, cfg: DocumentConfig, def_re: re.Pattern[str]) -> None:
        self.doc = doc
        self.cfg = cfg
        self.def_re = def_re
        self.is_act = cfg.doc_type == "act"
        self.back_res = [re.compile(p) for p in cfg.back_matter_markers]
        self.all: list[_B] = []
        self.ids: set[str] = set()
        self.stack: list[_B] = []
        self.chapter: _B | None = None
        self.pending_chapter: _B | None = None
        self.mode = "pre"  # pre | body | schedule | back
        self.table_mode = False  # inside an embedded table (rows are plain text of the unit)
        self.warnings: list[str] = []
        self.schedule_count = 0
        self.headings_bold = self._detect_bold_headings()

    # ------------------------------------------------------------------ helpers

    def _detect_bold_headings(self) -> bool:
        ratios = [
            _bold_ratio(m.group("head"), p.bold_text)
            for p in self.doc.paragraphs
            if (m := SECTION_RE.match(p.text))
        ]
        if not ratios:
            return False
        return sum(r >= 0.5 for r in ratios) / len(ratios) >= 0.5

    def _new(
        self,
        node_id: str,
        level: str,
        kind: str,
        start: int,
        end: int,
        parent: _B | None,
        **kw: Any,
    ) -> _B:
        base = node_id
        k = 2
        while node_id in self.ids:
            node_id = f"{base}~{k}"
            k += 1
        if node_id != base:
            self.warnings.append(f"duplicate label {base!r} in source; stored as {node_id!r}")
        self.ids.add(node_id)
        node = _B(
            node_id=node_id, level=level, kind=kind, start=start, end=end, parent=parent, **kw
        )
        if parent is not None:
            parent.children.append(node)
        self.all.append(node)
        self._extend(node, end)
        return node

    @staticmethod
    def _extend(node: _B | None, end: int) -> None:
        while node is not None:
            if end > node.end:
                node.end = end
            node = node.parent

    def _unit(self) -> _B | None:
        return self.stack[0] if self.stack else None

    # ------------------------------------------------------------------ main loop

    def run(self) -> None:
        for p in self.doc.paragraphs:
            self._paragraph(p)

    def _paragraph(self, p: Paragraph) -> None:
        t = p.text
        if self.mode != "back" and any(r.search(t) for r in self.back_res):
            node = self._new(
                f"{self.cfg.doc_id}:h-{slugify(t)}",
                "heading",
                "UNIT",
                p.char_start,
                p.char_end,
                None,
                heading=t,
            )
            self.stack, self.chapter, self.mode = [node], None, "back"
            return
        if self.mode == "back":
            self._extend(self.stack[0], p.char_end)
            return

        if self.mode != "schedule" and (m := CHAPTER_RE.match(t)):
            num = m.group("num")
            n = roman_to_int(num)
            if n is not None:
                key = str(n)
            elif len(num) > 1 and (base := roman_to_int(num[:-1])) is not None:
                key = f"{base}{num[-1]}"  # inserted chapter, e.g. "IVA" -> ch4A
            else:
                key = num
            rest = m.group("rest").strip() or None
            self.chapter = self._new(
                f"{self.cfg.doc_id}:ch{key}",
                "chapter",
                "CHAPTER",
                p.char_start,
                p.char_end,
                None,
                number=num,
                heading=rest,
            )
            self.pending_chapter = None if rest else self.chapter
            self.stack, self.mode, self.table_mode = [], "body", False
            return
        if self.pending_chapter is not None:
            chap = self.pending_chapter
            if is_all_caps_heading(t, 200) and not ITEM_RE.match(t) and not SECTION_RE.match(t):
                chap.heading = f"{chap.heading} {t}" if chap.heading else t
                self._extend(chap, p.char_end)
                return
            self.pending_chapter = None

        if len(t) <= 80 and (m := SCHEDULE_RE.match(t)):
            self.schedule_count += 1
            if m.group("ord"):
                num = str(_ORDINALS[m.group("ord")])
            elif m.group("num"):
                num_raw = m.group("num")
                num = num_raw if num_raw.isdigit() else str(roman_to_int(num_raw) or num_raw)
            else:
                num = str(self.schedule_count)
            node = self._new(
                f"{self.cfg.doc_id}:sch{num}",
                "schedule",
                "UNIT",
                p.char_start,
                p.char_end,
                None,
                number=num,
                heading=t,
            )
            self.stack, self.chapter, self.mode = [node], None, "schedule"
            return
        if self.mode == "schedule":
            self._extend(self.stack[0], p.char_end)
            return

        m = SECTION_RE.match(t)
        if m and (not self.headings_bold or _bold_ratio(m.group("head"), p.bold_text) >= 0.5):
            self._section(p, m)
            return

        if self.mode == "pre":
            unit = self._unit()
            if unit is None or unit.kind != "UNIT":
                node = self._new(
                    f"{self.cfg.doc_id}:preamble",
                    "heading",
                    "UNIT",
                    p.char_start,
                    p.char_end,
                    None,
                    heading="Preamble",
                )
                self.stack = [node]
            else:
                self._extend(unit, p.char_end)
            return
        if not self.stack:
            # text after a chapter heading but before its first section
            if self.chapter is not None:
                self._extend(self.chapter, p.char_end)
            return
        self._body(t, p.char_start, p.char_end, p.x0)

    # ------------------------------------------------------------------ sections

    def _section(self, p: Paragraph, m: re.Match[str]) -> None:
        num = m.group("num")
        head = m.group("head").strip().rstrip(".:").strip()
        if re.search(r"\betc$", head):
            head += "."
        prefix = "s" if self.is_act else "r"
        level = "section" if self.is_act else "rule"
        node = self._new(
            f"{self.cfg.doc_id}:{prefix}{num}",
            level,
            "UNIT",
            p.char_start,
            p.char_end,
            self.chapter,
            number=num,
            heading=head,
            x0=p.x0,
        )
        node.is_definitions = bool(self.def_re.match(head))
        self.stack = [node]
        self.mode = "body"
        self.table_mode = False
        rest = m.group("rest")
        if rest:
            offset = p.char_start + m.start("rest")
            if ITEM_RE.match(rest) or EXPL_RE.match(rest) or PROVISO_RE.match(rest):
                self._body(rest, offset, p.char_end, p.x0)

    # ------------------------------------------------------------------ body elements

    def _enter_table(self, end: int) -> None:
        unit = self.stack[0]
        if not self.table_mode:
            self.warnings.append(f"embedded table in {unit.node_id} kept as plain text of the unit")
        self.table_mode = True
        del self.stack[1:]
        self._extend(unit, end)

    def _body(self, t: str, start: int, end: int, x0: float) -> None:
        if self.table_mode or TABLE_HEADER_RE.match(t):
            self._enter_table(end)
            return
        if m := EXPL_RE.match(t):
            self._explanation(m, start, end, x0)
            return
        if PROVISO_RE.match(t):
            self._proviso(t, start, end, x0)
            return
        if m2 := ITEM_RE.match(t):
            self._item(m2.group("label"), start, end, x0, t[m2.end() :])
            return
        top = self.stack[-1]
        if top.kind in LIST_KINDS and len(self.stack) > 1:
            # un-numbered text after a list continues the provision that introduced the list
            self.stack.pop()
            top = self.stack[-1]
        self._extend(top, end)

    def _kind_for(self, label: str, x0: float) -> str:
        if label[0].isdigit():
            return "NUM"
        if label.isupper():
            return "UPPER"
        if not is_roman_candidate(label):
            return "ALPHA"
        alpha = next((b for b in reversed(self.stack) if b.kind == "ALPHA"), None)
        roman = next((b for b in reversed(self.stack) if b.kind == "ROMAN"), None)
        alpha_next = next_alpha(alpha.label) if alpha and alpha.label else "a"
        roman_val = roman_to_int(roman.label) if roman and roman.label else None
        roman_next = int_to_roman(roman_val + 1) if roman_val else "i"
        if label == roman_next and label != alpha_next:
            return "ROMAN"
        if label == alpha_next and label != roman_next:
            return "ALPHA"
        if label == alpha_next == roman_next:
            # e.g. "(i)" after "(h)": deeper indentation means a sub-clause of (h)
            return "ROMAN" if alpha is not None and x0 > alpha.x0 + 5 else "ALPHA"
        if roman is not None or len(label) > 1:
            return "ROMAN"
        return "ALPHA"

    def _item(self, label: str, start: int, end: int, x0: float, after: str) -> None:
        kind = self._kind_for(label, x0)
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].kind == kind:
                del self.stack[i:]
                break
        parent = self.stack[-1]
        siblings = [c.label for c in parent.children if c.kind == kind]
        if label in siblings and label in _SEQUENCE_STARTS and len(siblings) >= 2:
            # numbering restarts at (1)/(a)/(i) under the same parent: an embedded table/list
            self._enter_table(end)
            return
        node = self._new(
            f"{parent.node_id}({label})",
            "item",
            kind,
            start,
            end,
            parent,
            number=label,
            label=label,
            x0=x0,
        )
        unit = self.stack[0]
        if unit.is_definitions and parent.kind in ("UNIT", "NUM"):
            dm = DEF_RE.match(after)
            if dm:
                node.defined_term = dm.group("term").strip()
        self.stack.append(node)

    def _explanation(self, m: re.Match[str], start: int, end: int, x0: float) -> None:
        idx = self._expl_target(m.group("rest") or "")
        del self.stack[idx + 1 :]
        parent = self.stack[-1]
        n = 1 + sum(1 for c in parent.children if c.kind == "EXPL")
        node = self._new(
            f"{parent.node_id}-expl{n}",
            "explanation",
            "EXPL",
            start,
            end,
            parent,
            number=str(n),
            x0=x0,
        )
        self.stack.append(node)
        rest = m.group("rest")
        if rest and ITEM_RE.match(rest):
            self._item(ITEM_RE.match(rest).group("label"), start + m.start("rest"), end, x0, rest)  # type: ignore[union-attr]

    def _expl_target(self, rest: str) -> int:
        stack = self.stack
        unit = stack[0]

        def last(kinds: set[str]) -> int | None:
            return next((i for i in range(len(stack) - 1, 0, -1) if stack[i].kind in kinds), None)

        scope = _SCOPE_RE.search(rest[:120])
        if scope:
            w = scope.group(1).lower()
            if w in ("section", "rule", "chapter", "act"):
                return 0
            if w in ("sub-section", "sub-rule"):
                return last({"NUM"}) or 0
            if w == "clause":
                i = last({"ALPHA"})
                if i is None and unit.is_definitions:
                    i = last({"NUM"})
                return i if i is not None else (last(set(ITEM_KINDS)) or 0)
            if w == "sub-clause":
                i = last({"ROMAN"})
                return i if i is not None else (last(set(ITEM_KINDS)) or 0)
            if w == "proviso":
                return last({"PROV"}) or (len(stack) - 1)
        i = len(stack) - 1
        while i > 0 and stack[i].kind in ("EXPL", "PROV"):
            i -= 1
        if i > 0 and stack[i].kind in LIST_KINDS:
            i -= 1
        return i

    def _proviso(self, t: str, start: int, end: int, x0: float) -> None:
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].kind == "PROV":
                del self.stack[i:]
                break
        while (
            len(self.stack) > 1 and self.stack[-1].kind in ITEM_KINDS and x0 < self.stack[-1].x0 - 4
        ):
            self.stack.pop()
        if len(self.stack) > 1 and self.stack[-1].kind in LIST_KINDS:
            # "Provided that, in any case other than those mentioned in this rule, ..." after
            # the last clause of a list qualifies the rule / sub-rule, not that clause.
            scope = _SCOPE_RE.search(t[:200])
            if scope:
                w = scope.group(1).lower()
                if w in ("section", "rule"):
                    del self.stack[1:]
                elif w in ("sub-section", "sub-rule"):
                    i = next(
                        (
                            k
                            for k in range(len(self.stack) - 1, 0, -1)
                            if self.stack[k].kind == "NUM"
                        ),
                        0,
                    )
                    del self.stack[i + 1 :]
        parent = self.stack[-1]
        n = 1 + sum(1 for c in parent.children if c.kind == "PROV")
        node = self._new(
            f"{parent.node_id}-prov{n}", "proviso", "PROV", start, end, parent, number=str(n), x0=x0
        )
        self.stack.append(node)

    # ------------------------------------------------------------------ output

    def _assign_levels(self) -> None:
        for b in self.all:
            if b.kind not in ITEM_KINDS:
                continue
            anc = b.parent
            while anc is not None and anc.kind in ("EXPL", "PROV"):
                anc = anc.parent
            if anc is None or anc.kind == "UNIT":
                if b.kind == "NUM":
                    unit = anc
                    b.level = (
                        "clause"
                        if unit is not None and _numbered_definitions(unit)
                        else "subsection"
                    )
                else:
                    b.level = "clause"
            elif anc.level == "subsection":
                b.level = "clause"
            else:
                b.level = "subclause"

    def to_nodes(self) -> list[Node]:
        self._assign_levels()
        out: list[Node] = []
        text = self.doc.text
        for order, b in enumerate(self.all):
            end = b.end
            while end > b.start and text[end - 1].isspace():
                end -= 1
            p0, p1 = self.doc.pages_for(b.start, end)
            out.append(
                Node(
                    doc_id=self.cfg.doc_id,
                    doc_title=self.cfg.title,
                    doc_type=self.cfg.doc_type,
                    node_id=b.node_id,
                    level=b.level,  # type: ignore[arg-type]
                    number=b.number,
                    heading=b.heading,
                    text=text[b.start : end],
                    parent_id=b.parent.node_id if b.parent else None,
                    children=[c.node_id for c in b.children],
                    page_start=p0,
                    page_end=p1,
                    char_start=b.start,
                    char_end=end,
                    defined_term=b.defined_term,
                    order=order,
                )
            )
        return out


def _numbered_definitions(unit: _B) -> bool:
    if not unit.is_definitions:
        return False
    nums = [c for c in unit.children if c.kind == "NUM"]
    return bool(nums) and sum(1 for c in nums if c.defined_term) >= 0.5 * len(nums)


def _bold_ratio(head: str, bold_text: str) -> float:
    words = re.findall(r"[a-z]+", head.lower())
    if not words:
        return 0.0
    bold = Counter(re.findall(r"[a-z]+", bold_text.lower()))
    hits = 0
    for w, c in Counter(words).items():
        hits += min(c, bold[w])
    return hits / len(words)


def numbering_report(nodes: list[Node]) -> dict[str, Any]:
    """Contiguity check of section/rule numbers: gaps, duplicates, count vs maximum."""
    units = [n for n in nodes if n.level in ("section", "rule")]
    unit: str = units[0].level if units else "section"
    if units and units[0].doc_type == "constitution":
        unit = "article"  # articles use the node level "section"
    numbers = [n.number or "" for n in units]
    ints = sorted({int(m.group(0)) for x in numbers if (m := re.match(r"\d+", x))})
    plain = [int(x) for x in numbers if x.isdigit()]
    dup = sorted({x for x in plain if plain.count(x) > 1})
    missing: list[int] = []
    if ints:
        present = set(ints)
        missing = [i for i in range(1, max(ints) + 1) if i not in present]
    return {
        "unit": unit,
        "count": len(units),
        "distinct": len(set(numbers)),
        "max_number": max(ints) if ints else None,
        "missing": missing,
        "duplicates": dup,
        "count_equals_max": bool(ints) and len(set(plain)) == max(ints),
    }
