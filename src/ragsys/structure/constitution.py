"""ConstitutionParser: the Constitution of India with the statute parser's machinery.

Structure: Parts ("PART III" + its title line), Chapters inside Parts ("CHAPTER I.—THE
EXECUTIVE"), Articles ("21. Protection of life and personal liberty.—..."), including lettered
articles (21A, 243ZH; the printed "243Z-I" is stored as 243ZI), their clauses "(1)", sub-clauses
"(a)", "(i)", provisos, Explanations, and the Schedules (FIRST .. TWELFTH SCHEDULE). Group
headings printed between articles ("Right to Equality") end the preceding article and belong to
the enclosing Part or Chapter.

Node IDs (``<doc>`` is the configured doc_id, e.g. ``coi``)::

    Part           coi:part3          Part IVA      coi:part4A
    Chapter        coi:part5-ch1      (chapter numbering restarts in every Part)
    Article        coi:a21            lettered      coi:a21A, coi:a243ZH
    clause         coi:a21A(1)        sub-clause    coi:a19(1)(a)
    proviso        coi:a368(2)-prov1  Explanation   coi:a25-expl1
    Schedule       coi:sch7           preamble      coi:preamble

Articles use the node level ``section`` (the retrieval unit) and Parts/Chapters the level
``chapter``, so chunking, retrieval and the UI work unchanged; labels ("Article 21A", "Part III")
come from the document type ``constitution``.
"""

from __future__ import annotations

import re
from typing import Any

from ragsys.config import DocumentConfig, StructureConfig
from ragsys.ingest.clean import CleanDocument, Paragraph, is_all_caps_heading
from ragsys.structure.base import ParseResult, slugify
from ragsys.structure.statute import (
    _LEAD,
    ITEM_RE,
    _bold_ratio,
    _State,
    numbering_report,
    roman_to_int,
)

PART_RE = re.compile(r"^PART\s+(?P<num>[IVXL]+)(?P<suffix>[A-Z]?)\s*$")
CHAPTER_IN_PART_RE = re.compile(r"^CHAPTER\s+(?P<num>[IVXL]+)\b\s*\.?\s*(?:[—⎯-]\s*)?(?P<rest>.*)$")
ARTICLE_RE = re.compile(
    _LEAD
    + r"(?P<num>\d{1,3}[A-Z]{0,2}(?:-[A-Z])?)\s*\.\s*(?P<head>[A-Z\[(][^—\n]{1,300}?)"
    + r"\s*(?:[.:]?\s*—|[.:]\s*-(?=\s*[A-Z(\[]|\s*$)|-(?=\s*\())\s*(?P<rest>.*)$",
    re.DOTALL,
)
_ORDINALS = (
    "FIRST",
    "SECOND",
    "THIRD",
    "FOURTH",
    "FIFTH",
    "SIXTH",
    "SEVENTH",
    "EIGHTH",
    "NINTH",
    "TENTH",
    "ELEVENTH",
    "TWELFTH",
)
SCHEDULE_RE = re.compile(
    r"^(?:\[\s*)*(?P<ord>" + "|".join(_ORDINALS) + r")\s+SCHEDULE\b\s*[—⎯-]?\s*(?P<rest>.*)$"
)
_GROUP_HEADING_RE = re.compile(r"^[A-Z][A-Za-z ,'’\-]{2,78}[A-Za-z]$")
_OMITTED_RE = re.compile(r"^\W{0,3}Omitted\b")


def strip_amendment_brackets(head: str) -> str:
    """Remove the editorial brackets of substituted or inserted headings ("[Power of Parliament
    to amend the Constitution and procedure therefor]") while keeping inner brackets."""
    h = head.strip()
    while h.startswith("[") and h.count("[") > h.count("]"):
        h = h[1:].lstrip()
    while h.endswith("]") and h.count("]") > h.count("["):
        h = h[:-1].rstrip()
    if h.startswith("[") and h.endswith("]") and h.count("[") == 1 and h.count("]") == 1:
        h = h[1:-1].strip()
    return h or head


def _part_key(num: str, suffix: str) -> str | None:
    n = roman_to_int(num)
    if n is None or n > 30:
        return None
    return f"{n}{suffix}"


class _ConstitutionState(_State):
    def __init__(self, doc: CleanDocument, cfg: DocumentConfig, def_re: re.Pattern[str]) -> None:
        super().__init__(doc, cfg, def_re)
        self.part: Any = None
        self.pending_title: Any = None  # Part/Chapter whose title line is still to come

    def _is_group_heading(self, p: Paragraph) -> bool:
        t = p.text
        return (
            p.n_lines == 1
            and p.x0 >= 180
            and bool(_GROUP_HEADING_RE.match(t))
            and not is_all_caps_heading(t)
            and not ITEM_RE.match(t)
        )

    def _paragraph(self, p: Paragraph) -> None:
        t = p.text
        doc_id = self.cfg.doc_id
        if self.mode != "back" and any(r.search(t) for r in self.back_res):
            node = self._new(
                f"{doc_id}:h-{slugify(t)}",
                "heading",
                "UNIT",
                p.char_start,
                p.char_end,
                None,
                heading=t,
            )
            self.stack, self.chapter, self.part, self.mode = [node], None, None, "back"
            self.pending_title = None
            return
        if self.mode == "back":
            self._extend(self.stack[0], p.char_end)
            return

        if self.mode != "schedule":
            m = PART_RE.match(t)
            key = _part_key(m.group("num"), m.group("suffix")) if m else None
            if m and key is not None:
                self.part = self._new(
                    f"{doc_id}:part{key}",
                    "chapter",
                    "CHAPTER",
                    p.char_start,
                    p.char_end,
                    None,
                    number=f"{m.group('num')}{m.group('suffix')}",
                )
                self.chapter, self.pending_title = None, self.part
                self.stack, self.mode, self.table_mode = [], "body", False
                return
            m2 = CHAPTER_IN_PART_RE.match(t)
            if m2 and roman_to_int(m2.group("num")) is not None:
                n = roman_to_int(m2.group("num"))
                parent = self.part
                nid = f"{parent.node_id}-ch{n}" if parent is not None else f"{doc_id}:ch{n}"
                rest = m2.group("rest").strip() or None
                self.chapter = self._new(
                    nid,
                    "chapter",
                    "CHAPTER",
                    p.char_start,
                    p.char_end,
                    parent,
                    number=m2.group("num"),
                    heading=rest,
                )
                self.pending_title = self.chapter
                self.stack, self.mode, self.table_mode = [], "body", False
                return
            if self.pending_title is not None:
                head = self.pending_title
                if is_all_caps_heading(t, 200) and not ITEM_RE.match(t) and not ARTICLE_RE.match(t):
                    head.heading = f"{head.heading} {t}" if head.heading else t
                    self._extend(head, p.char_end)
                    return
                self.pending_title = None

        m3 = SCHEDULE_RE.match(t) if len(t) <= 200 else None
        if m3:
            num = str(_ORDINALS.index(m3.group("ord")) + 1)
            rest = m3.group("rest").strip(" —⎯-")
            node = self._new(
                f"{doc_id}:sch{num}",
                "schedule",
                "UNIT",
                p.char_start,
                p.char_end,
                None,
                number=num,
                heading=f"{m3.group('ord').title()} Schedule" + (f" — {rest}" if rest else ""),
            )
            self.stack, self.chapter, self.part, self.mode = [node], None, None, "schedule"
            self.pending_title = None
            return
        if self.mode == "schedule":
            self._extend(self.stack[0], p.char_end)
            return
        if t.strip() == "SCHEDULES":
            if self.stack:
                self._extend(self.stack[-1], p.char_end)
            return

        m4 = ARTICLE_RE.match(t)
        # omitted articles keep their number with the old heading in brackets and are set in
        # the regular font: "31. [Compulsory acquisition of property.].—Omitted by ..." or, with
        # the bracket before the number, "[259. Armed Forces in States ....].—Omitted by ..."
        omitted = m4 is not None and _OMITTED_RE.match(m4.group("rest")) is not None
        if m4 and (
            omitted or not self.headings_bold or _bold_ratio(m4.group("head"), p.bold_text) >= 0.5
        ):
            self._article(p, m4)
            return

        if self.mode == "pre":
            unit = self._unit()
            if unit is None or unit.kind != "UNIT":
                node = self._new(
                    f"{doc_id}:preamble",
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
        if self._is_group_heading(p):
            # "Right to Equality": ends the article before it; belongs to the Part/Chapter
            owner = self.chapter or self.part
            self.stack = []
            if owner is not None:
                self._extend(owner, p.char_end)
            return
        if not self.stack:
            owner = self.chapter or self.part
            if owner is not None:
                self._extend(owner, p.char_end)
            return
        self._body(t, p.char_start, p.char_end, p.x0)

    def _article(self, p: Paragraph, m: re.Match[str]) -> None:
        num = m.group("num").replace("-", "")
        head = strip_amendment_brackets(m.group("head").strip().rstrip(".:").strip())
        head = head.rstrip(".:").strip()
        if re.search(r"\betc$", head):
            head += "."
        node = self._new(
            f"{self.cfg.doc_id}:a{num}",
            "section",
            "UNIT",
            p.char_start,
            p.char_end,
            self.chapter or self.part,
            number=num,
            heading=head,
            x0=p.x0,
        )
        node.is_definitions = bool(self.def_re.match(head))
        self.stack = [node]
        self.mode = "body"
        self.table_mode = False
        self.pending_title = None
        rest = m.group("rest")
        if rest:
            offset = p.char_start + m.start("rest")
            self._body(rest, offset, p.char_end, p.x0)


class ConstitutionParser:
    """StructureParser for the Constitution of India (see the module docstring for IDs)."""

    name = "constitution"

    def __init__(self, cfg: StructureConfig | None = None) -> None:
        self.cfg = cfg or StructureConfig()
        self.def_heading_re = re.compile(self.cfg.definitions_heading_pattern, re.IGNORECASE)

    def parse(self, doc: CleanDocument, cfg: DocumentConfig) -> ParseResult:
        state = _ConstitutionState(doc, cfg, self.def_heading_re)
        state.run()
        nodes = state.to_nodes()
        warnings = list(state.warnings)
        numbering = numbering_report(nodes)
        if numbering["missing"]:
            warnings.append(f"gaps in article numbering: missing {numbering['missing']}")
        if numbering["duplicates"]:
            warnings.append(f"duplicate article numbers: {numbering['duplicates']}")
        parts = sum(1 for n in nodes if n.level == "chapter" and "-ch" not in n.node_id)
        stats: dict[str, Any] = {
            "numbering": numbering,
            "headings_bold": state.headings_bold,
            "parts": parts,
            "chapters_in_parts": sum(1 for n in nodes if n.level == "chapter") - parts,
            "articles": numbering["count"],
            "lettered_articles": sum(
                1 for n in nodes if n.level == "section" and (n.number or "")[-1:].isalpha()
            ),
        }
        return ParseResult(nodes=nodes, warnings=warnings, stats=stats)
