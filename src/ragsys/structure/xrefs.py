"""Cross-reference extraction and resolution.

Handles references such as "section 35", "sub-section (1) of section 34", "clause (7) of
section 2", "sections 69 to 72", "clauses (a), (p) and (q) of sub-section (2) of section 101",
"rule 4", "sub-rule (4)", "this Act", "these rules", and references from Rules to the parent
Act ("section 101 of the Act" -> cpa2019:s101). References are resolved to node IDs where
possible (falling back to the deepest existing ancestor = "partial"); external Acts and
unresolvable references are kept with their raw text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ragsys.structure.base import Node
from ragsys.structure.statute import int_to_roman, next_alpha, roman_to_int

Resolution = Literal["exact", "partial", "document", "external", "unresolved"]


class XRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_node: str
    source_doc: str
    target_node: str | None
    target_doc: str | None
    raw_text: str
    kind: Literal["section", "rule", "chapter", "schedule", "provision", "document"]
    resolution: Resolution
    char_start: int
    char_end: int


# --------------------------------------------------------------------------- patterns


def _nl(pattern: str) -> str:
    """Paragraphs are separated by '\\n' in the cleaned text; references never span them."""
    return pattern.replace(r"\s", r"[^\S\n]")


_LABELS = r"\(\w{1,5}\)(?:\s*,?\s*(?:and|or|to)?\s*\(\w{1,5}\))*"
_NUMS = r"\d{1,3}[A-Z]{0,2}(?:\s*(?:,|and|or|to)\s*\d{1,3}[A-Z]{0,2}\b)*"
_COMP = (
    r"(?:(?:first|second|third|fourth|fifth)\s+)?"
    r"(?:sub-sections?|sub-rules?|sub-clauses?|clauses?|items?|provisos?|Explanations?)"
    rf"\s*(?:{_LABELS})?\s+(?:of|to|in|under)\s+(?:the\s+said\s+|the\s+)?"
)
COMP_RE = re.compile(
    _nl(
        r"(?:(?P<ord>first|second|third|fourth|fifth)\s+)?"
        r"(?P<word>sub-sections?|sub-rules?|sub-clauses?|clauses?|items?|provisos?|Explanations?)"
        rf"\s*(?P<labels>{_LABELS})?\s+(?:of|to|in|under)\s+(?:the\s+said\s+|the\s+)?"
    ),
    re.IGNORECASE,
)
CHAIN_TAIL_RE = re.compile(_nl(rf"(?:{_COMP})+$"), re.IGNORECASE)
ANCHOR_RE = re.compile(
    _nl(rf"\b(?P<unit>sections?|rules?|Chapter|Schedule)\s+(?P<nums>{_NUMS}|(?-i:[IVXLC]+)\b)"),
    re.IGNORECASE,
)
# Named statutes: "... Act, 2019 (35 of 2019)", "Indian Penal Code, 1860",
# "Code of Criminal Procedure, 1973".
_NAMED_ACT = (
    r"(?P<name>[A-Z][\w.'&()-]*\s+(?:(?:[A-Z][\w.'&()-]*|and|of|the|for|on)\s+){0,9}(?:Act|Code)"
    r"|Code\s+of\s+(?:Civil|Criminal)\s+Procedure)"
    r"(?:,?\s+(?P<year>\d{4})(?:\s*\(\s*\d+\s+of\s+\d{4}\s*\))?"
    r"|\s*\(\s*\d+\s+of\s+(?P<year2>\d{4})\s*\))"  # "Indian Penal Code (45 of 1860)"
)
NAMED_ACT_RE = re.compile(_nl(_NAMED_ACT))
DOCREF_RE = re.compile(
    _nl(rf"^\s*,?\s*of\s+(?:(?P<this>this\s+Act)|(?P<the>the\s+Act)\b|(?:the\s+)?{_NAMED_ACT})")
)
REL_RE = re.compile(
    _nl(
        rf"(?:{_COMP})*(?P<last>(?:(?:first|second|third|fourth|fifth)\s+)?"
        rf"(?:sub-sections?|sub-rules?|sub-clauses?|clauses?)\s*{_LABELS})"
    ),
    re.IGNORECASE,
)
_REL_TAIL_CHAIN = re.compile(
    _nl(r"^\s+(?:of|to|in|under)\s+(?:the\s+said\s+|the\s+)?(?:sections?|rules?)\b"), re.I
)
DOC_SELF_RE = re.compile(r"\b(?P<what>this Act|these rules|the Act)\b")
_LAST_RE = re.compile(
    _nl(
        r"(?:(?P<ord>first|second|third|fourth|fifth)\s+)?"
        r"(?P<word>sub-sections?|sub-rules?|sub-clauses?|clauses?)"
        rf"\s*(?P<labels>{_LABELS})"
    ),
    re.IGNORECASE,
)
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def _split_labels(labels: str) -> list[str]:
    """'(a), (p) and (q)' -> [a, p, q]; '(f) to (m)' -> [f..m]; '(1) to (3)' -> [1, 2, 3]."""
    items = re.findall(r"\(\s*(\w{1,5})\s*\)", labels)
    if re.search(r"\)\s*to\s*\(", labels) and len(items) == 2:
        a, b = items
        if a.isdigit() and b.isdigit() and int(a) < int(b) <= int(a) + 60:
            return [str(i) for i in range(int(a), int(b) + 1)]
        ra, rb = roman_to_int(a), roman_to_int(b)
        if (
            a.islower()
            and b.islower()
            and len(a) == len(b) == 1
            and a < b
            and not (ra and rb and a != "i")
        ):
            out, cur = [a], a
            while cur != b and len(out) < 60:
                cur = next_alpha(cur)
                out.append(cur)
            return out
        if ra and rb and ra < rb <= ra + 30:
            return [int_to_roman(i) for i in range(ra, rb + 1)]
    return items


def _split_nums(nums: str) -> list[str]:
    parts = re.findall(r"\d{1,3}[A-Z]{0,2}", nums)
    if re.search(r"\bto\b", nums) and len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        a, b = int(parts[0]), int(parts[1])
        if a < b <= a + 60:
            return [str(i) for i in range(a, b + 1)]
    return parts


@dataclass
class _Comp:
    word: str
    labels: list[str]
    ordinal: int | None


def _components(chain: str) -> list[_Comp]:
    comps = []
    for m in COMP_RE.finditer(chain):
        comps.append(
            _Comp(
                word=m.group("word").lower(),
                labels=_split_labels(m.group("labels") or ""),
                ordinal=_ORD.get((m.group("ord") or "").lower()),
            )
        )
    return comps


def _suffix_paths(comps: list[_Comp]) -> list[str]:
    """Components are written innermost-first; build outer-to-inner ID suffixes."""
    paths = [""]
    for c in reversed(comps):
        if c.word.startswith("proviso"):
            suffix = f"-prov{c.ordinal or 1}" if c.word == "proviso" else ""
            paths = [p + suffix for p in paths]
        elif c.word.startswith("explanation"):
            paths = [p + "-expl1" for p in paths]
        elif c.labels:
            paths = [p + f"({lab})" for p in paths for lab in c.labels]
    return paths[:80]


class XRefResolver:
    def __init__(
        self,
        nodes: list[Node],
        doc_types: dict[str, str],
        parent_acts: dict[str, str | None],
        act_aliases: dict[str, str],
    ) -> None:
        self.ids = {n.node_id for n in nodes}
        self.by_id = {n.node_id: n for n in nodes}
        self.doc_types = doc_types
        self.parent_acts = parent_acts
        self.aliases = {self._alias_key(k): v for k, v in act_aliases.items()}

    @staticmethod
    def _alias_key(name: str) -> str:
        name = re.sub(r"\(\s*\d+\s+of\s+\d{4}\s*\)", "", name)
        name = re.sub(r"^the\s+", "", name.strip(), flags=re.I)
        return re.sub(r"[\s,]+", " ", name).strip().lower()

    def alias_doc(self, name: str, year: str) -> str | None:
        return self.aliases.get(self._alias_key(f"{name}, {year}"))

    def resolve_id(self, candidate: str) -> tuple[str | None, Resolution]:
        if candidate in self.ids:
            return candidate, "exact"
        cur = candidate
        while True:
            m = re.match(r"^(.*?)(\([^()]*\)|-prov\d+|-expl\d+)$", cur)
            if not m or not m.group(1) or m.group(1).endswith(":"):
                break
            cur = m.group(1)
            if cur in self.ids:
                return cur, "partial"
        return None, "unresolved"

    def unit_prefix(self, doc: str, unit_word: str) -> str | None:
        """'section' -> 's' in an act, 'rule' -> 'r' in rules."""
        dtype = self.doc_types.get(doc)
        if unit_word.startswith("section"):
            return "s" if dtype == "act" else None
        if unit_word.startswith("rule"):
            return "r" if dtype == "rules" else None
        return None


def _ancestors(node: Node, by_id: dict[str, Node]) -> list[Node]:
    out = [node]
    cur = node
    while cur.parent_id and cur.parent_id in by_id:
        cur = by_id[cur.parent_id]
        out.append(cur)
    return out


def _unit_of(node: Node, by_id: dict[str, Node]) -> Node | None:
    for a in _ancestors(node, by_id):
        if a.level in ("section", "rule", "schedule") or (
            a.level == "heading" and a.parent_id is None
        ):
            return a
    return None


def extract_xrefs(doc_id: str, text: str, nodes: list[Node], resolver: XRefResolver) -> list[XRef]:
    """Extract references from one document's cleaned text and resolve them."""
    doc_nodes = sorted((n for n in nodes if n.doc_id == doc_id), key=lambda n: n.order)
    if not doc_nodes or not text:
        return []
    owner: list[int] = [-1] * len(text)
    for idx, n in enumerate(doc_nodes):  # parents precede children -> deepest wins
        for i in range(n.char_start, min(n.char_end, len(text))):
            owner[i] = idx
    by_id = resolver.by_id
    doc_type = resolver.doc_types.get(doc_id)
    parent_act = resolver.parent_acts.get(doc_id)
    out: list[XRef] = []
    consumed: list[tuple[int, int]] = []

    def source_at(pos: int) -> Node | None:
        i = owner[pos] if 0 <= pos < len(owner) else -1
        return doc_nodes[i] if i >= 0 else None

    def add(
        src: Node,
        start: int,
        end: int,
        kind: str,
        target: str | None,
        tdoc: str | None,
        res: Resolution,
    ) -> None:
        out.append(
            XRef(
                source_node=src.node_id,
                source_doc=doc_id,
                target_node=target,
                target_doc=tdoc,
                raw_text=text[start:end],
                kind=kind,  # type: ignore[arg-type]
                resolution=res,
                char_start=start,
                char_end=end,
            )
        )

    # ---- anchored references: [chain of] section/rule/Chapter/Schedule N [of <doc>]
    for m in ANCHOR_RE.finditer(text):
        src = source_at(m.start())
        if src is None:
            continue
        unit_word = m.group("unit").lower()
        start, end = m.start(), m.end()
        window_start = max(0, start - 300)
        tail = CHAIN_TAIL_RE.search(text[window_start:start])
        comps: list[_Comp] = []
        if tail:
            comps = _components(tail.group(0))
            start = window_start + tail.start()
        dm = DOCREF_RE.match(text[end : end + 160])
        tdoc: str | None
        external = False
        if dm:
            end += dm.end()
            if dm.group("this"):
                tdoc = doc_id if doc_type == "act" else (parent_act or doc_id)
            elif dm.group("the"):
                tdoc = parent_act if doc_type == "rules" and parent_act else doc_id
            else:
                tdoc = resolver.alias_doc(dm.group("name"), dm.group("year") or dm.group("year2"))
                external = tdoc is None
        elif unit_word.startswith("section"):
            tdoc = doc_id if doc_type == "act" else parent_act
        else:
            tdoc = doc_id
        consumed.append((start, end))

        if unit_word in ("chapter", "schedule"):
            raw_num = m.group("nums").strip()
            num = str(roman_to_int(raw_num) or raw_num) if not raw_num.isdigit() else raw_num
            prefix = "ch" if unit_word == "chapter" else "sch"
            kind = "chapter" if unit_word == "chapter" else "schedule"
            if external or tdoc is None:
                add(src, start, end, kind, None, None, "external" if external else "unresolved")
            else:
                tid, res = resolver.resolve_id(f"{tdoc}:{prefix}{num}")
                add(src, start, end, kind, tid, tdoc, res)
            continue

        kind = "rule" if unit_word.startswith("rule") else "section"
        if comps:
            kind = "provision"
        if external:
            add(src, start, end, kind, None, None, "external")
            continue
        uprefix = resolver.unit_prefix(tdoc, unit_word) if tdoc else None
        if tdoc is None or uprefix is None:
            if unit_word.startswith("rule") and resolver.doc_types.get(tdoc or "") == "act":
                # an Act has no rules: "Rule 8 of Order XXXVII ..." belongs to another instrument
                add(src, start, end, kind, None, None, "external")
            else:
                add(src, start, end, kind, None, tdoc, "unresolved")
            continue
        for num in _split_nums(m.group("nums")):
            for suffix in _suffix_paths(comps) if comps else [""]:
                tid, res = resolver.resolve_id(f"{tdoc}:{uprefix}{num}{suffix}")
                add(src, start, end, kind, tid, tdoc, res)

    def overlaps(a: int, b: int) -> bool:
        return any(a < e and s < b for s, e in consumed)

    # ---- relative references without a section/rule anchor: "sub-section (2)", "clause (a)"
    for m in REL_RE.finditer(text):
        start, end = m.start(), m.end()
        if overlaps(start, end) or _REL_TAIL_CHAIN.match(text[end : end + 40]):
            continue
        src = source_at(start)
        if src is None:
            continue
        consumed.append((start, end))
        chain = text[start : m.start("last")]
        comps = _components(chain)
        last = _LAST_RE.match(m.group("last"))
        if last is None:
            continue
        outer = _Comp(last.group("word").lower(), _split_labels(last.group("labels")), None)
        suffixes = _suffix_paths([*comps, outer])
        unit = _unit_of(src, by_id)
        if outer.word.startswith(("sub-section", "sub-rule")):
            own = "sub-rule" if doc_type == "rules" else "sub-section"
            if unit is None or not outer.word.startswith(own):
                add(src, start, end, "provision", None, None, "unresolved")
                continue
            bases = [unit.node_id]
        else:
            bases = [a.node_id for a in _ancestors(src, by_id) if a.level not in ("chapter",)]
        hit: tuple[str | None, Resolution] = (None, "unresolved")
        for suffix in suffixes:
            for base in bases:
                if f"{base}{suffix}" in resolver.ids:
                    hit = (f"{base}{suffix}", "exact")
                    break
            if hit[0] is None and unit is not None:
                # "clause (c)" written inside sub-section (3) for clause (c) of sub-section (1):
                # accept when exactly one provision of the same section/rule has that label path
                same_unit = [
                    i
                    for i in resolver.ids
                    if i.startswith(unit.node_id + "(") and i.endswith(suffix)
                ]
                if len(same_unit) == 1:
                    hit = (same_unit[0], "exact")
            if hit[0] is None and bases:
                tid, res = resolver.resolve_id(f"{bases[0]}{suffix}")
                hit = (tid, "partial") if tid and tid != bases[0] else (None, "unresolved")
            add(src, start, end, "provision", hit[0], doc_id if hit[0] else None, hit[1])
            hit = (None, "unresolved")

    # ---- whole-document references: this Act / these rules / the Act / named Acts
    for m in NAMED_ACT_RE.finditer(text):
        start, end = m.start(), m.end()
        if overlaps(start, end):
            continue
        src = source_at(start)
        if src is None:
            continue
        consumed.append((start, end))
        tdoc = resolver.alias_doc(m.group("name"), m.group("year") or m.group("year2"))
        add(src, start, end, "document", None, tdoc, "document" if tdoc else "external")
    for m in DOC_SELF_RE.finditer(text):
        start, end = m.start(), m.end()
        if overlaps(start, end):
            continue
        src = source_at(start)
        if src is None:
            continue
        what = m.group("what")
        if what == "these rules":
            tdoc = doc_id
        elif what == "this Act":
            tdoc = doc_id if doc_type == "act" else parent_act
        else:  # "the Act"
            tdoc = parent_act if doc_type == "rules" else doc_id
        add(src, start, end, "document", None, tdoc, "document" if tdoc else "unresolved")

    out.sort(key=lambda x: (x.char_start, x.char_end, x.target_node or ""))
    return out


def xref_summary(xrefs: list[XRef]) -> dict[str, int]:
    counts = {k: 0 for k in ("exact", "partial", "document", "external", "unresolved")}
    for x in xrefs:
        counts[x.resolution] += 1
    counts["total"] = len(xrefs)
    counts["resolved_to_node"] = counts["exact"] + counts["partial"]
    return counts
