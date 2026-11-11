"""Cleaning for textbook PDFs (``doc_type: textbook``).

The statute cleaner (:mod:`ragsys.ingest.clean`) is tuned to Gazette layouts; textbooks need
different rules:

1. normalisation keeps code and math intact: no ``--`` -> em dash, no footnote-asterisk removal,
   the minus sign becomes ``-``; private-use glyphs (icon fonts) are removed;
2. running headers/footers: lines at the top/bottom edge of a page whose leading or trailing
   number is the printed page number (the PDF-page minus printed-page offset is learned from the
   whole document), repeated edge lines (as for statutes), configured patterns, standalone page
   numbers and small standalone margin markers;
3. code: a line is code when most of its characters are set in a monospace font; consecutive
   code lines form one code block that keeps its line breaks and indentation;
4. headings: lines set larger than the body text, or in a heading-only font family with
   section numbering ("2.1.1 The Basics"); wrapped headings and "Chapter 2" + title lines are
   merged; levels come from the numbering depth ("5" < "5.1" < "5.1.1"), else from the font
   size rank; the first heading on a selected chapter's first page is the chapter (level 1);
5. sections whose title matches ``textbook.drop_sections`` (Exercises, References, Index, ...)
   are dropped with their subsections; exercise lists inside a section ("Exercise 2.1. ...")
   are dropped up to the next heading;
6. paragraphs (gaps, indentation, list items; words broken across lines are re-joined) and the
   cleaned text with exact page provenance. Paragraphs carry ``kind`` ("heading", "code",
   "text") and heading level/number/title for :class:`ragsys.structure.generic.GenericParser`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ragsys.config import CleaningConfig, DocumentConfig, TextbookConfig
from ragsys.ingest.chapters import ChapterRange, normalise_title
from ragsys.ingest.clean import (
    _PAGE_NUMBER_RE,
    _WORD_RE,
    CleanDocument,
    Cleaner,
    CLine,
    Paragraph,
    _join_lines,
)
from ragsys.ingest.extract import Line, RawDocument, Span

# --------------------------------------------------------------------------- normalisation

_TB_TABLE = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "­": None,  # soft hyphen
        "​": None,  # zero-width space / joiners / BOM
        "‌": None,
        "‍": None,
        "﻿": None,
        " ": " ",  # no-break, thin and narrow no-break spaces
        " ": " ",
        " ": " ",
        "‐": "-",  # hyphen, non-breaking hyphen, minus sign
        "‑": "-",
        "−": "-",
        "	": "    ",
    }
)
_PUA_RE = re.compile("[\ue000-\uf8ff]")
_WS_RE = re.compile(r"\s+")


def tb_normalise(text: str) -> str:
    """Prose normalisation for textbooks (code fragments and math survive)."""
    text = unicodedata.normalize("NFKC", text).translate(_TB_TABLE)
    return _WS_RE.sub(" ", _PUA_RE.sub("", text)).strip()


def code_normalise(text: str) -> str:
    """Code-line normalisation: like :func:`tb_normalise` but inner spacing is kept."""
    text = unicodedata.normalize("NFKC", text).translate(_TB_TABLE)
    return _PUA_RE.sub("", text).rstrip()


# --------------------------------------------------------------------------- patterns

NUMBERED_RE = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+(?P<title>\S.*)$")
CHAPTER_WORD_RE = re.compile(r"^(?:Chapter|CHAPTER)\s+(?P<num>\d{1,2})\b[.:]?\s*(?P<title>.*)$")
_CAPTION_RE = re.compile(r"^(?:Fig\.|Figs\.|Figure|Table|Listing)\s*\d", re.IGNORECASE)
_LIST_START_RE = re.compile(r"^(?:\d{1,2}[.)]|[•▪◦∙*]|-|\([a-z0-9]{1,3}\))\s+\S")
_EDGE_LEAD_RE = re.compile(r"^(\d{1,4})(?:\s+\S.*)?$")
_EDGE_TRAIL_RE = re.compile(r"^.*\S\s+(\d{1,4})$")
_MARKER_RE = re.compile(r"^[\d*†‡§]{1,4}\.?$")
_STYLE_SUFFIXES = frozenset(
    {
        "regular",
        "regu",
        "roman",
        "bold",
        "italic",
        "medium",
        "semibold",
        "light",
        "oblique",
        "bolditalic",
        "mediumitalic",
        "boldoblique",
        "it",
        "bd",
        "bi",
        "book",
        "demi",
    }
)


def font_family(font: str) -> str:
    """'ABCDEF+TeXGyreTermesX-Bold' -> 'TeXGyreTermesX'; 'Sf-Kp-Regular' -> 'Sf-Kp'."""
    name = re.sub(r"^[A-Z]{6}\+", "", font)
    parts = name.split("-")
    if len(parts) > 1 and parts[-1].lower() in _STYLE_SUFFIXES:
        parts = parts[:-1]
    return "-".join(parts) or name


@dataclass
class TLine:
    """One visual line with the features the textbook cleaner needs."""

    cline: CLine
    mono_frac: float
    bold_frac: float
    family: str
    kind: str = "text"  # text | code | heading
    level: int | None = None
    number: str | None = None
    title: str | None = None
    code_text: str = ""

    @property
    def page(self) -> int:
        return self.cline.page

    @property
    def text(self) -> str:
        return self.cline.text

    @property
    def size(self) -> float:
        return self.cline.size

    @property
    def style(self) -> tuple[float, bool, str]:
        return (round(self.size, 1), self.bold_frac >= 0.5, self.family)


def _is_mono(span: Span, patterns: list[str]) -> bool:
    name = span.font.lower()
    return any(p in name for p in patterns)


def _to_tline(line: Line, page_height: float, tcfg: TextbookConfig) -> TLine:
    pats = [p.lower() for p in tcfg.monospace_font_patterns]
    n = mono = bold = 0
    fam: Counter[str] = Counter()
    for s in line.spans:
        k = len(s.text.strip())
        if not k:
            continue
        n += k
        if _is_mono(s, pats):
            mono += k
        else:
            fam[font_family(s.font)] += k
        if s.bold:
            bold += k
    raw = line.text
    text = tb_normalise(raw)
    cl = CLine(
        page=line.page,
        text=text,
        x0=line.x0,
        y0=line.y0,
        x1=line.x1,
        y1=line.y1,
        size=line.size,
        bold_chars=bold,
        n_chars=n,
        bold_text=tb_normalise(" ".join(s.text for s in line.spans if s.bold)),
        page_height=page_height,
        raw_text=raw,
        markers=list(line.markers),
        spans=list(line.spans),
    )
    return TLine(
        cline=cl,
        mono_frac=mono / n if n else 0.0,
        bold_frac=bold / n if n else 0.0,
        family=fam.most_common(1)[0][0] if fam else "",
        code_text=code_normalise(raw),
    )


# --------------------------------------------------------------------------- cleaner


@dataclass
class _Group:
    kind: str  # text | code | heading
    lines: list[TLine] = field(default_factory=list)


class TextbookCleaner(Cleaner):
    def __init__(
        self,
        cfg: CleaningConfig,
        tcfg: TextbookConfig,
        doc: DocumentConfig,
        chapters: list[ChapterRange] | None = None,
    ) -> None:
        super().__init__(cfg, doc)
        self.tcfg = tcfg
        self.chapters = [c for c in (chapters or []) if c.found]
        self.header_res += [re.compile(p) for p in doc.header_footer_patterns]
        self.drop_section_res = [re.compile(p) for p in tcfg.drop_sections]
        self.drop_from_res = [re.compile(p) for p in tcfg.drop_from_patterns]

    # ---------------------------------------------------------------- page edges

    @staticmethod
    def _page_number_ids(pages: dict[int, list[TLine]], body: float) -> tuple[set[int], int | None]:
        """id()s of edge lines that carry the printed page number, and the learned offset.
        Lines set larger than the body text are headings ("1 Basics" opening on printed page
        1), never running headers."""
        cands: list[tuple[TLine, int, int]] = []
        for pno, lines in pages.items():
            ordered = sorted((ln for ln in lines if ln.text), key=lambda ln: ln.cline.y0)
            edge = ordered[:2] + ordered[-2:] if len(ordered) > 4 else ordered
            for ln in edge:
                if len(ln.text) > 120 or ln.size > body + 0.5:
                    continue
                m = _EDGE_LEAD_RE.match(ln.text) or _EDGE_TRAIL_RE.match(ln.text)
                if m:
                    cands.append((ln, pno, int(m.group(1))))
        offsets = Counter(p - v for _ln, p, v in cands)
        if not offsets:
            return set(), None
        off, n = offsets.most_common(1)[0]
        if n < max(3, 0.3 * len(pages)):
            return set(), None
        return {id(ln) for ln, p, v in cands if p - v == off}, off

    # ---------------------------------------------------------------- headings

    def _heading_candidates(self, lines: list[TLine], body: float, body_family: str) -> None:
        t = self.tcfg
        big: list[TLine] = []
        for ln in lines:
            if ln.kind != "text":
                continue
            txt = ln.text
            letters = sum(1 for c in txt if c.isalpha())
            if letters < 2 or len(txt) > t.max_heading_chars or _CAPTION_RE.match(txt):
                continue
            if txt.endswith(";"):
                continue
            if ln.size >= body + t.heading_size_delta:
                ln.kind = "heading"
                big.append(ln)
        families = {ln.family for ln in big if ln.family and ln.family != body_family}
        for ln in lines:
            if ln.kind != "text" or not ln.family or ln.family not in families:
                continue
            if ln.size < body - 0.5 or len(ln.text) > t.max_heading_chars:
                continue
            m = NUMBERED_RE.match(ln.text)
            if (m and "." in m.group("num")) or CHAPTER_WORD_RE.match(ln.text):
                ln.kind = "heading"

    def _merge_headings(self, lines: list[TLine]) -> list[TLine]:
        out: list[TLine] = []
        for ln in lines:
            prev = out[-1] if out else None
            if (
                prev is not None
                and ln.kind == "heading"
                and prev.kind == "heading"
                and ln.page == prev.page
            ):
                gap = ln.cline.y0 - prev.cline.y1
                chapter_word = bool(re.fullmatch(r"(?:Chapter|CHAPTER)\s+\d{1,2}", prev.text))
                wrapped = (
                    prev.style == ln.style
                    and gap < 0.8 * ln.cline.height
                    and not NUMBERED_RE.match(ln.text)
                    and not CHAPTER_WORD_RE.match(ln.text)
                )
                if (chapter_word and gap < 4 * ln.cline.height) or wrapped:
                    prev.cline.text = f"{prev.text} {ln.text}"
                    prev.cline.y1 = max(prev.cline.y1, ln.cline.y1)
                    if chapter_word:  # the title line carries the chapter's style
                        prev.cline.size = max(prev.size, ln.size)
                    continue
            out.append(ln)
        return out

    def _assign_levels(self, lines: list[TLine]) -> None:
        heads = [ln for ln in lines if ln.kind == "heading"]
        for ln in heads:
            cm = CHAPTER_WORD_RE.match(ln.text)
            nm = NUMBERED_RE.match(ln.text)
            if cm and cm.group("title").strip():
                ln.number, ln.title, ln.level = cm.group("num"), cm.group("title").strip(), 1
            elif nm:
                ln.number, ln.title = nm.group("num"), nm.group("title").strip()
                ln.level = ln.number.count(".") + 1
            else:
                ln.title = ln.text
        by_style: dict[tuple[float, bool, str], Counter[int]] = defaultdict(Counter)
        for ln in heads:
            if ln.level is not None:
                by_style[ln.style][ln.level] += 1
        numbered_sizes = [(st[0], c.most_common(1)[0][0]) for st, c in by_style.items()]
        unnumbered_sizes = sorted(
            {round(ln.size, 1) for ln in heads if ln.level is None and ln.style not in by_style},
            reverse=True,
        )
        for ln in heads:
            if ln.level is not None:
                continue
            if ln.style in by_style:
                ln.level = by_style[ln.style].most_common(1)[0][0]
            elif numbered_sizes:
                deeper = [d for s, d in numbered_sizes if s > ln.size + 0.2]
                ln.level = min(6, (max(deeper) if deeper else 0) + 1)
            else:
                ln.level = min(6, unnumbered_sizes.index(round(ln.size, 1)) + 1)
        # the first heading on a selected chapter's first page is that chapter
        for ch in self.chapters:
            first = next((ln for ln in heads if ln.page == ch.start), None)
            if first is None:
                self.warnings.append(f"no heading found on the first page of chapter {ch.title!r}")
                continue
            if normalise_title(first.title or first.text) != normalise_title(ch.title):
                self.warnings.append(
                    f"chapter {ch.title!r}: first heading on p.{ch.start} is {first.text!r}"
                )
            first.level = 1
            self.stats["chapters_detected"] += 1

    def _mark_pseudocode(self, lines: list[TLine]) -> None:
        """Pseudocode set in the text font: a short line indented past the page's text margin
        that does not end a sentence and is not a list item becomes a code line."""
        by_page: dict[int, list[TLine]] = defaultdict(list)
        for ln in lines:
            if ln.kind == "text":
                by_page[ln.page].append(ln)
        for plines in by_page.values():
            lefts = Counter(round(ln.cline.x0) for ln in plines)
            common = [x for x, n in lefts.items() if n >= max(2, 0.1 * len(plines))]
            if not common:
                continue  # too few text lines on this page to know its margin
            left = min(common)
            right = max(ln.cline.x1 for ln in plines)
            width = max(right - left, 1.0)
            for ln in plines:
                cl = ln.cline
                if (
                    cl.x0 >= left + 10
                    and (cl.x1 - cl.x0) < 0.6 * width
                    and not _LIST_START_RE.match(ln.text)
                    and not ln.text.rstrip().endswith((".", "?", "!", ":"))
                ):
                    ln.kind = "code"
                    self.stats["pseudocode_lines"] += 1

    # ---------------------------------------------------------------- dropping

    def _drop_sections(self, lines: list[TLine]) -> list[TLine]:
        kept: list[TLine] = []
        drop_level: int | None = None
        dropping_items = False
        dropped: list[str] = []
        for ln in lines:
            if ln.kind == "heading":
                lvl = ln.level or 6
                if drop_level is not None and lvl > drop_level:
                    self.stats["dropped_section_lines"] += 1
                    continue
                drop_level = None
                dropping_items = False
                title = (ln.title or ln.text).strip()
                if any(r.search(title) for r in self.drop_section_res):
                    drop_level = lvl
                    dropped.append(f"{ln.number + ' ' if ln.number else ''}{title} (p.{ln.page})")
                    self.stats["dropped_section_lines"] += 1
                    continue
                kept.append(ln)
                continue
            if drop_level is not None:
                self.stats["dropped_section_lines"] += 1
                continue
            if not dropping_items and ln.kind == "text":
                dropping_items = any(r.search(ln.text) for r in self.drop_from_res)
            if dropping_items:
                self.stats["dropped_exercise_lines"] += 1
                continue
            kept.append(ln)
        self.stats["sections_dropped"] = dropped
        return kept

    # ---------------------------------------------------------------- main

    def clean_textbook(self, raw: RawDocument) -> CleanDocument:
        st = self.stats
        t = self.tcfg
        st["pages"] = raw.page_count
        st["pages_kept"] = sorted(p.number for p in raw.pages)
        st["rotated_lines_dropped"] = sum(p.rotated_lines_dropped for p in raw.pages)
        pages: dict[int, list[TLine]] = {
            p.number: [_to_tline(ln, p.height, t) for ln in p.lines] for p in raw.pages
        }
        for pno in pages:
            pages[pno] = [ln for ln in pages[pno] if ln.text or ln.code_text.strip()]

        all_lines = [ln for pno in sorted(pages) for ln in pages[pno]]
        sizes: Counter[float] = Counter()
        for ln in all_lines:
            if ln.mono_frac < t.code_min_mono_fraction:
                sizes[round(ln.size, 1)] += max(len(ln.text), 1)
        body = sizes.most_common(1)[0][0] if sizes else 10.0

        pn_ids, offset = self._page_number_ids(pages, body)
        st["printed_page_offset"] = offset
        hf_ids = self._header_footer_ids({p: [ln.cline for ln in v] for p, v in pages.items()})
        stream: list[TLine] = []
        for pno in sorted(pages):
            for ln in pages[pno]:
                cl = ln.cline
                if id(ln) in pn_ids:
                    st["page_number_lines_removed"] += 1
                    continue
                if id(cl) in hf_ids or any(r.search(ln.text) for r in self.header_res):
                    st["header_footer_lines_removed"] += 1
                    continue
                band = self.cfg.page_number_band * cl.page_height
                in_band = cl.y0 < band or cl.y1 > cl.page_height - band
                if in_band and _PAGE_NUMBER_RE.match(ln.text):
                    st["page_number_lines_removed"] += 1
                    continue
                if _MARKER_RE.match(ln.text) and ln.size < t.small_number_ratio * body:
                    st["margin_markers_removed"] += 1
                    continue
                if any(r.search(ln.text) for r in self.drop_res):
                    st["pattern_lines_removed"] += 1
                    continue
                if ln.mono_frac >= t.code_min_mono_fraction and ln.code_text.strip():
                    ln.kind = "code"
                elif not ln.text:
                    continue
                stream.append(ln)

        # footnotes: small text at the bottom of a page (never code)
        text_lines = [ln.cline for ln in stream if ln.kind == "text"]
        kept_lines, fn_lines = self._split_footnotes(text_lines, body)
        kept_ids = {id(c) for c in kept_lines}
        stream = [ln for ln in stream if ln.kind == "code" or id(ln.cline) in kept_ids]
        footnotes = self._group_footnotes(fn_lines)
        st["footnotes"] = len(footnotes)

        fam: Counter[str] = Counter()
        for ln in stream:
            if ln.kind == "text" and abs(ln.size - body) < 0.3 and ln.family:
                fam[ln.family] += len(ln.text)
        body_family = fam.most_common(1)[0][0] if fam else ""
        self._heading_candidates(stream, body, body_family)
        stream = self._merge_headings(stream)
        self._assign_levels(stream)
        if self.doc.pseudocode_by_indent:
            self._mark_pseudocode(stream)
        stream = self._drop_sections(stream)

        heads = [ln for ln in stream if ln.kind == "heading"]
        st["headings"] = len(heads)
        st["headings_by_level"] = dict(sorted(Counter(ln.level or 0 for ln in heads).items()))
        st["code_lines"] = sum(1 for ln in stream if ln.kind == "code")
        st["body_font_size"] = body
        st["body_font_family"] = body_family

        paragraphs, text, page_spans = self._build(stream)
        st["code_blocks"] = sum(1 for p in paragraphs if p.kind == "code")
        st["paragraphs"] = len(paragraphs)
        st["chars"] = len(text)
        if not text:
            self.warnings.append("no body text left after cleaning")
        return CleanDocument(
            doc_id=self.doc.doc_id,
            text=text,
            paragraphs=paragraphs,
            footnotes=footnotes,
            page_spans=page_spans,
            body_font_size=body,
            page_count=raw.page_count,
            stats=dict(st),
            warnings=list(self.warnings),
        )

    # ---------------------------------------------------------------- paragraphs

    def _groups(self, stream: list[TLine]) -> list[_Group]:
        groups: list[_Group] = []
        left: float | None = None  # body text's left margin on the current page
        for ln in stream:
            prev = groups[-1].lines[-1] if groups and groups[-1].lines else None
            if ln.kind == "heading":
                groups.append(_Group("heading", [ln]))
                continue
            if ln.kind == "code":
                if (
                    groups
                    and groups[-1].kind == "code"
                    and prev is not None
                    and (
                        ln.page != prev.page or ln.cline.y0 - prev.cline.y1 < 2.0 * ln.cline.height
                    )
                ):
                    groups[-1].lines.append(ln)
                else:
                    groups.append(_Group("code", [ln]))
                continue
            new = not groups or groups[-1].kind != "text" or prev is None
            if not new and prev is not None:
                cl, pc = ln.cline, prev.cline
                if _LIST_START_RE.match(ln.text):
                    new = True
                elif cl.page == pc.page:
                    gap = cl.y0 - pc.y1
                    new = gap > self.cfg.paragraph_gap_ratio * min(cl.height, pc.height) + 1.0
                if not new and left is not None and cl.x0 > left + 8 and cl.x0 > pc.x0 + 8:
                    new = True  # first-line indentation
            if new:
                groups.append(_Group("text", [ln]))
            else:
                groups[-1].lines.append(ln)
            if left is None or ln.page != (prev.page if prev else None):
                left = ln.cline.x0
            else:
                left = min(left, ln.cline.x0)
        return groups

    def _build(
        self, stream: list[TLine]
    ) -> tuple[list[Paragraph], str, list[tuple[int, int, int]]]:
        vocab: Counter[str] = Counter()
        for ln in stream:
            if ln.kind == "text":
                vocab.update(w.lower() for w in _WORD_RE.findall(ln.text))
        paragraphs: list[Paragraph] = []
        parts: list[str] = []
        page_spans: list[tuple[int, int, int]] = []
        offset = 0
        for g in self._groups(stream):
            starts: list[tuple[int, int]]
            if g.kind == "code":
                text, starts = _join_code(g.lines)
            elif g.kind == "heading":
                ln = g.lines[0]
                text, starts = ln.text, [(0, ln.page)]
            else:
                text, starts = _join_lines([ln.cline for ln in g.lines], vocab)
            if not text.strip():
                continue
            if parts:
                parts.append("\n")
                offset += 1
            start = offset
            parts.append(text)
            offset += len(text)
            for (rel, page), nxt in zip(starts, [*starts[1:], (len(text), -1)], strict=True):
                seg = (start + rel, start + nxt[0], page)
                if page_spans and page_spans[-1][2] == page and page_spans[-1][1] >= seg[0] - 1:
                    page_spans[-1] = (page_spans[-1][0], seg[1], page)
                else:
                    page_spans.append(seg)
            first = g.lines[0]
            n_chars = sum(ln.cline.n_chars for ln in g.lines) or 1
            paragraphs.append(
                Paragraph(
                    text=text,
                    char_start=start,
                    char_end=offset,
                    page_start=first.page,
                    page_end=g.lines[-1].page,
                    x0=first.cline.x0,
                    size=first.size,
                    bold_ratio=sum(ln.cline.bold_chars for ln in g.lines) / n_chars,
                    bold_text=" ".join(ln.cline.bold_text for ln in g.lines if ln.cline.bold_text),
                    n_lines=len(g.lines),
                    kind=g.kind,
                    level=first.level if g.kind == "heading" else None,
                    number=first.number if g.kind == "heading" else None,
                    title=first.title if g.kind == "heading" else None,
                )
            )
        return paragraphs, "".join(parts), page_spans


def _join_code(lines: list[TLine]) -> tuple[str, list[tuple[int, int]]]:
    """Code lines joined with newlines; indentation is rebuilt from the x positions."""
    widths: list[float] = []
    for ln in lines:
        for s in ln.cline.spans:
            k = len(s.text)
            if k and s.text.strip():
                widths.append((s.x1 - s.x0) / k)
    char_w = sorted(widths)[len(widths) // 2] if widths else 5.0
    left = min(ln.cline.x0 for ln in lines)
    out = ""
    starts: list[tuple[int, int]] = []
    for ln in lines:
        indent = max(0, round((ln.cline.x0 - left) / max(char_w, 1.0)))
        body = ln.code_text.strip()
        if out:
            out += "\n"
        starts.append((len(out), ln.page))
        out += " " * indent + body
    return out, starts


def clean_textbook(
    raw: RawDocument,
    doc: DocumentConfig,
    cfg: CleaningConfig,
    tcfg: TextbookConfig,
    chapters: list[ChapterRange] | None = None,
) -> CleanDocument:
    """Run the textbook cleaning pipeline for one configured document."""
    return TextbookCleaner(cfg, tcfg, doc, chapters).clean_textbook(raw)


def chapter_report(chapters: list[ChapterRange]) -> list[dict[str, Any]]:
    return [c.to_dict() for c in chapters]
