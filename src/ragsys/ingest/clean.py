"""Cleaning: turn positioned PDF lines into normalised paragraphs with exact provenance.

Steps (all layout-aware, domain-agnostic; corpus specifics live in configs/ingest.yaml):
  1. normalise text (unicode quotes/dashes, whitespace, footnote asterisks)
  2. drop pages that are predominantly Devanagari, then Devanagari lines (> threshold of letters)
  3. drop running headers/footers (repeated text at similar y on >= 50% of pages, or of the
     pages with the same parity) and standalone page numbers
  4. drop the table of contents (ARRANGEMENT OF SECTIONS ...) block
  5. apply document start/end markers (front matter, signature blocks, other notifications)
  6. separate footnotes (small font at the bottom of a page) into footnote records
  7. build paragraphs (structural markers, vertical gaps), de-hyphenate, join wrapped lines
  8. emit one cleaned text per document with paragraph/page character offsets
"""

from __future__ import annotations

import bisect
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ragsys.config import CleaningConfig, DocumentConfig
from ragsys.ingest.extract import Line, RawDocument, Span

# --------------------------------------------------------------------------- normalisation

DEVANAGARI_CHAR_RE = re.compile(r"[ऀ-ॿ꣠-ꣿ᳐-᳿]")
_DEVANAGARI_RANGE = ("ऀ", "ॿ")

_QUOTE_TABLE = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
        "‖": '"',  # Gazette font artefact used as a closing double quote
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "­": None,  # soft hyphen
        "​": None,
        "‌": None,
        "‍": None,
        "﻿": None,
        " ": " ",
        " ": " ",
        "‐": "-",
        "‑": "-",
        "\t": " ",
    }
)
_DASH_RE = re.compile("[‒–—―−﹘﹣－]")
# Gazette artefact: U+2015 HORIZONTAL BAR used as an opening double quote before a word.
_BAR_QUOTE_RE = re.compile("(?:(?<=^)|(?<=[\\s(\\[]))―(?=[A-Za-z0-9])")
_MULTI_HYPHEN_RE = re.compile(r"-{2,}")
_COLON_HYPHEN_RE = re.compile(r":-(?=\s|$|\()")
_DASH_SPACE_RE = re.compile(r"\s*—\s*")
_FOOTNOTE_STAR_RE = re.compile(r"(?<=[A-Za-z\)\]])\*+(?=[\s.,;:)\]]|$)")
_WS_RE = re.compile(r"\s+")

# Hindi clause labels (क), (ख), ... occasionally used inside the English text of Gazette
# copies; transliterated to (a), (b), ... in Hindi alphabetical order.
_HINDI_LABELS = "कखगघङचछजझञटठडढणतथदधनपफबभमय"
_HINDI_LABEL_RE = re.compile(r"\(\s*([क-ह])\s*\)")


def normalise_text(text: str) -> str:
    """Unicode/typographic normalisation that preserves legal clause markers."""
    text = unicodedata.normalize("NFKC", text)
    text = _BAR_QUOTE_RE.sub('"', text)
    text = text.translate(_QUOTE_TABLE)
    text = _DASH_RE.sub("—", text)
    text = _MULTI_HYPHEN_RE.sub("—", text)
    text = _COLON_HYPHEN_RE.sub(":—", text)
    text = _DASH_SPACE_RE.sub("—", text)
    text = _FOOTNOTE_STAR_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def devanagari_fraction(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    lo, hi = _DEVANAGARI_RANGE
    return sum(1 for c in letters if lo <= c <= hi) / len(letters)


def has_letters(text: str) -> bool:
    return any(c.isalpha() for c in text)


def transliterate_hindi_labels(text: str) -> tuple[str, int]:
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        idx = _HINDI_LABELS.find(m.group(1))
        if idx < 0:
            return m.group(0)
        count += 1
        return f"({chr(ord('a') + idx)})"

    return _HINDI_LABEL_RE.sub(repl, text), count


def strip_devanagari(text: str) -> tuple[str, int]:
    n = len(DEVANAGARI_CHAR_RE.findall(text))
    if not n:
        return text, 0
    return _WS_RE.sub(" ", DEVANAGARI_CHAR_RE.sub("", text)).strip(), n


# --------------------------------------------------------------------------- data model


@dataclass
class CLine:
    """A line after normalisation, with the layout facts the cleaner needs."""

    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold_chars: int
    n_chars: int
    bold_text: str
    page_height: float
    raw_text: str
    markers: list[str] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@dataclass
class Paragraph:
    text: str
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    x0: float
    size: float
    bold_ratio: float
    bold_text: str
    n_lines: int


@dataclass
class Footnote:
    doc_id: str
    footnote_id: str
    page: int
    marker: str | None
    text: str
    section_ids: list[str] = field(default_factory=list)


@dataclass
class CleanDocument:
    doc_id: str
    text: str
    paragraphs: list[Paragraph]
    footnotes: list[Footnote]
    page_spans: list[tuple[int, int, int]]  # (char_start, char_end, page)
    body_font_size: float
    page_count: int
    stats: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def page_at(self, offset: int) -> int:
        if not self.page_spans:
            return 1
        starts = [s for s, _e, _p in self.page_spans]
        i = max(bisect.bisect_right(starts, offset) - 1, 0)
        return self.page_spans[i][2]

    def pages_for(self, start: int, end: int) -> tuple[int, int]:
        return self.page_at(start), self.page_at(max(start, end - 1))


# --------------------------------------------------------------------------- helpers

# Standalone page numbers: "7", "(1)", "Page 3 of 9". No trailing full stop and at most three
# digits, so a wrapped year such as "2020." at the top of a page is kept as text.
_PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s+)?[\(\[]?\s*\d{1,3}\s*[\)\]]?(?:\s+of\s+\d{1,4})?$", re.IGNORECASE
)
_TOC_ENTRY_RE = re.compile(r"^\d{1,3}[A-Z]{0,2}\.\s+\S")
_TOC_CHAPTER_RE = re.compile(r"^(?:CHAPTER|PART)\s+[IVXLC]+[A-Z]?\.?$")
_TOC_SECTIONS_RE = re.compile(r"^(?:SECTIONS?|RULES?|CLAUSES?|REGULATIONS?)\.?$")
_TOC_RULE_RE = re.compile(r"^[_\-—=.·\s]+$")
_CHAPTER_RE = re.compile(r"^(?:CHAPTER|PART)\s+[IVXLC]+[A-Z]?\b")
_SCHEDULE_RE = re.compile(r"^(?:THE\s+)?(?:[A-Z]+\s+)?SCHEDULE\b")

MARKER_START_RE = re.compile(
    r"""^(?:
        (?:\[\s*)*\(\s*(?:\d{1,3}[A-Z]{0,2}|[a-z]{1,3}|[A-Z]{1,2})\s*\)  # (1) (1A) (a) (iv) (za) (A)
      | (?:\[\s*)*\d{1,3}[A-Z]{0,2}\.\s+[\[A-Z(]                       # 12. Heading / 2A. Heading
      | Explanation\b | Provided\b | Illustrations?\b
      | (?:CHAPTER|PART)\s+[IVXLC]+
      | (?:THE\s+)?(?:[A-Z]+\s+)?SCHEDULE\b
    )""",
    re.VERBOSE,
)
# A wrapped cross-reference ("...sub-section\n(1) of section 34") is not a new item; note that
# "(b) to call for ..." IS an item, so "to" only counts when a range follows: "(a) to (c)".
_REF_CONTINUATION_RE = re.compile(
    r"^(?:\[\s*)*\(\s*\w{1,5}\s*\)\s*(?:of\b|and\b|or\b|read\b|,|to\s+\()"
)
_FOOTNOTE_START_RE = re.compile(r"^(\*+\.?|\d{1,3}[.)]|[a-z][.)])\s+\S")
_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
# fmt: off
_COMPOUND_PREFIXES = frozenset({
    "sub", "non", "self", "multi", "co", "ex", "e", "pre", "post", "anti", "inter", "intra",
    "cross", "well", "ill", "semi", "quasi", "counter", "over", "under", "mid", "vice", "cum",
    "bi", "tri", "tele", "full", "half", "part", "long", "short",
})
# fmt: on


def is_all_caps_heading(text: str, max_len: int = 120) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and len(text) <= max_len and all(c.isupper() for c in letters)


def _heading_like(text: str) -> bool:
    return bool(_CHAPTER_RE.match(text) or _SCHEDULE_RE.match(text)) or is_all_caps_heading(text)


def _norm_key(text: str) -> str:
    return re.sub(r"\d+", "#", _WS_RE.sub(" ", text).strip().lower())


def _to_cline(line: Line, page_height: float) -> CLine:
    bold_parts = [s.text for s in line.spans if s.bold and s.text.strip()]
    n_chars = sum(len(s.text.strip()) for s in line.spans)
    bold_chars = sum(len(s.text.strip()) for s in line.spans if s.bold)
    return CLine(
        page=line.page,
        text=normalise_text(line.text),
        x0=line.x0,
        y0=line.y0,
        x1=line.x1,
        y1=line.y1,
        size=line.size,
        bold_chars=bold_chars,
        n_chars=n_chars,
        bold_text=normalise_text(" ".join(bold_parts)),
        page_height=page_height,
        raw_text=line.text,
        markers=list(line.markers),
        spans=list(line.spans),
    )


# --------------------------------------------------------------------------- cleaner


class Cleaner:
    def __init__(self, cfg: CleaningConfig, doc: DocumentConfig) -> None:
        self.cfg = cfg
        self.doc = doc
        self.toc_res = [re.compile(p) for p in cfg.toc_markers]
        self.header_res = [re.compile(p) for p in cfg.header_footer_patterns]
        self.drop_res = [re.compile(p) for p in cfg.drop_line_patterns]
        self.start_re = re.compile(doc.start_marker) if doc.start_marker else None
        self.end_re = re.compile(doc.end_marker) if doc.end_marker else None
        self.stats: dict[str, Any] = defaultdict(int)
        self.warnings: list[str] = []

    # ---------------------------------------------------------------- page-level filters

    def _devanagari_pages(self, raw: RawDocument) -> set[int]:
        dropped = set()
        for page in raw.pages:
            if devanagari_fraction(page.text) > self.cfg.devanagari_page_threshold:
                dropped.add(page.number)
        return dropped

    def _header_footer_ids(self, pages: dict[int, list[CLine]]) -> set[int]:
        """Return id()s of lines that are running headers/footers."""
        n_pages = len(pages)
        parity_counts = Counter(p % 2 for p in pages)
        occurrences: dict[str, list[tuple[int, float, CLine]]] = defaultdict(list)
        for pno, lines in pages.items():
            for ln in lines:
                if ln.text:
                    occurrences[_norm_key(ln.text)].append((pno, ln.y0, ln))
        tol = self.cfg.header_footer_y_tolerance
        frac = self.cfg.header_footer_min_page_fraction
        min_pages = self.cfg.header_footer_min_pages
        band = self.cfg.header_footer_band
        result: set[int] = set()
        for occ in occurrences.values():
            if len(occ) < min_pages:
                continue
            for pno, y0, ln in occ:
                # running headers/footers live in the page margins
                if band * ln.page_height < y0 < (1 - band) * ln.page_height:
                    continue
                same = {p for p, y, _ in occ if abs(y - y0) <= tol}
                same_parity = {p for p in same if p % 2 == pno % 2}
                need_all = max(min_pages, math.ceil(frac * n_pages))
                need_par = max(min_pages, math.ceil(frac * parity_counts[pno % 2]))
                if len(same) >= need_all or len(same_parity) >= need_par:
                    result.add(id(ln))
        return result

    # ---------------------------------------------------------------- TOC

    def _find_toc(self, lines: list[CLine]) -> tuple[int, int] | None:
        start = next(
            (i for i, ln in enumerate(lines) if any(r.match(ln.text) for r in self.toc_res)),
            None,
        )
        if start is None:
            return None

        def lookahead_entry(j: int) -> bool:
            for k in range(j + 1, min(j + 5, len(lines))):
                t = lines[k].text
                if "—" in t:
                    return False
                if _TOC_ENTRY_RE.match(t):
                    return True
            return False

        j = start + 1
        entries = 0
        last_entry_x: float | None = None
        prev_kind = "marker"
        while j < len(lines):
            t = lines[j].text
            nxt = lines[j + 1].text if j + 1 < len(lines) else ""
            if _TOC_ENTRY_RE.match(t):
                if "—" in t:
                    break  # first body heading ("1. Short title.—(1) ...")
                entries += 1
                last_entry_x = lines[j].x0
                prev_kind = "entry"
            elif _TOC_SECTIONS_RE.match(t) or _TOC_RULE_RE.match(t):
                prev_kind = "label"
            elif _TOC_CHAPTER_RE.match(t) and lookahead_entry(j):
                prev_kind = "chapter"
            elif is_all_caps_heading(t) and (
                prev_kind == "chapter"
                or _TOC_SECTIONS_RE.match(nxt)
                or _TOC_CHAPTER_RE.match(nxt)
                or (_TOC_ENTRY_RE.match(nxt) and "—" not in nxt)
            ):
                prev_kind = "title"
            elif (
                prev_kind in ("entry", "wrap")
                and last_entry_x is not None
                and lines[j].x0 > last_entry_x + 4
                and lines[j].page == lines[j - 1].page
                and lines[j].y0 - lines[j - 1].y1 < 0.5 * lines[j].height
                and not is_all_caps_heading(t)
                and "—" not in t
            ):
                prev_kind = "wrap"
            else:
                break
            j += 1
        if entries < 3:
            self.warnings.append(
                f"TOC marker found on page {lines[start].page} but only {entries} entries; not removed"
            )
            return None
        return start, j

    # ---------------------------------------------------------------- marginal headings

    def _attach_marginal_headings(self, lines: list[CLine], body_size: float) -> list[CLine]:
        """Gazette-style marginal notes ("Short title." printed beside "1. (1) This Act ...")
        are folded into the numbered body line as an inline heading ("1. Short title.—(1) ..."),
        the form the structure parser expects.

        Notes printed on the same baseline as body text were merged into that line during
        extraction, so margins are detected per span: the body column is taken from spans in
        the body font size, and spans lying wholly left/right of it are margin text."""
        starts: list[float] = []
        ends: list[float] = []
        for ln in lines:
            body_spans = [s for s in ln.spans if s.text.strip() and s.size >= body_size - 0.5]
            if body_spans and len(ln.text) > 20:
                starts.append(body_spans[0].x0)
                ends.append(body_spans[-1].x1)
        if len(starts) < 5:
            return lines
        starts.sort()
        ends.sort()
        left, right = starts[len(starts) // 20] - 4, ends[(len(ends) * 19) // 20] + 4

        # (line, margin text, line keeps body text?)
        pieces: list[tuple[CLine, str, bool]] = []
        for ln in lines:
            inner = [s for s in ln.spans if not (s.x1 < left or s.x0 > right)]
            outer = [s for s in ln.spans if (s.x1 < left or s.x0 > right) and s.text.strip()]
            if not outer:
                continue
            note = normalise_text(" ".join(s.text for s in outer))
            body_text = normalise_text("".join(s.text for s in inner))
            body_text, _ = transliterate_hindi_labels(body_text)
            body_text, _ = strip_devanagari(body_text)
            if body_text:
                ln.text = body_text
                ln.spans = inner
                ln.x0 = min(s.x0 for s in inner if s.text.strip()) if inner else ln.x0
                ln.bold_text = normalise_text(" ".join(s.text for s in inner if s.bold))
            pieces.append((ln, note, bool(body_text)))
        if not pieces:
            return lines

        blocks: list[list[tuple[CLine, str, bool]]] = []
        for piece in pieces:
            prev = blocks[-1][-1][0] if blocks else None
            ln = piece[0]
            if prev is not None and prev.page == ln.page and 0 <= ln.y0 - prev.y1 < 0.8 * ln.height:
                blocks[-1].append(piece)
            else:
                blocks.append([piece])
        numbered = re.compile(r"^((?:\[\s*)*\d{1,3}[A-Z]{0,2}\.)\s*(.*)$")
        drop: set[int] = {id(ln) for ln, _note, keeps in pieces if not keeps}
        for block in blocks:
            top = block[0][0]
            target = next(
                (
                    ln
                    for ln in lines
                    if id(ln) not in drop
                    and ln.page == top.page
                    and abs(ln.y0 - top.y0) <= 0.6 * max(ln.height, top.height)
                    and numbered.match(ln.text)
                    and "—" not in ln.text[:200]
                ),
                None,
            )
            if target is None:
                self.stats["marginal_notes_dropped"] += 1
                continue
            heading = _join_simple([note for _ln, note, _k in block]).rstrip(" .:—")
            m = numbered.match(target.text)
            assert m is not None
            target.text = f"{m.group(1)} {heading}.—{m.group(2)}"
            target.bold_text = f"{target.bold_text} {heading}".strip()
            self.stats["marginal_headings_attached"] += 1
        return [ln for ln in lines if id(ln) not in drop]

    # ---------------------------------------------------------------- footnotes

    def _split_footnotes(
        self, lines: list[CLine], body_size: float
    ) -> tuple[list[CLine], dict[int, list[CLine]]]:
        limit = body_size * self.cfg.footnote_max_size_ratio
        by_page: dict[int, list[CLine]] = defaultdict(list)
        for ln in lines:
            by_page[ln.page].append(ln)
        fn_ids: set[int] = set()
        fn_lines: dict[int, list[CLine]] = {}
        for pno, plines in by_page.items():
            body_bottom = max((ln.y1 for ln in plines if ln.size > limit), default=0.0)
            cand = [
                ln
                for ln in plines
                if ln.size <= limit and ln.y0 >= body_bottom - 1.0 and ln.y0 > 0.5 * ln.page_height
            ]
            if cand:
                fn_lines[pno] = cand
                fn_ids.update(id(ln) for ln in cand)
        return [ln for ln in lines if id(ln) not in fn_ids], fn_lines

    def _group_footnotes(self, fn_lines: dict[int, list[CLine]]) -> list[Footnote]:
        out: list[Footnote] = []
        for pno in sorted(fn_lines):
            lines = fn_lines[pno]
            min_x = min(ln.x0 for ln in lines)
            groups: list[list[CLine]] = []
            for ln in lines:
                starts = bool(_FOOTNOTE_START_RE.match(ln.text)) and ln.x0 <= min_x + 3
                if starts or not groups:
                    groups.append([ln])
                else:
                    groups[-1].append(ln)
            for k, g in enumerate(groups, start=1):
                text = _join_simple([ln.text for ln in g])
                m = re.match(r"^(\*+|\d{1,3}|[a-z])[.)]?\s+", text)
                marker = m.group(1) if m and _FOOTNOTE_START_RE.match(text) else None
                out.append(
                    Footnote(
                        doc_id=self.doc.doc_id,
                        footnote_id=f"{self.doc.doc_id}:fn-p{pno}-{k}",
                        page=pno,
                        marker=marker,
                        text=text,
                    )
                )
        return out

    # ---------------------------------------------------------------- main

    def clean(self, raw: RawDocument) -> CleanDocument:
        st = self.stats
        st["pages"] = raw.page_count
        st["rotated_lines_dropped"] = sum(p.rotated_lines_dropped for p in raw.pages)
        st["superscript_markers_dropped"] = sum(p.markers_dropped for p in raw.pages)

        pages: dict[int, list[CLine]] = {
            p.number: [_to_cline(ln, p.height) for ln in p.lines] for p in raw.pages
        }
        for pno in pages:
            pages[pno] = [ln for ln in pages[pno] if ln.text]

        # 1) running headers / footers (computed on all pages, incl. Hindi ones)
        hf_ids = self._header_footer_ids(pages)
        deva_pages = self._devanagari_pages(raw)
        st["pages_dropped_devanagari"] = sorted(deva_pages)

        stream: list[CLine] = []
        for pno in sorted(pages):
            if pno in deva_pages:
                continue
            plines = pages[pno]
            kept: list[CLine] = []
            for ln in plines:
                if id(ln) in hf_ids or any(r.match(ln.text) for r in self.header_res):
                    st["header_footer_lines_removed"] += 1
                    continue
                band = self.cfg.page_number_band * ln.page_height
                in_band = ln.y0 < band or ln.y1 > ln.page_height - band
                if in_band and _PAGE_NUMBER_RE.match(ln.text):
                    st["page_number_lines_removed"] += 1
                    continue
                kept.append(ln)
            stream.extend(self._filter_devanagari_lines(kept))

        # 2) table of contents
        toc = self._find_toc(stream)
        if toc is not None:
            a, b = toc
            st["toc_lines_removed"] = b - a
            st["toc_pages"] = sorted({ln.page for ln in stream[a:b]})
            stream = stream[:a] + stream[b:]

        # 3) start / end markers
        if self.start_re is not None:
            idx = next((i for i, ln in enumerate(stream) if self.start_re.search(ln.text)), None)
            if idx is None:
                self.warnings.append(f"start_marker {self.doc.start_marker!r} not found")
            else:
                st["front_matter_lines_removed"] = idx
                stream = stream[idx:]
        if self.end_re is not None:
            idx = next((i for i, ln in enumerate(stream) if self.end_re.search(ln.text)), None)
            if idx is None:
                self.warnings.append(f"end_marker {self.doc.end_marker!r} not found")
            else:
                st["end_matter_lines_removed"] = len(stream) - idx
                st["body_ends_on_page"] = stream[idx].page
                stream = stream[:idx]
        if self.drop_res:
            before = len(stream)
            stream = [ln for ln in stream if not any(r.search(ln.text) for r in self.drop_res)]
            st["pattern_lines_removed"] = before - len(stream)

        # 4) marginal headings -> inline headings, then footnotes
        body_size = _dominant_size(stream)
        stream = self._attach_marginal_headings(stream, body_size)
        stream, fn_lines = self._split_footnotes(stream, body_size)
        footnotes = self._group_footnotes(fn_lines)
        st["footnote_lines"] = sum(len(v) for v in fn_lines.values())
        st["footnotes"] = len(footnotes)

        # 5) paragraphs + text
        paragraphs, text, page_spans = self._build_paragraphs(stream)
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
            body_font_size=body_size,
            page_count=raw.page_count,
            stats=dict(st),
            warnings=list(self.warnings),
        )

    def _filter_devanagari_lines(self, lines: list[CLine]) -> list[CLine]:
        st = self.stats
        thr = self.cfg.devanagari_line_threshold
        is_deva = [devanagari_fraction(ln.text) > thr for ln in lines]
        letterless = [not has_letters(ln.text) for ln in lines]

        def neighbour(i: int, step: int) -> bool | None:
            j = i + step
            while 0 <= j < len(lines):
                if not letterless[j]:
                    return is_deva[j]
                j += step
            return None

        out: list[CLine] = []
        for i, ln in enumerate(lines):
            if is_deva[i]:
                st["devanagari_lines_removed"] += 1
                continue
            if letterless[i]:
                prev_d, next_d = neighbour(i, -1), neighbour(i, 1)
                if (
                    (prev_d and next_d)
                    or (prev_d and next_d is None)
                    or (next_d and prev_d is None)
                ):
                    st["devanagari_lines_removed"] += 1
                    continue
            text, n_lab = transliterate_hindi_labels(ln.text)
            text, n_chars = strip_devanagari(text)
            if n_lab:
                st["devanagari_labels_transliterated"] += n_lab
            if n_chars:
                st["devanagari_chars_stripped"] += n_chars
                st["mixed_lines_cleaned"] += 1
            if not text:
                st["devanagari_lines_removed"] += 1
                continue
            ln.text = text
            out.append(ln)
        return out

    # ---------------------------------------------------------------- paragraphs

    def _starts_paragraph(self, ln: CLine, prev: CLine | None, gap_limit: float) -> bool:
        if prev is None:
            return True
        marker = bool(MARKER_START_RE.match(ln.text)) and not (
            _REF_CONTINUATION_RE.match(ln.text)
            and not prev.text.rstrip().endswith((".", ";", ":", "—"))
        )
        if marker or _heading_like(ln.text) or _heading_like(prev.text):
            return True
        if ln.page != prev.page:
            return False
        gap = ln.y0 - prev.y1
        return gap > gap_limit * min(ln.height, prev.height)

    def _build_paragraphs(
        self, lines: list[CLine]
    ) -> tuple[list[Paragraph], str, list[tuple[int, int, int]]]:
        vocab: Counter[str] = Counter()
        for ln in lines:
            vocab.update(w.lower() for w in _WORD_RE.findall(ln.text))

        groups: list[list[CLine]] = []
        prev: CLine | None = None
        for ln in lines:
            if self._starts_paragraph(ln, prev, self.cfg.paragraph_gap_ratio):
                groups.append([ln])
            else:
                groups[-1].append(ln)
            prev = ln

        paragraphs: list[Paragraph] = []
        parts: list[str] = []
        page_spans: list[tuple[int, int, int]] = []
        offset = 0
        for g in groups:
            text, line_starts = _join_lines(g, vocab)
            if not text:
                continue
            if parts:
                parts.append("\n")
                offset += 1
            start = offset
            parts.append(text)
            offset += len(text)
            for (rel, page), nxt in zip(
                line_starts, [*line_starts[1:], (len(text), -1)], strict=True
            ):
                seg = (start + rel, start + nxt[0], page)
                if page_spans and page_spans[-1][2] == page and page_spans[-1][1] >= seg[0] - 1:
                    page_spans[-1] = (page_spans[-1][0], seg[1], page)
                else:
                    page_spans.append(seg)
            n_chars = sum(ln.n_chars for ln in g) or 1
            paragraphs.append(
                Paragraph(
                    text=text,
                    char_start=start,
                    char_end=offset,
                    page_start=g[0].page,
                    page_end=g[-1].page,
                    x0=g[0].x0,
                    size=_dominant_size(g),
                    bold_ratio=sum(ln.bold_chars for ln in g) / n_chars,
                    bold_text=" ".join(ln.bold_text for ln in g if ln.bold_text),
                    n_lines=len(g),
                )
            )
        return paragraphs, "".join(parts), page_spans


def _dominant_size(lines: list[CLine]) -> float:
    weights: Counter[float] = Counter()
    for ln in lines:
        weights[round(ln.size, 1)] += max(len(ln.text), 1)
    if not weights:
        return 0.0
    return max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _should_dehyphenate(w1: str, w2: str, vocab: Counter[str]) -> bool:
    hyph = f"{w1}-{w2}".lower()
    joined = f"{w1}{w2}".lower()
    if vocab[hyph] > vocab[joined]:
        return False
    if vocab[joined] > 0:
        return True
    return w1.lower() not in _COMPOUND_PREFIXES


def _join_lines(lines: list[CLine], vocab: Counter[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join wrapped lines, de-hyphenating words broken across lines.

    Returns the paragraph text and (offset_in_paragraph, page) for each line start.
    """
    out = ""
    starts: list[tuple[int, int]] = []
    for ln in lines:
        t = ln.text
        if not out:
            starts.append((0, ln.page))
            out = t
            continue
        m = re.search(r"([A-Za-z]+)-$", out)
        w2m = re.match(r"[a-z]+", t)
        if m and w2m and not out.endswith(" -"):
            if _should_dehyphenate(m.group(1), w2m.group(0), vocab):
                out = out[:-1]
            starts.append((len(out), ln.page))
            out += t
        elif out.endswith(("—", "/")) or t.startswith(("—", ",", ".", ";", ":", ")")):
            starts.append((len(out), ln.page))
            out += t
        else:
            out += " "
            starts.append((len(out), ln.page))
            out += t
    return out, starts


def _join_simple(texts: list[str]) -> str:
    out = ""
    for t in texts:
        if out.endswith("-") and t[:1].islower():
            out = out[:-1] + t
        elif out:
            out += " " + t
        else:
            out = t
    return _WS_RE.sub(" ", out).strip()


def clean_document(raw: RawDocument, doc: DocumentConfig, cfg: CleaningConfig) -> CleanDocument:
    """Run the full cleaning pipeline for one configured document."""
    return Cleaner(cfg, doc).clean(raw)
