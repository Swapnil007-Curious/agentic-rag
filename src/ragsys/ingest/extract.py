"""PyMuPDF text extraction that keeps line positions, font sizes and weights.

Layout information is needed downstream to separate running headers/footers, footnotes,
marginal/bold headings and table-of-contents blocks from the body text.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

# PyMuPDF span flags
_FLAG_SUPERSCRIPT = 1
_FLAG_ITALIC = 2
_FLAG_BOLD = 16

_MARKER_TEXT_RE = re.compile(r"^[\d*†‡§]{1,3}$")


@dataclass
class Span:
    text: str
    size: float
    font: str
    bold: bool
    italic: bool
    superscript: bool
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class Line:
    """A visual line: all spans sharing (approximately) one baseline on a page."""

    page: int  # 1-based
    spans: list[Span]
    x0: float
    y0: float
    x1: float
    y1: float
    markers: list[str] = field(default_factory=list)  # superscript footnote markers removed

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def size(self) -> float:
        """Dominant font size, weighted by number of non-space characters."""
        weights: dict[float, int] = {}
        for s in self.spans:
            n = len(s.text.strip())
            if n:
                key = round(s.size, 1)
                weights[key] = weights.get(key, 0) + n
        if not weights:
            return round(max((s.size for s in self.spans), default=0.0), 1)
        return max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]


@dataclass
class Page:
    number: int  # 1-based
    width: float
    height: float
    lines: list[Line]
    rotated_lines_dropped: int = 0
    markers_dropped: int = 0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class RawDocument:
    path: Path
    pages: list[Page]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def n_chars(self) -> int:
        return sum(len(p.text.strip()) for p in self.pages)

    @property
    def has_text_layer(self) -> bool:
        """False for scanned/image-only PDFs (no OCR in scope)."""
        if not self.pages:
            return False
        return self.n_chars >= 20 * len(self.pages) or self.n_chars >= 200


def _make_span(s: dict[str, Any]) -> Span:
    flags = int(s.get("flags", 0))
    font = str(s.get("font", ""))
    x0, y0, x1, y1 = (float(v) for v in s["bbox"])
    return Span(
        text=str(s.get("text", "")),
        size=float(s.get("size", 0.0)),
        font=font,
        bold=bool(flags & _FLAG_BOLD) or "bold" in font.lower(),
        italic=bool(flags & _FLAG_ITALIC) or "italic" in font.lower(),
        superscript=bool(flags & _FLAG_SUPERSCRIPT),
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
    )


def _vertical_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = min(a1, b1) - max(a0, b0)
    denom = max(min(a1 - a0, b1 - b0), 1e-6)
    return inter / denom


def _merge_same_baseline(raw_lines: list[list[Span]], page_no: int) -> list[Line]:
    """Merge PyMuPDF lines that share a baseline (e.g. a rule number and its heading
    printed as separate text objects), then order lines top-to-bottom, left-to-right."""
    items = []
    for spans in raw_lines:
        if not spans:
            continue
        x0 = min(s.x0 for s in spans)
        y0 = min(s.y0 for s in spans)
        x1 = max(s.x1 for s in spans)
        y1 = max(s.y1 for s in spans)
        items.append((y0, x0, y1, x1, spans))
    items.sort(key=lambda t: (round(t[0], 1), t[1]))

    groups: list[list[tuple[float, float, float, float, list[Span]]]] = []
    for it in items:
        placed = False
        for g in reversed(groups[-3:]):
            gy0 = min(m[0] for m in g)
            gy1 = max(m[2] for m in g)
            if _vertical_overlap(it[0], it[2], gy0, gy1) > 0.6:
                g.append(it)
                placed = True
                break
        if not placed:
            groups.append([it])

    lines: list[Line] = []
    for g in groups:
        g.sort(key=lambda m: m[1])
        merged: list[Span] = []
        prev_x1: float | None = None
        for _y0, x0, _y1, x1, sp in g:
            if merged and prev_x1 is not None:
                last = merged[-1].text
                first = sp[0].text if sp else ""
                if x0 - prev_x1 > 0.5 and not last.endswith(" ") and not first.startswith(" "):
                    merged.append(
                        Span(
                            " ",
                            sp[0].size,
                            sp[0].font,
                            False,
                            False,
                            False,
                            prev_x1,
                            sp[0].y0,
                            x0,
                            sp[0].y1,
                        )
                    )
            merged.extend(sp)
            prev_x1 = x1 if prev_x1 is None else max(prev_x1, x1)
        lines.append(
            Line(
                page=page_no,
                spans=merged,
                x0=min(m[1] for m in g),
                y0=min(m[0] for m in g),
                x1=max(m[3] for m in g),
                y1=max(m[2] for m in g),
            )
        )
    lines.sort(key=lambda ln: (round(ln.y0, 1), ln.x0))
    return lines


def _strip_markers(line: Line, max_ratio: float) -> int:
    """Remove superscript/small footnote-marker spans (e.g. 'date¹', '¹[ ²[(1A) ...')."""
    sizes = [s.size for s in line.spans if s.text.strip()]
    if len(sizes) < 2:
        return 0
    body = statistics.median(sizes)
    big = [s.size for s in line.spans if s.text.strip() and s.size > max_ratio * body]
    if not big:
        return 0
    ref = max(big)
    kept: list[Span] = []
    removed = 0
    for s in line.spans:
        t = s.text.strip()
        is_small = s.size <= max_ratio * ref
        # Only digit/symbol markers: superscript ordinals ("20th") must stay in the text.
        if t and is_small and _MARKER_TEXT_RE.match(t):
            line.markers.append(t)
            removed += 1
            continue
        kept.append(s)
    line.spans = kept
    return removed


def _mark_scripts(line: Line, max_ratio: float) -> int:
    """Textbook mode: keep small raised/lowered spans, written as ``^x`` / ``_x`` (math
    exponents and indices such as ``tanh^2(x)``, ``W^(1)``, ``x_i``) instead of dropping them."""
    sizes = [s.size for s in line.spans if s.text.strip()]
    if len(sizes) < 2:
        return 0
    body = statistics.median(sizes)
    big = [s for s in line.spans if s.text.strip() and s.size > max_ratio * body]
    if not big:
        return 0
    ref = max(s.size for s in big)
    base = statistics.median(s.y1 for s in big)
    marked = 0
    for s in line.spans:
        t = s.text.strip()
        if not t or s.size > max_ratio * ref or len(t) > 8:
            continue
        if s.y1 < base - 0.2 * ref:
            s.text = "^" + t
        elif s.y1 > base + 0.12 * ref:
            s.text = "_" + t
        else:
            continue
        marked += 1
    return marked


_PUA_ONLY_RE = re.compile("^\\s*[-][\\s-]*$")


def extract_pdf(
    path: str | Path,
    marker_max_size_ratio: float = 0.8,
    pages: Collection[int] | None = None,
    scripts: str = "strip",
) -> RawDocument:
    """Extract horizontal text lines with span-level font metadata from a PDF.

    ``pages`` (1-based PDF page numbers) restricts extraction to those pages; page numbers in
    the result are always the PDF's own. ``scripts``: "strip" removes small superscript
    footnote markers (statutes); "caret" keeps small raised/lowered spans as ``^x``/``_x`` and
    drops icon-font glyphs (textbooks)."""
    path = Path(path)
    wanted = set(pages) if pages is not None else None
    out_pages: list[Page] = []
    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
        for index in range(doc.page_count):
            if wanted is not None and index + 1 not in wanted:
                continue
            page = doc[index]
            data = page.get_text("dict")
            raw_lines: list[list[Span]] = []
            rotated = 0
            for block in data.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for ln in block.get("lines", []):
                    dx, dy = ln.get("dir", (1.0, 0.0))
                    if abs(float(dy)) > 0.02 or float(dx) < 0.98:
                        rotated += 1  # watermarks / rotated stamps
                        continue
                    spans = [_make_span(s) for s in ln.get("spans", []) if s.get("text")]
                    if scripts == "caret":  # icon-font glyphs (private use area) carry no text
                        spans = [s for s in spans if not _PUA_ONLY_RE.match(s.text)]
                    if spans:
                        raw_lines.append(spans)
            lines = _merge_same_baseline(raw_lines, index + 1)
            if scripts == "caret":
                markers = sum(_mark_scripts(ln, marker_max_size_ratio) for ln in lines)
            else:
                markers = sum(_strip_markers(ln, marker_max_size_ratio) for ln in lines)
            rect = page.rect
            out_pages.append(
                Page(
                    number=index + 1,
                    width=float(rect.width),
                    height=float(rect.height),
                    lines=lines,
                    rotated_lines_dropped=rotated,
                    markers_dropped=markers,
                )
            )
    return RawDocument(path=path, pages=out_pages)


def pdf_page_count(path: str | Path) -> int:
    with pymupdf.open(Path(path)) as doc:  # type: ignore[no-untyped-call]
        return int(doc.page_count)
