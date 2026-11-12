"""Chapter selection by title for long PDFs (textbooks).

Page ranges come from the PDF outline (bookmarks) when it lists the chapter: the chapter starts on
its outline page and ends on the page before the next outline entry of the same or a higher
level. Without an outline entry, chapter starts are found by heading detection: a page whose top
half holds a line set at least ``1.4 x`` the body font size that matches the title. Every
requested title is reported, found or not, with the method and the resolved 1-based page range.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf

_CHAPTER_PREFIX_RE = re.compile(r"^(?:chapter|part|appendix)\s+[0-9ivxlcA-Z]+\b[.:]?\s*", re.I)
_NUMBER_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+")


def normalise_title(title: str) -> str:
    """Comparable form of a chapter title: case-folded, numbering and punctuation removed."""
    t = unicodedata.normalize("NFKC", title).replace("’", "'").strip()
    t = _CHAPTER_PREFIX_RE.sub("", t)
    t = _NUMBER_PREFIX_RE.sub("", t)
    t = re.sub(r"[^\w\s']", " ", t.lower())
    return " ".join(t.split())


@dataclass
class ChapterRange:
    title: str
    found: bool
    method: str  # "outline" | "headings" | "not_found"
    start: int | None = None  # 1-based PDF pages, inclusive
    end: int | None = None
    matched: str | None = None  # the outline entry / heading text that matched

    @property
    def pages(self) -> list[int]:
        if self.start is None or self.end is None:
            return []
        return list(range(self.start, self.end + 1))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_pages"] = len(self.pages)
        return d


def _from_outline(
    toc: list[list[Any]], wanted: str, page_count: int
) -> tuple[int, int, str] | None:
    key = normalise_title(wanted)
    hits = [i for i, (_lvl, title, _p) in enumerate(toc) if normalise_title(str(title)) == key]
    if not hits:
        return None
    i = min(hits, key=lambda j: (int(toc[j][0]), j))  # the outermost entry with that title
    level, title, start = int(toc[i][0]), str(toc[i][1]), int(toc[i][2])
    if start < 1:
        return None
    end = page_count
    for lvl, _t, page in toc[i + 1 :]:
        if int(lvl) <= level and int(page) > start:
            end = int(page) - 1
            break
    return start, max(start, end), title


def _page_big_lines(page: Any) -> tuple[list[tuple[float, float, str]], list[float]]:
    """(size, y0, text) of every line, and the per-character sizes (for the body size)."""
    lines: list[tuple[float, float, str]] = []
    sizes: list[float] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for ln in block.get("lines", []):
            spans = [s for s in ln.get("spans", []) if str(s.get("text", "")).strip()]
            if not spans:
                continue
            text = " ".join(str(s["text"]).strip() for s in spans)
            size = max(float(s.get("size", 0.0)) for s in spans)
            lines.append((size, float(ln["bbox"][1]), text))
            for s in spans:
                sizes.extend([round(float(s.get("size", 0.0)), 1)] * len(str(s["text"]).strip()))
    return lines, sizes


def _chapter_starts_by_headings(doc: Any) -> list[tuple[int, float, str]]:
    """(page, size, heading text) of pages with a large line (>= 1.4 x body) in their top half;
    the text joins the page's lines of that largest size."""
    all_sizes: Counter[float] = Counter()
    per_page: list[list[tuple[float, float, str]]] = []
    for index in range(doc.page_count):
        page = doc[index]
        lines, sizes = _page_big_lines(page)
        all_sizes.update(sizes)
        per_page.append([(s, y / max(float(page.rect.height), 1.0), t) for s, y, t in lines])
    if not all_sizes:
        return []
    body = all_sizes.most_common(1)[0][0]
    starts: list[tuple[int, float, str]] = []
    for index, lines in enumerate(per_page):
        big = [(s, t) for s, y, t in lines if s >= 1.4 * body and y < 0.5]
        if big:
            top = max(s for s, _t in big)
            starts.append((index + 1, top, " ".join(t for s, t in big if s >= top - 0.5)))
    return starts


def outline_chapter_titles(path: str | Path, pattern: str) -> list[str]:
    """Titles of the level-1 outline entries matching ``pattern`` (in outline order), e.g.
    ``^Chapter\\s+\\d+`` for every numbered chapter of a book without its front and back
    matter. An empty list when the PDF has no outline."""
    rx = re.compile(pattern)
    with pymupdf.open(Path(path)) as doc:  # type: ignore[no-untyped-call]
        toc = doc.get_toc(simple=True) or []
    out: list[str] = []
    for level, title, page in toc:
        t = " ".join(str(title).split())
        if int(level) == 1 and int(page) >= 1 and rx.search(t) and t not in out:
            out.append(t)
    return out


def resolve_chapters(path: str | Path, titles: list[str]) -> list[ChapterRange]:
    """Resolve chapter titles to page ranges (outline first, then heading detection)."""
    out: list[ChapterRange] = []
    with pymupdf.open(Path(path)) as doc:  # type: ignore[no-untyped-call]
        toc = doc.get_toc(simple=True) or []
        page_count = int(doc.page_count)
        heading_starts: list[tuple[int, float, str]] | None = None
        for wanted in titles:
            hit = _from_outline(toc, wanted, page_count) if toc else None
            if hit is not None:
                start, end, matched = hit
                out.append(ChapterRange(wanted, True, "outline", start, end, matched))
                continue
            if heading_starts is None:
                heading_starts = _chapter_starts_by_headings(doc)
            key = normalise_title(wanted)
            found = None
            for j, (page, size, text) in enumerate(heading_starts):
                norm = normalise_title(text)
                if norm == key or norm.endswith(" " + key) or norm.startswith(key + " "):
                    # the chapter ends before the next page with a heading as large as its title
                    end = page_count
                    for page2, size2, _t in heading_starts[j + 1 :]:
                        if size2 >= size - 0.5:
                            end = page2 - 1
                            break
                    found = (page, max(page, end), text)
                    break
            if found is None:
                out.append(ChapterRange(wanted, False, "not_found"))
            else:
                out.append(ChapterRange(wanted, True, "headings", found[0], found[1], found[2]))
    return out
