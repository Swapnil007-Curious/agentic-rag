"""Textbook layouts of the app-only books (chapters from the PDF outline): a Sphinx chapter
opening ("CHAPTER" / "ONE" / title), a box heading set in the chapter style (OpenStax
"CHEMISTRY MATTERS") that must stay a section, and a licence sentence hyphenated across a line
break."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from ragsys.config import DocumentConfig, IngestConfig
from ragsys.ingest.licence import detect_licence
from ragsys.ingest.pipeline import IngestResult, run_ingest

BOLD, REG = "tibo", "tiro"


def _page(doc: pymupdf.Document) -> tuple[pymupdf.Page, list[float]]:
    return doc.new_page(width=595, height=842), [90.0]


def _put(page: pymupdf.Page, y: list[float], s: str, font: str, size: float, gap: float) -> None:
    page.insert_text((72, y[0]), s, fontname=font, fontsize=size)
    y[0] += gap


def build_outline_book(path: Path) -> Path:
    doc = pymupdf.open()
    body = [
        "This paragraph is ordinary body text of the chapter, long enough to be a paragraph.",
        "It continues on a second line so that the cleaner sees a block of running text.",
    ]
    for word, title, sec, box in (
        ("ONE", "GETTING STARTED", "1.1 First Steps", None),
        ("TWO", "MORE TOPICS", "2.1 Second Steps", "CHEMISTRY MATTERS"),
    ):
        page, y = _page(doc)
        _put(page, y, "CHAPTER", BOLD, 10, 20)
        _put(page, y, word, BOLD, 20, 55)
        _put(page, y, title, BOLD, 20, 40)
        for s in body:
            _put(page, y, s, REG, 10, 14)
        y[0] += 12
        _put(page, y, sec, BOLD, 14, 26)
        for s in body:
            _put(page, y, s, REG, 10, 14)
        if box:
            y[0] += 12
            _put(page, y, box, BOLD, 20, 32)
            for s in body:
                _put(page, y, s, REG, 10, 14)
    doc.set_toc([[1, "Getting Started", 1], [1, "More Topics", 2]])
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="module")
def book(tmp_path_factory: pytest.TempPathFactory) -> IngestResult:
    raw = tmp_path_factory.mktemp("outline_raw")
    build_outline_book(raw / "book.pdf")
    cfg = IngestConfig(
        documents=[
            DocumentConfig(
                file="book.pdf",
                doc_id="ob",
                title="An Outline Book",
                doc_type="textbook",
                parser="generic",
                outline_chapters=".+",
            )
        ],
        sources_format="textbook",
    )
    return run_ingest(cfg, raw, tmp_path_factory.mktemp("outline_out"), write=False)


def test_spelled_out_chapter_numbers(book: IngestResult) -> None:
    ids = {n.node_id: n for n in book.nodes}
    assert ids["ob:1"].heading == "Getting Started" and ids["ob:1"].level == "chapter"
    assert ids["ob:2"].heading == "More Topics" and ids["ob:2"].level == "chapter"
    assert ids["ob:1.1"].parent_id == "ob:1" and ids["ob:2.1"].parent_id == "ob:2"
    assert not any(n.node_id.startswith("ob:h-one") for n in book.nodes)
    warnings = book.report["documents"]["ob"]["warnings"]
    assert not any("first heading" in w for w in warnings), warnings


def test_chapter_styled_box_stays_inside_its_chapter(book: IngestResult) -> None:
    chapters = [n for n in book.nodes if n.level == "chapter"]
    assert [n.node_id for n in chapters] == ["ob:1", "ob:2"]
    box = next(n for n in book.nodes if n.heading == "CHEMISTRY MATTERS")
    assert box.level != "chapter" and box.parent_id == "ob:2"


def test_licence_hyphenated_across_lines(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        (72, 90), "Permission is granted to copy this document under the Cre-", fontsize=10
    )
    page.insert_text(
        (72, 104), "ative Commons Attribution-NonCommercial 3.0 Unported License.", fontsize=10
    )
    doc.save(tmp_path / "lic.pdf")
    doc.close()
    found, _vague = detect_licence(tmp_path / "lic.pdf")
    assert found is not None and found.licence == "CC BY-NC 3.0 Unported" and found.page == 1
