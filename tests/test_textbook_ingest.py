"""Textbook ingestion on a synthetic book: chapter selection by title (outline and heading
fallback), running headers, code blocks, math, dropped sections, numbered and unnumbered node
IDs, cross-references, licence detection, SOURCES.csv and structural chunking."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ragsys.chunking.base import DocInput, heading_path
from ragsys.chunking.structural import StructuralChunker
from ragsys.config import DocumentConfig, IngestConfig
from ragsys.ingest.chapters import normalise_title, resolve_chapters
from ragsys.ingest.licence import detect_licence
from ragsys.ingest.pipeline import IngestResult, run_ingest
from ragsys.ingest.sources import TEXTBOOK_FIELDS, write_textbook_sources_csv
from ragsys.structure.base import Node
from textbook_fixtures import build_textbook_pdf


def _cfg() -> IngestConfig:
    return IngestConfig(
        documents=[
            DocumentConfig(
                file="tbook.pdf",
                doc_id="tb",
                title="The Testing Book",
                doc_type="textbook",
                parser="generic",
                authors="Ann Author, Bo Writer",
                edition="First",
                chapters=["Basics", "Graphs", "Nonexistent"],
                licence="CC BY 4.0",
                licence_url="https://example.invalid/by",
            ),
            DocumentConfig(
                file="tbook_plain.pdf",
                doc_id="tbp",
                title="The Testing Book (no outline)",
                doc_type="textbook",
                parser="generic",
                chapters=["Basics", "Graphs"],
            ),
        ],
        sources_format="textbook",
    )


@pytest.fixture(scope="module")
def book(tmp_path_factory: pytest.TempPathFactory) -> tuple[IngestResult, Path, Path]:
    raw = tmp_path_factory.mktemp("tb_raw")
    build_textbook_pdf(raw / "tbook.pdf")
    build_textbook_pdf(raw / "tbook_plain.pdf", with_outline=False)
    processed = tmp_path_factory.mktemp("tb_processed")
    return run_ingest(_cfg(), raw, processed, write=True), raw, processed


def _nodes(res: IngestResult, doc: str = "tb") -> dict[str, Node]:
    return {n.node_id: n for n in res.nodes if n.doc_id == doc}


def test_chapter_selection_by_title(book: tuple[IngestResult, Path, Path]) -> None:
    res, raw, _p = book
    rep = res.report["documents"]["tb"]
    ranges = {r["title"]: r for r in rep["chapter_ranges"]}
    assert (ranges["Basics"]["start"], ranges["Basics"]["end"], ranges["Basics"]["method"]) == (
        4,
        6,
        "outline",
    )
    assert (ranges["Graphs"]["start"], ranges["Graphs"]["end"]) == (7, 7)
    assert not ranges["Nonexistent"]["found"] and rep["chapters_not_found"] == ["Nonexistent"]
    assert any("Nonexistent" in w for w in rep["warnings"])
    assert rep["pages_kept"] == 4 and rep["pages"] == 9
    # no outline: the same ranges from heading detection
    plain = {r["title"]: r for r in res.report["documents"]["tbp"]["chapter_ranges"]}
    assert (plain["Basics"]["start"], plain["Basics"]["end"], plain["Basics"]["method"]) == (
        4,
        6,
        "headings",
    )
    assert (plain["Graphs"]["start"], plain["Graphs"]["end"]) == (7, 7)
    assert [r.found for r in resolve_chapters(raw / "tbook.pdf", ["Extras", "Missing"])] == [
        True,
        False,
    ]
    assert normalise_title("Chapter 3 Git Branching") == normalise_title("3 git  branching")


def test_numbered_and_unnumbered_node_ids(book: tuple[IngestResult, Path, Path]) -> None:
    res, _r, _p = book
    nodes = _nodes(res)
    levels = {nid: n.level for nid, n in nodes.items() if n.level != "code"}
    assert levels == {
        "tb:1": "chapter",
        "tb:1.1": "section",
        "tb:1.1.1": "subsection",
        "tb:h-amortized-cost": "heading",
        "tb:1.2": "section",
        "tb:h-1.2-summary": "heading",  # "Summary" occurs twice: qualified by the parent
        "tb:2": "chapter",
        "tb:2.1": "section",
        "tb:h-2.1-summary": "heading",
    }
    assert nodes["tb:1"].heading == "Basics" and nodes["tb:1"].number == "1"
    assert nodes["tb:1.1.1"].parent_id == "tb:1.1" and nodes["tb:1.2"].parent_id == "tb:1"
    assert nodes["tb:h-amortized-cost"].parent_id == "tb:1.1.1"
    assert nodes["tb:1.1"].is_unit and not nodes["tb:1"].is_unit
    text = res.clean_docs["tb"].text
    for n in nodes.values():
        assert n.text == text[n.char_start : n.char_end]
    # same structure without an outline
    assert set(_nodes(res, "tbp")) == {i.replace("tb:", "tbp:") for i in nodes}


def test_code_kept_inside_its_section_with_indentation(
    book: tuple[IngestResult, Path, Path],
) -> None:
    res, _r, _p = book
    nodes = _nodes(res)
    code = nodes["tb:1.1/code-1"]
    assert code.level == "code" and code.parent_id == "tb:1.1"
    assert code.text == "def grow(a):\n    b = new_array(2 * len(a))\n    return b"
    sec = nodes["tb:1.1"]
    assert sec.char_start < code.char_start and code.char_end <= sec.char_end
    assert "tb:1.1/code-1" in sec.children


def test_headers_math_and_dropped_sections(book: tuple[IngestResult, Path, Path]) -> None:
    res, _r, _p = book
    text = res.clean_docs["tb"].text
    rep = res.report["documents"]["tb"]
    assert rep["printed_page_offset"] == 3
    assert "The Testing Book" not in text and "\nBasics\n" not in text  # running headers
    assert "y = W x + b (1.1)" in text and "O(n^2)" in text  # math kept, superscript marked
    assert "Show that doubling" not in text and "Exercise 2.1" not in text  # exercises
    assert "Example Press" not in text and "References" not in text  # bibliography
    assert "Extras" not in text and "array, 1" not in text  # unselected chapter, index
    assert "Creative Commons" not in text and "Contents" not in text  # front matter
    assert rep["sections_dropped"] == ["1.3 Exercises (p.6)", "References (p.7)"]
    assert text.startswith("1 Basics\n")  # chapter 1 opens on printed page 1


def test_index_and_contents_dropped_without_selection(tmp_path: Path) -> None:
    build_textbook_pdf(tmp_path / "tbook.pdf")
    cfg = IngestConfig(
        documents=[
            DocumentConfig(
                file="tbook.pdf",
                doc_id="all",
                title="Whole book",
                doc_type="textbook",
                parser="generic",
            )
        ]
    )
    res = run_ingest(cfg, tmp_path, tmp_path / "out", write=False)
    dropped = res.report["documents"]["all"]["sections_dropped"]
    assert {"Contents (p.3)", "Index (p.9)", "1.3 Exercises (p.6)"} <= set(dropped)
    assert "array, 1" not in res.clean_docs["all"].text


def test_cross_references(book: tuple[IngestResult, Path, Path]) -> None:
    res, _r, _p = book
    refs = [(x.raw_text, x.target_node, x.resolution) for x in res.xrefs if x.source_doc == "tb"]
    assert ("Section 1.2", "tb:1.2", "exact") in refs
    assert ("Chapter 2", "tb:2", "exact") in refs
    assert ("Section 1.1.7", "tb:1.1", "partial") in refs  # nearest existing ancestor
    assert ("Section 9.4", None, "external") in refs  # chapter 9 was not ingested
    assert ("Fig. 1.1", None, "unresolved") in refs
    assert not any("2.1" in r for r, _t, _s in refs)  # "[3, Section 2.1]" cites another book


def test_licence_and_sources_csv(book: tuple[IngestResult, Path, Path], tmp_path: Path) -> None:
    res, raw, _p = book
    found, _vague = detect_licence(raw / "tbook.pdf")
    assert found is not None and found.licence == "CC BY-SA 4.0 International" and found.page == 2
    lic = res.report["documents"]["tb"]["licence"]
    assert lic["from_pdf"] and lic["source"].startswith("PDF p.2:")
    out = tmp_path / "SOURCES.csv"
    write_textbook_sources_csv(_cfg(), raw, out, res.report)
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == TEXTBOOK_FIELDS
    tb = rows[0]
    assert (
        tb["authors"] == "Ann Author, Bo Writer" and tb["licence"] == "CC BY-SA 4.0 International"
    )
    assert tb["pages"] == "9" and tb["pages_kept"] == "4" and len(tb["sha256"]) == 64
    assert "Basics (pp. 4-6)" in tb["chapters"] and "Nonexistent (not found)" in tb["chapters"]


def test_licence_fallback_is_marked(tmp_path: Path) -> None:
    import pymupdf

    from ragsys.ingest.licence import resolve_licence

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 90), "No licence statement on this page.")
    doc.save(tmp_path / "x.pdf")
    doc.close()
    d = DocumentConfig(
        file="x.pdf",
        doc_id="x",
        title="X",
        doc_type="textbook",
        parser="generic",
        licence="CC BY 2.5 Canada",
        licence_note="from the publisher's site",
    )
    info = resolve_licence(tmp_path / "x.pdf", d)
    assert info.licence == "CC BY 2.5 Canada" and not info.from_pdf
    assert "not stated" in info.source and "publisher" in info.source


def test_structural_chunks_cover_chapter_intro(
    book: tuple[IngestResult, Path, Path], tokenizer: object
) -> None:
    res, _r, _p = book
    cd = res.clean_docs["tb"]
    doc = DocInput(
        doc_id="tb",
        title="The Testing Book",
        doc_type="textbook",
        text=cd.text,
        page_spans=cd.page_spans,
        nodes=[n for n in res.nodes if n.doc_id == "tb"],
    )
    plain = StructuralChunker(tokenizer, 384, 48, True)  # type: ignore[arg-type]
    with_orphans = StructuralChunker(tokenizer, 384, 48, True, orphan_text=True)  # type: ignore[arg-type]
    a, b = plain.chunk_document(doc), with_orphans.chunk_document(doc)
    intro = "This chapter introduces lists"
    assert not any(intro in c.display_text for c in a)
    assert any(intro in c.display_text and c.section_ids == [] for c in b)
    assert all(len(c.section_ids) <= 1 for c in b)  # never across sections
    assert [c.char_start for c in b] == sorted(c.char_start for c in b)
    path = heading_path(doc, doc.by_id["tb:1.1.1"])
    assert path == "Testing Book > Chapter 1 — Basics > 1.1 Lists and Arrays > 1.1.1 Growing Arrays"
