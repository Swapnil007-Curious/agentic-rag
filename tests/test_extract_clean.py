"""Extraction and cleaning on synthetic PDFs + normalisation unit tests."""

from __future__ import annotations

import re
from pathlib import Path

from ragsys.config import CleaningConfig, DocumentConfig
from ragsys.ingest.clean import (
    devanagari_fraction,
    normalise_text,
    strip_devanagari,
    transliterate_hindi_labels,
)
from ragsys.ingest.extract import extract_pdf
from ragsys.ingest.pipeline import IngestResult


def test_extract_keeps_layout(pdf_dir: Path) -> None:
    raw = extract_pdf(pdf_dir / "testact.pdf")
    assert raw.page_count == 4
    assert raw.has_text_layer
    line = next(
        ln
        for p in raw.pages
        for ln in p.lines
        if ln.text.startswith("1. Short title and commencement.--")
    )
    assert any(s.bold for s in line.spans) and not all(s.bold for s in line.spans)
    assert line.size == 10.0
    # the superscript footnote marker after "date" is removed from the body line
    marked = next(
        ln for p in raw.pages for ln in p.lines if "come into force on such date" in ln.text
    )
    assert marked.markers == ["1"]
    assert "date as the Government" in marked.text


def test_headers_footers_and_page_numbers_removed(ingested: IngestResult) -> None:
    text = ingested.clean_docs["testact"].text
    assert "THE TEST PROTECTION ACT, 2024" not in text
    assert not re.search(r"^\d+$", text, flags=re.MULTILINE)  # standalone page numbers
    stats = ingested.clean_docs["testact"].stats
    assert stats["header_footer_lines_removed"] >= 6  # 3 headers + 3 page numbers on kept pages


def test_toc_dropped_without_duplicate_headings(ingested: IngestResult) -> None:
    cd = ingested.clean_docs["testact"]
    assert "ARRANGEMENT OF SECTIONS" not in cd.text
    assert cd.stats["toc_pages"] == [1]
    for heading in (
        "1. Short title and commencement.",
        "2. Definitions.",
        "3. Filing of complaint.",
        "4. Appeal.",
    ):
        assert cd.text.count(heading) == 1, heading
    assert cd.text.count("CHAPTER I\n") == 1
    sections = [n.node_id for n in ingested.nodes if n.doc_id == "testact" and n.level == "section"]
    assert sections == [f"testact:s{i}" for i in range(1, 6)]


def test_devanagari_removed(ingested: IngestResult) -> None:
    cd = ingested.clean_docs["testact"]
    assert cd.stats["pages_dropped_devanagari"] == [4]
    assert not re.search(r"[ऀ-ॿ]", cd.text)
    # mixed line (mostly English) is kept without its Devanagari word
    assert "The authority shall publish an annual report every year in English" in cd.text
    # Devanagari-dominant line is dropped entirely (its only English word was "report")
    assert cd.stats["devanagari_lines_removed"] >= 1
    assert cd.stats["mixed_lines_cleaned"] == 1
    assert not cd.text.rstrip().endswith("report")


def test_dehyphenation_and_line_joining(ingested: IngestResult) -> None:
    text = ingested.clean_docs["testact"].text
    assert "includes a registered user;" in text
    assert "regis-" not in text
    # wrapped heading + body joined into one paragraph
    assert (
        "5. Powers of the authority to issue guidelines in respect of complaints and appeals.—The authority"
        in text
    )


def test_footnotes_separated(ingested: IngestResult) -> None:
    cd = ingested.clean_docs["testact"]
    assert "Notification No. S.O. 1(E)" not in cd.text
    assert len(cd.footnotes) == 1
    fn = cd.footnotes[0]
    assert fn.page == 2 and fn.marker == "1"
    assert fn.text.startswith("1. 1st January, 2025")
    assert "testact:s1" in fn.section_ids


def test_start_and_end_markers(ingested: IngestResult) -> None:
    cd = ingested.clean_docs["testrules2024"]
    assert cd.text.startswith("MINISTRY OF TESTING")
    assert "PUBLISHED BY AUTHORITY" not in cd.text
    assert "F. No." not in cd.text and "OFFICER" not in cd.text


def test_provenance_offsets_and_pages(ingested: IngestResult, processed_dir: Path) -> None:
    cd = ingested.clean_docs["testact"]
    on_disk = (processed_dir / "clean" / "testact.txt").read_text(encoding="utf-8")
    assert on_disk == cd.text
    for p in cd.paragraphs:
        assert cd.text[p.char_start : p.char_end] == p.text
    assert cd.page_at(cd.text.index("An Act to provide")) == 2
    assert cd.page_at(cd.text.index("3. Filing of complaint")) == 3
    for n in ingested.nodes:
        doc = ingested.clean_docs[n.doc_id]
        assert doc.text[n.char_start : n.char_end] == n.text


def test_normalise_text() -> None:
    assert normalise_text("“consumer” ‘x’") == "\"consumer\" 'x'"
    assert normalise_text("Definitions. – (1) In") == "Definitions.—(1) In"
    assert normalise_text("Cell.---(1) Every") == "Cell.—(1) Every"
    assert normalise_text("namely:- (a)") == "namely:—(a)"
    assert normalise_text("―PAN” means") == '"PAN" means'  # Gazette quote artefacts
    assert normalise_text("entity‖ means") == 'entity" means'
    assert normalise_text("Kashmir*. It") == "Kashmir. It"
    assert normalise_text("a  b c­x") == "a b cx"
    assert (
        normalise_text("(1) (a) (iv) Explanation.— Provided that")
        == "(1) (a) (iv) Explanation.—Provided that"
    )


def test_devanagari_helpers() -> None:
    assert devanagari_fraction("abc") == 0.0
    assert devanagari_fraction("भारत ab") > 0.3
    assert transliterate_hindi_labels("(क) legal name; (ख) address") == (
        "(a) legal name; (b) address",
        2,
    )
    assert strip_devanagari("report भारत now") == ("report now", 4)


def test_scanned_pdf_has_no_text_layer(tmp_path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 200, 200), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    doc.save(tmp_path / "scan.pdf")
    raw = extract_pdf(tmp_path / "scan.pdf")
    assert not raw.has_text_layer


def test_cleaning_config_forbids_unknown_keys() -> None:
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        CleaningConfig(not_a_key=1)  # type: ignore[call-arg]
    with pytest.raises(pydantic.ValidationError):
        DocumentConfig(file="a.pdf", doc_id="A", title="t", doc_type="act")  # bad doc_id pattern


def test_marginal_heading_becomes_inline(ingested: IngestResult) -> None:
    cd = ingested.clean_docs["testact"]
    assert cd.stats["marginal_headings_attached"] == 1
    assert "4. Appeal.—(1) Any person aggrieved" in cd.text
    assert "\nAppeal.\n" not in cd.text


def test_page_number_pattern_keeps_years() -> None:
    from ragsys.ingest.clean import _PAGE_NUMBER_RE

    for t in ("7", "(1)", "12", "Page 3 of 9"):
        assert _PAGE_NUMBER_RE.match(t), t
    for t in ("2020.", "2020", "7.", "(2) They shall"):
        assert not _PAGE_NUMBER_RE.match(t), t
