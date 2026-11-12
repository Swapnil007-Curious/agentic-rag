"""ConstitutionParser on a synthetic Constitution-like PDF (Parts, a Chapter inside a Part,
Articles 21 and 21A with clauses, a proviso, an Explanation, a group heading, the Seventh
Schedule, an appendix) and the opt-in statute layouts used by the Indian law collection
(section headings printed above the numbered line; private-use glyphs)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from pdf_fixtures import BODY, BOLD, REG, _Page
from ragsys.chunking.base import heading_path, node_label
from ragsys.config import DocumentConfig, IngestConfig
from ragsys.ingest.pipeline import run_ingest


def build_constitution_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    p = _Page(doc, 1, None)
    p.line(240, [("CONTENTS", REG)])
    p.line(90, [("PREAMBLE", REG)])
    p.line(90, [("21. Protection of life and personal liberty.", REG)])
    p.page = doc.new_page(width=595, height=842)
    p.y = 90.0
    p.line(260, [("PREAMBLE", BOLD)])
    p.line(
        90,
        [("WE, THE PEOPLE OF TESTLAND, having solemnly resolved to constitute a Republic.", REG)],
    )
    p.para_gap()
    p.line(270, [("PART III", BOLD)])
    p.line(230, [("FUNDAMENTAL RIGHTS", REG)])
    p.para_gap()
    p.line(250, [("Right to Freedom", "tiit")])
    p.para_gap()
    p.line(
        90, [("21. Protection of life and personal liberty.", BOLD), ("--No person shall be", REG)]
    )
    p.line(72, [("deprived of his life except according to procedure established by law.", REG)])
    p.para_gap()
    p.line(
        90,
        [
            ("21A. Right to education.", BOLD),
            ("--(1) The State shall provide free education.", REG),
        ],
    )
    p.para_gap()
    p.line(90, [("(2) The State may by law determine the manner--", REG)])
    p.para_gap()
    p.line(110, [("(a) for children of six years; and", REG)])
    p.para_gap()
    p.line(110, [("(b) for children of fourteen years:", REG)])
    p.para_gap()
    p.line(90, [("Provided that no fee shall be charged.", REG)])
    p.para_gap()
    p.line(
        90,
        [
            ("Explanation.", "tiit"),
            ("--In this article, child means a person below fourteen.", REG),
        ],
    )
    p.para_gap()
    p.line(270, [("PART V", BOLD)])
    p.line(250, [("THE UNION", REG)])
    p.line(220, [("CHAPTER I.--THE EXECUTIVE", REG)])
    p.para_gap()
    p.line(90, [("52. The President.", BOLD), ("--There shall be a President of Testland.", REG)])
    p.para_gap()
    # an omitted article: regular font, the amendment bracket before the number
    p.line(90, [("[60. Oath by the President.].--Omitted by the Test Amendment Act.", REG)])
    p.para_gap()
    p.line(
        90,
        [
            ("243ZH. Definitions.", BOLD),
            ("--In this Part, board means the board of a society.", REG),
        ],
    )
    p.para_gap()
    p.line(240, [("SEVENTH SCHEDULE", BOLD)])
    p.line(90, [("List I--Union List", REG)])
    p.line(90, [("1. Defence of Testland and every part thereof.", REG)])
    p.para_gap()
    p.line(260, [("APPENDIX I", BOLD)])
    p.line(90, [("An amendment Act reproduced for reference.", REG)])
    doc.save(path)
    doc.close()
    return path


def build_heading_above_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    p = _Page(doc, 1, None)
    p.line(90, [("CHAPTER I", BOLD)])
    p.line(90, [("PRELIMINARY", BOLD)])
    p.para_gap()
    p.line(72, [("Short title and commencement.", BOLD)])
    p.line(72, [("1. (1) This Act may be called the Testing Tax Act, 2025.", BOLD)])
    p.line(72, [("(2) It extends to the whole of Testland.", REG)])
    p.para_gap()
    p.line(72, [("Definitions.", BOLD)])
    p.line(72, [("2. In this Act, unless the context otherwise requires,--", BOLD)])
    p.line(86, [('(1) "assessee" means a person by whom tax is payable;', REG)])
    p.para_gap()
    p.line(72, [("Charge of tax.", BOLD)])
    p.line(72, [("3. Tax shall be charged at the rates in force for the year.", BOLD)])
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="module")
def coi(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    raw = tmp_path_factory.mktemp("coi_raw")
    build_constitution_pdf(raw / "coi.pdf")
    cfg = IngestConfig(
        documents=[
            DocumentConfig(
                file="coi.pdf",
                doc_id="coi",
                title="The Constitution of Testland",
                doc_type="constitution",
                parser="constitution",
                start_marker=r"^WE, THE PEOPLE OF",
                back_matter_markers=[r"^APPENDIX\s+[IVX]+$"],
                strip_private_use=True,
            )
        ]
    )
    res = run_ingest(cfg, raw, tmp_path_factory.mktemp("coi_out"), write=False)
    return res


def test_constitution_ids(coi) -> None:  # type: ignore[no-untyped-def]
    ids = {n.node_id: n for n in coi.nodes}
    for nid in (
        "coi:preamble",
        "coi:part3",
        "coi:a21",
        "coi:a21A",
        "coi:a21A(1)",
        "coi:a21A(2)",
        "coi:a21A(2)(a)",
        "coi:a21A(2)(b)",
        "coi:part5",
        "coi:part5-ch1",
        "coi:a52",
        "coi:a60",
        "coi:a243ZH",
        "coi:sch7",
    ):
        assert nid in ids, (nid, sorted(ids))
    assert "Omitted" in ids["coi:a60"].text and ids["coi:a60"].parent_id == "coi:part5-ch1"
    assert ids["coi:a60"].heading == "Oath by the President"  # editorial brackets removed
    assert "Oath" not in ids["coi:a52"].text  # the omitted article is not glued to article 52
    assert ids["coi:a21"].parent_id == "coi:part3" and ids["coi:a21"].level == "section"
    assert ids["coi:a21"].heading == "Protection of life and personal liberty"
    assert ids["coi:a21A"].number == "21A" and ids["coi:a21A(1)"].parent_id == "coi:a21A"
    assert ids["coi:part3"].heading == "FUNDAMENTAL RIGHTS" and ids["coi:part3"].number == "III"
    assert (
        ids["coi:part5-ch1"].parent_id == "coi:part5"
        and ids["coi:a52"].parent_id == "coi:part5-ch1"
    )
    assert ids["coi:sch7"].level == "schedule" and "Union List" in ids["coi:sch7"].text
    assert "Defence" not in ids["coi:a243ZH"].text  # schedule entries are not articles
    provisos = [n for n in coi.nodes if n.level == "proviso"]
    expls = [n for n in coi.nodes if n.level == "explanation"]
    assert provisos and provisos[0].node_id.startswith("coi:a21A")
    assert expls and expls[0].node_id.startswith("coi:a21A")
    # the group heading ends article 21 instead of being glued to it
    assert "Right to Freedom" not in ids["coi:a21"].text
    # the appendix is back matter, not part of the Seventh Schedule
    assert "amendment Act reproduced" not in ids["coi:sch7"].text
    assert any(n.node_id.startswith("coi:h-appendix") for n in coi.nodes)


def test_constitution_report_and_labels(coi) -> None:  # type: ignore[no-untyped-def]
    rep = coi.report["documents"]["coi"]
    num = rep["numbering"]
    assert num["unit"] == "article" and num["count"] == 5 and num["max_number"] == 243
    assert rep["parser_stats"]["parts"] == 2 and rep["parser_stats"]["chapters_in_parts"] == 1
    assert rep["parser_stats"]["lettered_articles"] == 2  # 21A, 243ZH
    ids = {n.node_id: n for n in coi.nodes}
    assert node_label(ids["coi:a21A"], "constitution") == "Article 21A — Right to education"
    assert node_label(ids["coi:part3"], "constitution").startswith("Part III")
    assert node_label(ids["coi:part5-ch1"], "constitution").startswith("Chapter I")


def test_heading_above_layout(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    build_heading_above_pdf(raw / "tax.pdf")

    def parse(above: bool) -> dict[str, object]:
        cfg = IngestConfig(
            documents=[
                DocumentConfig(
                    file="tax.pdf",
                    doc_id="tax",
                    title="The Testing Tax Act, 2025",
                    doc_type="act",
                    parser="statute",
                    section_heading_above=above,
                )
            ]
        )
        res = run_ingest(cfg, raw, tmp_path / f"out{above}", write=False)
        return {n.node_id: n for n in res.nodes}

    off = parse(False)
    on = parse(True)
    assert "tax:s1" not in off  # the default layout does not see these sections
    for nid, head in (
        ("tax:s1", "Short title and commencement"),
        ("tax:s2", "Definitions"),
        ("tax:s3", "Charge of tax"),
    ):
        assert nid in on and on[nid].heading == head  # type: ignore[attr-defined]
    assert "tax:s1(1)" in on and "tax:s2(1)" in on
    assert on["tax:s2(1)"].defined_term == "assessee"  # type: ignore[attr-defined]


def test_textbook_options_are_off_by_default() -> None:
    from ragsys.config import TextbookConfig

    t = TextbookConfig()
    assert t.drop_to_chapter_end == [] and t.drop_outline_headings is False
    d = DocumentConfig(file="x.pdf", doc_id="x", title="X", doc_type="act")
    assert not d.section_heading_above and not d.strip_private_use and d.outline_chapters is None
    assert BODY > 0 and heading_path is not None


def build_body_font_footnotes_pdf(path: Path) -> Path:
    """Income-tax Act print: amendment footnotes in the body font at the foot of the page, and
    a heading carrying an amendment marker ("18[Deduction for testing.]") on the next page."""
    doc = pymupdf.open()
    p = _Page(doc, 1, None)
    p.line(72, [("Charge of tax.", BOLD)])
    p.line(72, [("3. Tax shall be charged at the rates in force for the tax year.", BOLD)])
    p.line(72, [("(1) The rates are those of the Finance Act of the year.", REG)])
    p.y = 700.0
    p.line(72, [("7. Sub. by Act No. 4 of 2026, w.e.f. 1-4-2026. Prior to its substitution,", REG)])
    p.line(86, [("clause (a) read as under: the old rate applied.", REG)])
    p.page = doc.new_page(width=595, height=842)
    p.y = 90.0
    p.line(72, [("18[Deduction for testing.]", BOLD)])
    p.line(72, [("4. (1) A deduction is allowed for testing expenses.", REG)])
    p.line(72, [("(2) The deduction shall not exceed the expenses.", REG)])
    doc.save(path)
    doc.close()
    return path


def test_body_font_footnotes_and_marked_headings(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    build_body_font_footnotes_pdf(raw / "ita.pdf")

    def parse(pattern: str | None) -> object:
        cfg = IngestConfig(
            documents=[
                DocumentConfig(
                    file="ita.pdf",
                    doc_id="ita",
                    title="The Testing Income-tax Act, 2025",
                    doc_type="act",
                    parser="statute",
                    section_heading_above=True,
                    footnote_start_pattern=pattern,
                )
            ]
        )
        return run_ingest(cfg, raw, tmp_path / f"out{pattern is None}", write=False)

    res = parse(r"^\d{1,3}\.\s+(?:Subs?|Ins)\b")
    ids = {n.node_id: n for n in res.nodes}  # type: ignore[attr-defined]
    assert ids["ita:s3"].heading == "Charge of tax"
    assert ids["ita:s4"].heading == "Deduction for testing"  # marker and brackets removed
    assert not any("Sub. by Act" in n.text for n in res.nodes)  # type: ignore[attr-defined]
    notes = [f.text for f in res.footnotes]  # type: ignore[attr-defined]
    assert any(t.startswith("7. Sub. by Act No. 4 of 2026") and "old rate" in t for t in notes)
    # without the option the footnote stays in the body text (the default layout is unchanged)
    off = parse(None)
    assert any("Sub. by Act" in n.text for n in off.nodes)  # type: ignore[attr-defined]
