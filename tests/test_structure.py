"""Structure parsing: node IDs, hierarchy, definitions, sanity checks, generic parser."""

from __future__ import annotations

from typing import Any

from ragsys.config import DocumentConfig
from ragsys.ingest.clean import CleanDocument, Paragraph
from ragsys.ingest.pipeline import IngestResult
from ragsys.structure.statute import IndianStatuteParser, next_alpha, numbering_report, roman_to_int

EXPECTED_ACT = [
    ("testact:preamble", "heading", None),
    ("testact:ch1", "chapter", None),
    ("testact:s1", "section", "testact:ch1"),
    ("testact:s1(1)", "subsection", "testact:s1"),
    ("testact:s1(2)", "subsection", "testact:s1"),
    ("testact:s2", "section", "testact:ch1"),
    ("testact:s2(1)", "clause", "testact:s2"),
    ("testact:s2(2)", "clause", "testact:s2"),
    ("testact:s2(3)", "clause", "testact:s2"),
    ("testact:s2(3)(i)", "subclause", "testact:s2(3)"),
    ("testact:s2(3)(ii)", "subclause", "testact:s2(3)"),
    ("testact:s2(4)", "clause", "testact:s2"),
    ("testact:ch2", "chapter", None),
    ("testact:s3", "section", "testact:ch2"),
    ("testact:s3(1)", "subsection", "testact:s3"),
    ("testact:s3(1)(a)", "clause", "testact:s3(1)"),
    ("testact:s3(1)(b)", "clause", "testact:s3(1)"),
    ("testact:s3(1)-prov1", "proviso", "testact:s3(1)"),
    ("testact:s3(1)-expl1", "explanation", "testact:s3(1)"),
    ("testact:s3(2)", "subsection", "testact:s3"),
    ("testact:s4", "section", "testact:ch2"),
    ("testact:s4(1)", "subsection", "testact:s4"),
    ("testact:s4(2)", "subsection", "testact:s4"),
    ("testact:s5", "section", "testact:ch2"),
]


def test_act_node_ids_and_hierarchy_exact(ingested: IngestResult) -> None:
    got = [(n.node_id, n.level, n.parent_id) for n in ingested.nodes if n.doc_id == "testact"]
    assert got == EXPECTED_ACT


def test_children_lists_consistent(ingested: IngestResult, nodes_by_id: dict[str, Any]) -> None:
    for n in ingested.nodes:
        for c in n.children:
            assert nodes_by_id[c].parent_id == n.node_id
            assert (
                n.char_start <= nodes_by_id[c].char_start <= nodes_by_id[c].char_end <= n.char_end
            )


def test_headings_numbers_and_pages(nodes_by_id: dict[str, Any]) -> None:
    s5 = nodes_by_id["testact:s5"]
    assert (
        s5.heading
        == "Powers of the authority to issue guidelines in respect of complaints and appeals"
    )
    assert s5.number == "5"
    assert nodes_by_id["testact:ch1"].heading == "PRELIMINARY"
    # heading printed in the page margin is folded into the section heading
    assert nodes_by_id["testact:s4"].heading == "Appeal"
    assert nodes_by_id["testact:s4"].text.startswith("4. Appeal.—(1) Any person aggrieved")
    assert nodes_by_id["testact:ch2"].number == "II"
    assert (nodes_by_id["testact:s1"].page_start, nodes_by_id["testact:s3"].page_start) == (2, 3)
    assert nodes_by_id["testact:s3(1)-prov1"].text.startswith("Provided that no fee")
    assert nodes_by_id["testact:s3(1)-expl1"].text.startswith("Explanation.—For the purposes")


def test_definitions_at_clause_level(ingested: IngestResult) -> None:
    defs = {n.node_id: n.defined_term for n in ingested.nodes if n.defined_term}
    assert defs == {
        "testact:s2(1)": "consumer",
        "testact:s2(2)": "goods",
        "testact:s2(3)": "trader",
        "testact:s2(4)": "Government",
        "testrules2024:r2(1)(a)": "Act",
        "testrules2024:r2(1)(b)": "entity",
    }
    assert all(n.level == "clause" for n in ingested.nodes if n.defined_term)


def test_rules_ids(ingested: IngestResult) -> None:
    ids = [n.node_id for n in ingested.nodes if n.doc_id == "testrules2024"]
    assert ids == [
        "testrules2024:preamble",
        "testrules2024:r1",
        "testrules2024:r2",
        "testrules2024:r2(1)",
        "testrules2024:r2(1)(a)",
        "testrules2024:r2(1)(b)",
        "testrules2024:r3",
        "testrules2024:r3(1)",
        "testrules2024:r3(2)",
        "testrules2024:r3(3)",
    ]


def test_generic_parser_output(ingested: IngestResult) -> None:
    got = [
        (n.node_id, n.level, n.parent_id, n.heading) for n in ingested.nodes if n.doc_id == "report"
    ]
    assert got == [
        ("report:h1", "heading", None, "Annual Garden Report"),
        ("report:h1.1", "heading", "report:h1", "Introduction"),
        ("report:h1.2", "heading", "report:h1", "Methods"),
        ("report:h1.2.1", "heading", "report:h1.2", "Watering Schedule"),
        ("report:h1.3", "heading", "report:h1", "Results"),
    ]
    h12 = next(n for n in ingested.nodes if n.node_id == "report:h1.2")
    assert "compost" in h12.text and "watered" in h12.text and "tomatoes" not in h12.text


def test_report_sanity(ingested: IngestResult) -> None:
    rep = ingested.report["documents"]["testact"]
    assert rep["sections_or_rules"] == 5
    assert rep["numbering"]["missing"] == [] and rep["numbering"]["count_equals_max"]
    assert rep["definitions"] == 4
    assert rep["pages_dropped_devanagari"] == [4]
    assert "config" in ingested.report and "run_info" in ingested.report


# --------------------------------------------------------------------------- text-level parsing


def _doc(paras: list[tuple[str, float]], doc_id: str = "t") -> tuple[CleanDocument, DocumentConfig]:
    text_parts, paragraphs, off = [], [], 0
    for txt, x0 in paras:
        if text_parts:
            text_parts.append("\n")
            off += 1
        paragraphs.append(Paragraph(txt, off, off + len(txt), 1, 1, x0, 10.0, 0.0, "", 1))
        text_parts.append(txt)
        off += len(txt)
    text = "".join(text_parts)
    cd = CleanDocument(doc_id, text, paragraphs, [], [(0, len(text), 1)], 10.0, 1, {})
    cfg = DocumentConfig(file="x.pdf", doc_id=doc_id, title="T", doc_type="act")
    return cd, cfg


def test_roman_alpha_disambiguation_and_tables() -> None:
    cd, cfg = _doc(
        [
            ("7. Fees.—(1) A fee shall be paid as follows:—", 90),
            ("(a) first;", 108),
            ("(h) eighth;", 108),
            ("(i) ninth item;", 108),  # alphabetic (i) after (h), same indentation
            ("(j) tenth item, namely:—", 108),
            ("(i) first sub-item;", 126),  # roman under (j)
            ("(ii) second sub-item;", 126),
            ("(2) The fee shall be credited.", 90),
            ("(1) (2) (3) Value Fee", 72),  # table header -> plain text of the section
            ("(1) Upto five lakh Nil", 72),
            ("(2) Above five lakh 200", 72),
            ("8. Next.—Text of section 8 under sub-section (2).", 90),
        ]
    )
    res = IndianStatuteParser().parse(cd, cfg)
    ids = [n.node_id for n in res.nodes]
    assert ids == [
        "t:s7",
        "t:s7(1)",
        "t:s7(1)(a)",
        "t:s7(1)(h)",
        "t:s7(1)(i)",
        "t:s7(1)(j)",
        "t:s7(1)(j)(i)",
        "t:s7(1)(j)(ii)",
        "t:s7(2)",
        "t:s8",
    ]
    s7 = next(n for n in res.nodes if n.node_id == "t:s7")
    assert "Above five lakh 200" in s7.text
    assert any("embedded table" in w for w in res.warnings)


def test_duplicate_labels_and_proviso_scope() -> None:
    cd, cfg = _doc(
        [
            ("4. Matters.—The following matters, namely:—", 96),
            ("(a) one;", 96),
            ("(a) again;", 96),  # drafting error in the source
            ("(b) two:", 96),
            (
                "Provided that, in any case other than those mentioned in this section, it applies.",
                144,
            ),
        ]
    )
    res = IndianStatuteParser().parse(cd, cfg)
    ids = [(n.node_id, n.parent_id) for n in res.nodes]
    assert ("t:s4(a)~2", "t:s4") in ids
    assert ("t:s4-prov1", "t:s4") in ids
    assert any("duplicate label" in w for w in res.warnings)


def test_numbering_report_gaps_and_duplicates() -> None:
    cd, cfg = _doc([("1. A.—x", 90), ("2. B.—y", 90), ("4. D.—z", 90), ("4. D again.—w", 90)])
    res = IndianStatuteParser().parse(cd, cfg)
    rep = numbering_report(res.nodes)
    assert rep["missing"] == [3]
    assert rep["max_number"] == 4 and rep["count"] == 4
    assert any("missing [3]" in w for w in res.warnings)


def test_helpers() -> None:
    assert roman_to_int("xiv") == 14 and roman_to_int("abc") is None
    assert next_alpha("h") == "i" and next_alpha("z") == "za" and next_alpha("zf") == "zg"
