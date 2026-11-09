"""Cross-reference extraction and resolution."""

from __future__ import annotations

from ragsys.ingest.pipeline import IngestResult
from ragsys.structure.base import Node
from ragsys.structure.xrefs import XRefResolver, extract_xrefs


def _edges(ingested: IngestResult, source: str) -> list[tuple[str, str | None, str]]:
    return [
        (x.raw_text, x.target_node, x.resolution) for x in ingested.xrefs if x.source_node == source
    ]


def test_section_and_subsection_refs(ingested: IngestResult) -> None:
    assert ("clause (1) of section 2", "testact:s2(1)", "exact") in _edges(
        ingested, "testact:s3(1)"
    )
    assert ("sub-section (1) of section 3", "testact:s3(1)", "exact") in _edges(
        ingested, "testact:s4(2)"
    )
    # relative reference resolved inside the same section
    assert ("clause (a) of sub-section (1)", "testact:s3(1)(a)", "exact") in _edges(
        ingested, "testact:s3(2)"
    )


def test_range_expansion(ingested: IngestResult) -> None:
    targets = [t for raw, t, _ in _edges(ingested, "testact:s4(1)") if raw == "sections 1 to 3"]
    assert targets == ["testact:s1", "testact:s2", "testact:s3"]


def test_rules_to_act(ingested: IngestResult) -> None:
    e = _edges(ingested, "testrules2024:r3(2)")
    assert ("sub-section (1) of section 3 of the Act", "testact:s3(1)", "exact") in e
    rng = [
        t
        for raw, t, _ in _edges(ingested, "testrules2024:r3(3)")
        if raw.startswith("sections 3 to 5")
    ]
    assert rng == ["testact:s3", "testact:s4", "testact:s5"]
    # named Act resolved through act_aliases
    assert ("section 5 of the Test Protection Act, 2024", "testact:s5", "exact") in _edges(
        ingested, "testrules2024:preamble"
    )
    # sub-rule and rule references stay inside the Rules
    assert ("sub-rule (1)", "testrules2024:r3(1)", "exact") in _edges(
        ingested, "testrules2024:r3(3)"
    )
    assert ("rule 3", "testrules2024:r3", "exact") in _edges(ingested, "testrules2024:r2(1)(b)")


def test_unresolved_kept_with_raw_text(ingested: IngestResult) -> None:
    x = next(x for x in ingested.xrefs if x.raw_text == "section 101 of the Act")
    assert x.source_node == "testrules2024:r3(1)"
    assert x.target_node is None and x.target_doc == "testact" and x.resolution == "unresolved"


def test_document_refs(ingested: IngestResult) -> None:
    docs = {(x.raw_text, x.target_doc) for x in ingested.xrefs if x.kind == "document"}
    assert ("this Act", "testact") in docs
    assert ("these rules", "testrules2024") in docs


def _node(
    node_id: str, start: int, end: int, parent: str | None = None, level: str = "section"
) -> Node:
    return Node(
        doc_id=node_id.split(":")[0],
        doc_title="T",
        doc_type="act",
        node_id=node_id,
        level=level,  # type: ignore[arg-type]
        number=None,
        text="",
        parent_id=parent,
        page_start=1,
        page_end=1,
        char_start=start,
        char_end=end,
    )


def test_text_level_patterns() -> None:
    text = (
        "under sub-clauses (i) and (ii) of clause (a) of sub-section (1) of section 47; "
        "clauses (f) to (h) of section 2; section 21 of the Indian Penal Code (45 of 1860); "
        "second proviso to section 67."
    )
    nodes = [
        _node("a:s1", 0, len(text)),
        _node("a:s47", len(text), len(text) + 1),
        _node("a:s47(1)", len(text), len(text) + 1, "a:s47", "subsection"),
        _node("a:s47(1)(a)", len(text), len(text) + 1, "a:s47(1)", "clause"),
        _node("a:s47(1)(a)(i)", len(text), len(text) + 1, "a:s47(1)(a)", "subclause"),
        _node("a:s47(1)(a)(ii)", len(text), len(text) + 1, "a:s47(1)(a)", "subclause"),
        _node("a:s2", len(text), len(text) + 1),
        _node("a:s2(g)", len(text), len(text) + 1, "a:s2", "clause"),
        _node("a:s21", len(text), len(text) + 1),
        _node("a:s67", len(text), len(text) + 1),
        _node("a:s67-prov2", len(text), len(text) + 1, "a:s67", "proviso"),
    ]
    resolver = XRefResolver(nodes, {"a": "act"}, {"a": None}, {})
    xs = extract_xrefs("a", text, nodes, resolver)
    got = [(x.target_node, x.resolution) for x in xs]
    assert ("a:s47(1)(a)(i)", "exact") in got and ("a:s47(1)(a)(ii)", "exact") in got
    assert ("a:s2(g)", "exact") in got  # (f) and (h) do not exist ...
    assert ("a:s2", "partial") in got  # ... and fall back to the section
    ipc = next(x for x in xs if "Indian Penal Code" in x.raw_text)
    assert ipc.resolution == "external" and ipc.target_node is None  # not a:s21
    assert ("a:s67-prov2", "exact") in got
