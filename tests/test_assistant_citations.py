"""Citation labels of chat replies: parsing, normalising, stripping invalid labels, and the
source line."""

from __future__ import annotations

from ragsys.assistant.citations import (
    GENERAL_LINE,
    clean_citations,
    find_labels,
    source_line,
)


def test_find_labels_in_prose_only() -> None:
    text = (
        "A [C2]. B [C1][C2].\n\n```python\nx = a[C3]\n```\nInline `b[C4]` and 【C5】 and [C6, C7]."
    )
    assert find_labels(text) == ["C2", "C1", "C5", "C6", "C7"]


def test_invalid_labels_are_stripped_and_valid_kept() -> None:
    r = clean_citations("Linear time [C1]. Wrong [C9]. Both [C1][C2].", ["C1", "C2", "C3"])
    assert r.text == "Linear time [C1]. Wrong. Both [C1][C2]."
    assert r.used == ["C1", "C2"] and r.removed == ["C9"] and r.grounded
    assert r.source_line() == "Answered with documents (2 sources)"


def test_variants_are_normalised() -> None:
    r = clean_citations("See 【C2】 and [C1, C3] and 【C4†L1-L3】.", ["C1", "C2", "C3"])
    assert r.text == "See [C2] and [C1][C3] and." and r.used == ["C2", "C1", "C3"]
    assert r.removed == ["C4"]


def test_no_passages_removes_every_label() -> None:
    r = clean_citations("A fact [C1]. Another [C2].", [])
    assert r.text == "A fact. Another." and r.used == [] and not r.grounded
    assert r.source_line() == GENERAL_LINE


def test_code_is_never_changed() -> None:
    text = (
        "Use it [C1].\n\n```python\nrows = table[C2]\nprint([C1, C2])\n```\n\nAnd `m[C5]` too [C7]."
    )
    r = clean_citations(text, ["C1"])
    assert "rows = table[C2]" in r.text and "print([C1, C2])" in r.text and "`m[C5]`" in r.text
    assert r.text.endswith("too.") and r.removed == ["C7"]


def test_source_line_texts() -> None:
    assert source_line(0) == "General knowledge — no matching documents"
    assert source_line(1) == "Answered with documents (1 source)"
    assert source_line(3) == "Answered with documents (3 sources)"
