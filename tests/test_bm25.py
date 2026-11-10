"""Legal-aware tokenizer and BM25 ranking."""

from __future__ import annotations

from ragsys.retrieval.bm25 import BM25Index, light_stem, reference_tokens, tokenize


def test_statute_references_become_single_tokens() -> None:
    toks = tokenize("A consumer under section 2(7) may complain.")
    assert {"s2(7)", "2(7)", "s2", "consumer", "section"} <= set(toks)
    assert "under" not in toks and "may" not in toks  # stopwords
    assert {"r4(3)", "4(3)", "r4"} <= set(tokenize("as required by rule 4(3)"))
    # nested chain written the long way maps to the same compact token
    chain = tokenize("clause (zg) of sub-section (2) of section 101")
    assert {"s101(2)(zg)", "101(2)(zg)", "s101(2)", "s101"} <= set(chain)
    assert "s3(1)" in tokenize("sub-section (1) of section 3 of the Act")
    # compact forms typed by a user
    assert {"s35(1)", "35(1)", "s35"} <= set(tokenize("what does s35(1) say"))
    assert "34(1)" in tokenize("limit in 34(1)")
    assert reference_tokens("sections 69 to 72") == ["s69"]


def test_unicode_hyphens_and_years() -> None:
    toks = tokenize("E‑commerce entities and sub-sections; Act, 2019")
    assert "ecommerce" in toks and "subsections" in toks and "2019" in toks
    assert not any(t.startswith("s2019") for t in toks)


def test_light_stemming_is_optional() -> None:
    assert light_stem("complaints") == "complaint"
    assert light_stem("authorities") == "authority"
    assert light_stem("filing") == "fil" or light_stem("filing") == "filing"
    assert light_stem("class") == "class"
    assert "complaints" in tokenize("complaints")
    assert "complaint" in tokenize("complaints", stem=True)


def test_bm25_ranking_on_mini_corpus() -> None:
    docs = {
        "d1": "The District Commission hears complaints about defective goods.",
        "d2": "Section 2(7) defines a consumer who buys goods for consideration.",
        "d3": "Mediation shall be completed within three months.",
        "d4": "The State Commission hears appeals against District Commission orders.",
        "d5": "An e-commerce entity shall display the country of origin.",
        "d6": "Fees are payable for filing a complaint.",
    }
    idx = BM25Index(list(docs), list(docs.values()))
    assert idx.search("Who hears complaints about defective goods?", 3)[0][0] == "d1"
    assert idx.search("what is in s2(7)", 3)[0][0] == "d2"
    assert idx.search("ecommerce country of origin", 1)[0][0] == "d5"
    assert idx.search("zebra unicorn", 5) == []  # no positive score -> nothing
    hits = idx.search("commission", 10)
    assert {h[0] for h in hits} == {"d1", "d4"} and all(s > 0 for _i, s in hits)
    assert idx.vocab_size > 20 and idx.avg_doc_len > 3
