"""Chunker invariants on the synthetic corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragsys.chunking.base import Chunk, DocInput
from ragsys.chunking.fixed import FixedChunker, token_windows
from ragsys.chunking.pipeline import chunk_stats, load_corpus
from ragsys.chunking.semantic import Embedder, EmbeddingCache, SemanticChunker, split_units
from ragsys.chunking.structural import StructuralChunker
from ragsys.ingest.pipeline import IngestResult


@pytest.fixture(scope="module")
def corpus(ingested: IngestResult, processed_dir: Path) -> list[DocInput]:
    return load_corpus(processed_dir)


def _run(chunker: Any, corpus: list[DocInput]) -> list[Chunk]:
    out: list[Chunk] = []
    for d in corpus:
        out.extend(chunker.chunk_document(d))
    return out


def _assert_provenance(chunks: list[Chunk], corpus: list[DocInput]) -> None:
    docs = {d.doc_id: d for d in corpus}
    for c in chunks:
        d = docs[c.doc_id]
        assert d.text[c.char_start : c.char_end] == c.display_text
        assert (c.page_start, c.page_end) == d.pages_for(c.char_start, c.char_end)
        assert c.primary_node_id is None or c.primary_node_id in d.by_id


def _assert_full_coverage(chunks: list[Chunk], corpus: list[DocInput]) -> None:
    for d in corpus:
        covered = [False] * len(d.text)
        for c in chunks:
            if c.doc_id == d.doc_id:
                for i in range(c.char_start, c.char_end):
                    covered[i] = True
        gaps = [ch for i, ch in enumerate(d.text) if not covered[i] and not ch.isspace()]
        assert gaps == [], f"{d.doc_id}: uncovered characters {''.join(gaps)[:50]!r}"


def test_token_windows_snap_to_words() -> None:
    text = "alpha beta-gamma delta epsilon"
    offs = [(0, 5), (6, 10), (10, 11), (11, 16), (17, 22), (23, 30)]
    wins = token_windows(text, offs, size=3, overlap=1)
    assert wins[0] == (0, 1)  # "beta-gamma" (3 tokens) does not fit after "alpha"
    assert all(e - s <= 3 for s, e in wins)
    assert wins[-1][1] == len(offs)


def test_fixed_chunker(corpus: list[DocInput], tokenizer: Any) -> None:
    chunker = FixedChunker("fixed_64", 64, 8, tokenizer)
    a, b = _run(chunker, corpus), _run(chunker, corpus)
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]  # deterministic
    assert len([c for c in a if c.doc_id == "testact"]) > 3
    for c in a:
        assert c.n_tokens <= 64 and tokenizer.count(c.display_text) == c.n_tokens
        assert c.text == c.display_text
        # snapped to word boundaries: never starts/ends inside a word
        d = next(x for x in corpus if x.doc_id == c.doc_id)
        assert c.char_start == 0 or d.text[c.char_start - 1].isspace()
        assert c.char_end == len(d.text) or not d.text[c.char_end].isalnum()
    _assert_full_coverage(a, corpus)
    _assert_provenance(a, corpus)
    s3 = [c for c in a if "testact:s3" in c.section_ids]
    assert s3 and all(c.heading_path.startswith("Test Protection Act, 2024") for c in s3)


def test_fixed_chunk_xrefs_and_sections(corpus: list[DocInput], tokenizer: Any) -> None:
    chunks = _run(FixedChunker("fixed_512", 512, 64, tokenizer), corpus)
    act = [c for c in chunks if c.doc_id == "testact"]
    assert len(act) == 1  # the whole synthetic Act fits into one 512-token window
    assert {"testact:s1", "testact:s5"} <= set(act[0].section_ids)
    assert {"testact:s2(1)", "testact:s3(1)", "testact:s3(1)(a)"} <= set(act[0].xref_targets)
    stats = chunk_stats(chunks, corpus)
    assert stats["chunks_crossing_section_boundaries"] >= 1


def test_split_units_is_legal_aware() -> None:
    text = (
        "2. Definitions.—In this Act, viz. the s. 3 rule, No. 5 applies. It ends here. (a) item; D. L. Office.\n"
        "(1) Filed under clause (a). The fee is paid."
    )
    units = [text[a:b] for a, b in split_units(text)]
    assert units == [
        "2. Definitions.—In this Act, viz. the s. 3 rule, No. 5 applies.",
        "It ends here. (a) item; D. L. Office.",  # no break before an inline "(a)"
        "(1) Filed under clause (a).",
        "The fee is paid.",
    ]


def test_semantic_chunker(corpus: list[DocInput], tokenizer: Any, tmp_path: Path) -> None:
    emb = Embedder("BAAI/bge-small-en-v1.5", "cpu", 16, cache_dir=tmp_path)
    chunker = SemanticChunker(
        tokenizer, emb, breakpoint_percentile=70, min_tokens=24, max_tokens=80
    )
    first = _run(chunker, corpus)
    assert emb.stats["computed"] > 0
    # deterministic, and the second pass is served from the on-disk cache
    emb2 = Embedder("BAAI/bge-small-en-v1.5", "cpu", 16, cache_dir=tmp_path)
    second = _run(SemanticChunker(tokenizer, emb2, 70, 24, 80), corpus)
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]
    assert emb2.stats["computed"] == 0 and emb2.stats["cached"] > 0
    assert len(EmbeddingCache(tmp_path / "embeddings_BAAI_bge_small_en_v1_5.npz")) > 0
    for c in first:
        assert c.n_tokens <= 80
    act = [c for c in first if c.doc_id == "testact"]
    assert len(act) >= 3
    assert all(c.n_tokens >= 24 for c in act[:-1])  # min enforced (last chunk may be short)
    _assert_full_coverage(first, corpus)
    _assert_provenance(first, corpus)


def test_semantic_breakpoints_follow_percentile(tokenizer: Any) -> None:
    def fake_embed(texts: list[str]) -> np.ndarray:
        # topic switch between unit 2 and 3 -> one big distance
        vecs = [np.array([1.0, 0.0]) if i < 3 else np.array([0.0, 1.0]) for i in range(len(texts))]
        return np.stack(vecs).astype(np.float32)

    text = "Apples are red fruit grown here. Apples are sweet too. Apples keep well. Trains run fast. Trains carry freight. Trains are long."
    doc = DocInput("x", "X Report", "generic", text, [(0, len(text), 1)], [])
    chunks = SemanticChunker(tokenizer, fake_embed, 50.0, 1, 512).chunk_document(doc)
    assert [c.display_text for c in chunks] == [
        "Apples are red fruit grown here. Apples are sweet too. Apples keep well.",
        "Trains run fast. Trains carry freight. Trains are long.",
    ]


def test_structural_chunker(corpus: list[DocInput], tokenizer: Any) -> None:
    chunker = StructuralChunker(tokenizer, max_tokens=60, min_tokens=12)
    chunks = _run(chunker, corpus)
    assert [c.model_dump() for c in chunks] == [c.model_dump() for c in _run(chunker, corpus)]
    _assert_provenance(chunks, corpus)
    for c in chunks:
        assert len(c.section_ids) == 1, c.chunk_id  # never crosses a section/rule boundary
        assert tokenizer.count(c.display_text) <= 60
        assert c.text == f"{c.heading_path}\n{c.display_text}"  # contextual header
        assert c.heading_path not in c.display_text
    by_primary = {c.primary_node_id: c for c in chunks}
    # every definition clause is its own chunk; the section intro rides on the first one
    for d in ("testact:s2(1)", "testact:s2(2)", "testact:s2(3)", "testact:s2(4)"):
        assert d in by_primary
    assert by_primary["testact:s2(1)"].display_text.startswith("2. Definitions.—In this Act")
    assert by_primary["testact:s2(1)"].heading_path == (
        'Test Protection Act, 2024 > Chapter I > Section 2 — Definitions > Clause (1) — "consumer"'
    )
    # short sections stay whole
    assert by_primary["testact:s1"].display_text.startswith("1. Short title")
    assert by_primary["testact:s1"].heading_path == (
        "Test Protection Act, 2024 > Chapter I > Section 1 — Short title and commencement"
    )
    # xref targets come from the covered text
    s4 = [c for c in chunks if "testact:s4" in c.section_ids]
    assert any("testact:s3(1)" in c.xref_targets for c in s4)
    # every unit is covered by at least one chunk
    units = {n.node_id for d in corpus for n in d.nodes if n.is_unit}
    assert units == {c.section_ids[0] for c in chunks}


def test_structural_default_limits_keep_small_sections_whole(
    corpus: list[DocInput], tokenizer: Any
) -> None:
    chunks = _run(StructuralChunker(tokenizer), corpus)
    s3 = [c for c in chunks if c.section_ids == ["testact:s3"]]
    assert len(s3) == 1 and s3[0].primary_node_id == "testact:s3"
    assert chunk_stats(chunks, corpus)["chunks_crossing_section_boundaries"] == 0
