"""Structure-aware cross-reference expansion."""

from __future__ import annotations

from pathlib import Path

from ragsys.chunking.base import Chunk
from ragsys.ingest.pipeline import load_nodes, load_xrefs
from ragsys.retrieval.corpus import ChunkCorpus, NodeGraph, load_chunks
from ragsys.retrieval.xref_expand import XrefExpander
from ragsys.structure.base import Node

# doc A (act): ch1 > s1, s2 > s2(1), s2(2), s3 > s3(1); doc R (rules): r1
A_SPANS = {
    "A:ch1": (0, 400, None, "chapter"),
    "A:s1": (10, 100, "A:ch1", "section"),
    "A:s2": (100, 250, "A:ch1", "section"),
    "A:s2(1)": (120, 180, "A:s2", "subsection"),
    "A:s2(2)": (181, 250, "A:s2", "subsection"),
    "A:s3": (250, 400, "A:ch1", "section"),
    "A:s3(1)": (270, 400, "A:s3", "subsection"),
}


def _nodes() -> list[Node]:
    out = []
    text = "x" * 400
    for i, (nid, (a, b, parent, level)) in enumerate(A_SPANS.items()):
        out.append(
            Node(
                doc_id="A",
                doc_title="Act",
                doc_type="act",
                node_id=nid,
                level=level,  # type: ignore[arg-type]
                text=text[a:b],
                parent_id=parent,
                page_start=1,
                page_end=1,
                char_start=a,
                char_end=b,
                order=i,
            )
        )
    out.append(
        Node(
            doc_id="R",
            doc_title="Rules",
            doc_type="rules",
            node_id="R:r1",
            level="rule",
            text="y" * 50,
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=50,
            order=99,
        )
    )
    return out


def _chunk(cid: str, doc: str, a: int, b: int, node_ids: list[str], targets: list[str]) -> Chunk:
    return Chunk(
        chunk_id=cid,
        chunker="t",
        doc_id=doc,
        text="t",
        display_text="t",
        node_ids=node_ids,
        primary_node_id=None,
        section_ids=[],
        heading_path="",
        page_start=1,
        page_end=1,
        char_start=a,
        char_end=b,
        n_tokens=5,
        xref_targets=targets,
    )


def _expander() -> XrefExpander:
    chunks = [
        _chunk("c0", "A", 10, 100, ["A:s1"], ["A:s2(1)", "A:ch1"]),
        _chunk("c1", "A", 100, 180, ["A:s2", "A:s2(1)"], []),
        _chunk("c2", "A", 181, 250, ["A:s2", "A:s2(2)"], []),
        _chunk("c3", "R", 0, 50, ["R:r1"], ["A:s2(2)", "A:s1"]),
        _chunk("c4", "A", 250, 400, ["A:s3", "A:s3(1)"], ["A:s3(1)"]),
    ]
    return XrefExpander(ChunkCorpus("t", chunks), NodeGraph(_nodes()))


def test_expansion_follows_targets_in_rank_order() -> None:
    ex = _expander()
    out = ex.expand(["c3", "c0"], top_m=2, max_added=10)
    # c3 -> A:s2(2) (c2); its A:s1 target is c0, already in the top-m; c0 -> A:s2(1) (c1);
    # c0's chapter target is ignored
    assert [(e.chunk_id, e.source_chunk_id, e.target_node) for e in out] == [
        ("c2", "c3", "A:s2(2)"),
        ("c1", "c0", "A:s2(1)"),
    ]
    assert out[0].provenance == "xref_from:c3"


def test_expansion_caps_exclusions_and_self_targets() -> None:
    ex = _expander()
    assert [e.chunk_id for e in ex.expand(["c3", "c0"], top_m=2, max_added=1)] == ["c2"]
    assert [e.chunk_id for e in ex.expand(["c3", "c0"], top_m=2, exclude={"c2"})] == ["c1"]
    assert [e.chunk_id for e in ex.expand(["c3", "c0"], top_m=1)] == ["c2", "c0"]
    assert ex.expand(["c4"], top_m=5) == []  # target inside the source chunk itself
    assert ex.expand(["c3"], top_m=5, max_added=0) == []


def test_target_to_chunk_mapping_prefers_full_cover_then_start() -> None:
    ex = _expander()
    assert ex.chunks_for_target("A:s2", limit=2) == ["c1", "c2"]  # both inside; c1 holds start
    assert ex.chunks_for_target("A:s2(2)") == ["c2"]
    assert ex.chunks_for_target("A:nope") == []


def test_expansion_works_for_every_chunker(retrieval_dir: Path) -> None:
    nodes = load_nodes(retrieval_dir / "structure" / "nodes.jsonl")
    xrefs = load_xrefs(retrieval_dir / "structure" / "xrefs.jsonl")
    graph = NodeGraph(nodes)
    for chunker in ("fixed_64", "structural"):
        corpus = load_chunks(retrieval_dir, chunker)
        ex = XrefExpander(corpus, graph, xrefs)
        src = next(
            c
            for c in corpus.chunks
            if c.doc_id == "testrules2024" and "testact:s3(1)" in c.xref_targets
        )
        added = ex.expand([src.chunk_id], top_m=1, max_added=5)
        hit = [e for e in added if e.target_node == "testact:s3(1)"]
        assert hit, chunker
        assert "testact:s3(1)" in corpus.by_id[hit[0].chunk_id].node_ids
        assert all(corpus.by_id[e.chunk_id].doc_id in ("testact", "testrules2024") for e in added)
