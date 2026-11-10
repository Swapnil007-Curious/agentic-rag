"""Relevance definition (own-text coverage, descendants/ancestors, ~2/-prov IDs) and metrics."""

from __future__ import annotations

import math

import pytest

from ragsys.chunking.base import Chunk
from ragsys.eval.metrics import (
    auroc,
    budget_coverage,
    budget_prefix,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
)
from ragsys.eval.relevance import RelevanceJudge
from ragsys.retrieval.corpus import ChunkCorpus, NodeGraph
from ragsys.structure.base import Node

PARTS = {
    "s5": "5. Heading.—Intro text\n",
    "s5(1)": "(1) Sub one lead:\n",
    "s5(1)(a)": "(a) alpha clause;\n",
    "s5(1)(a)~2": "(a) dup clause;\n",
    "s5(1)-prov1": "Provided that proviso text.\n",
    "s5(2)": "(2) Sub two text.\n",
    "s6": "6. Other.—Different section.",
}
TEXT = "CHAPTER I\n" + "".join(PARTS.values())


def _off(key: str) -> int:
    return TEXT.index(PARTS[key])


def _end(key: str) -> int:
    return _off(key) + len(PARTS[key].rstrip("\n"))


SPANS = {
    "ch1": (0, len(TEXT)),
    "s5": (_off("s5"), _end("s5(2)")),
    "s5(1)": (_off("s5(1)"), _end("s5(1)-prov1")),
    "s5(1)(a)": (_off("s5(1)(a)"), _end("s5(1)(a)")),
    "s5(1)(a)~2": (_off("s5(1)(a)~2"), _end("s5(1)(a)~2")),
    "s5(1)-prov1": (_off("s5(1)-prov1"), _end("s5(1)-prov1")),
    "s5(2)": (_off("s5(2)"), _end("s5(2)")),
    "s6": (_off("s6"), _end("s6")),
}
PARENT = {
    "ch1": None,
    "s5": "ch1",
    "s5(1)": "s5",
    "s5(1)(a)": "s5(1)",
    "s5(1)(a)~2": "s5(1)",
    "s5(1)-prov1": "s5(1)",
    "s5(2)": "s5",
    "s6": "ch1",
}
LEVEL = {
    "ch1": "chapter",
    "s5": "section",
    "s6": "section",
    "s5(1)": "subsection",
    "s5(2)": "subsection",
    "s5(1)(a)": "clause",
    "s5(1)(a)~2": "clause",
    "s5(1)-prov1": "proviso",
}


def nodes() -> list[Node]:
    out = []
    for i, (k, (a, b)) in enumerate(SPANS.items()):
        out.append(
            Node(
                doc_id="d",
                doc_title="D",
                doc_type="act",
                node_id=f"d:{k}",
                level=LEVEL[k],  # type: ignore[arg-type]
                text=TEXT[a:b],
                parent_id=f"d:{PARENT[k]}" if PARENT[k] else None,
                children=[f"d:{c}" for c, p in PARENT.items() if p == k],
                page_start=1,
                page_end=1,
                char_start=a,
                char_end=b,
                order=i,
            )
        )
    return out


def chunk(cid: str, start: int, end: int, all_nodes: list[Node]) -> Chunk:
    overl = [n for n in all_nodes if n.char_start < end and n.char_end > start]
    return Chunk(
        chunk_id=cid,
        chunker="t",
        doc_id="d",
        text=TEXT[start:end],
        display_text=TEXT[start:end],
        node_ids=[n.node_id for n in overl if n.level != "chapter"],
        primary_node_id=None,
        section_ids=[n.node_id for n in overl if n.level == "section"],
        heading_path="D",
        page_start=1,
        page_end=1,
        char_start=start,
        char_end=end,
        n_tokens=10,
        xref_targets=[],
    )


@pytest.fixture(scope="module")
def judge() -> RelevanceJudge:
    ns = nodes()
    chunks = [
        chunk("intro", *(_off("s5"), _end("s5")), ns),  # only s5's own lead-in text
        chunk("dup", *(_off("s5(1)(a)~2"), _end("s5(1)(a)~2")), ns),
        chunk("prov", *(_off("s5(1)-prov1"), _end("s5(1)-prov1")), ns),
        chunk("two", *(_off("s5(2)"), _end("s5(2)")), ns),
        chunk("six", *(_off("s6"), _end("s6")), ns),
    ]
    return RelevanceJudge(NodeGraph(ns), ChunkCorpus("t", chunks))


def test_covered_nodes_use_own_text(judge: RelevanceJudge) -> None:
    assert judge.covered_nodes("intro") == {"d:s5"}
    # "two" overlaps s5 (its parent) but none of s5's own text
    assert "d:s5" in judge.corpus.by_id["two"].node_ids
    assert judge.covered_nodes("two") == {"d:s5(2)"}
    assert judge.covered_nodes("dup") == {"d:s5(1)(a)~2"}


def test_relevance_descendants_ancestors_and_suffix_ids(judge: RelevanceJudge) -> None:
    def rel(g: str) -> set[str]:
        return {c.chunk_id for c in judge.corpus.chunks if judge.is_relevant(c.chunk_id, [g])}

    assert rel("d:s5(1)(a)~2") == {"dup", "intro"}  # itself + ancestor s5's lead-in
    assert rel("d:s5(1)(a)") == {"intro"}  # the "~2" sibling is not relevant
    assert rel("d:s5(1)") == {"dup", "prov", "intro"}  # descendants (incl. -prov1) + ancestor
    assert rel("d:s5(1)-prov1") == {"prov", "intro"}
    assert rel("d:s5(2)") == {"two", "intro"}
    assert rel("d:ch1") == {"intro", "dup", "prov", "two", "six"}  # chapter gold: descendants
    assert rel("d:s6") == {"six"}  # the chapter is never used as an ancestor
    assert rel("d:missing") == set()
    assert judge.relevant_golds("dup", ["d:s5(1)", "d:s6"]) == {"d:s5(1)"}
    assert judge.n_relevant_chunks(["d:s5(1)-prov1"]) == 2
    assert judge.n_relevant_chunks([]) == 0


PROFILE = [set(), {"g1"}, set(), {"g1", "g2"}, {"g3"}]
GOLDS = ["g1", "g2", "g3", "g4"]


def test_rank_metrics_hand_computed() -> None:
    assert recall_at_k(PROFILE, GOLDS, 1) == 0.0
    assert recall_at_k(PROFILE, GOLDS, 3) == 0.25
    assert recall_at_k(PROFILE, GOLDS, 5) == 0.75  # multi-gold recall
    assert recall_at_k(PROFILE, GOLDS, 10) == 0.75
    assert (hit_at_k(PROFILE, 1), hit_at_k(PROFILE, 3)) == (0.0, 1.0)
    assert mrr_at_k(PROFILE, 10) == 0.5 and mrr_at_k(PROFILE, 1) == 0.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(PROFILE, 10, n_relevant_total=3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([set()], 10, 0) == 0.0
    assert recall_at_k(PROFILE, [], 5) == 0.0


def test_token_budget_coverage() -> None:
    toks = [100, 200, 300, 400, 50]
    assert budget_prefix(toks, 512) == 2  # 100 + 200 fit, 300 would exceed
    assert budget_coverage(PROFILE, toks, GOLDS, 512) == 0.25
    assert budget_prefix(toks, 1024) == 4  # stops at the first chunk that does not fit
    assert budget_coverage(PROFILE, toks, GOLDS, 1024) == 0.5
    assert budget_coverage(PROFILE, toks, GOLDS, 2048) == 0.75
    assert budget_prefix([600], 512) == 0


def test_auroc_and_bootstrap() -> None:
    assert auroc([3.0, 2.0], [1.0, 2.0]) == pytest.approx(0.875)
    assert auroc([1.0], []) is None
    a, b = [1.0, 0.0, 1.0, 1.0, 0.5], [0.0, 0.0, 1.0, 0.0, 0.5]
    r1 = paired_bootstrap(a, b, n_resamples=1000, seed=42)
    r2 = paired_bootstrap(a, b, n_resamples=1000, seed=42)
    assert r1 == r2  # reproducible with the same seed
    assert r1["mean_diff"] == pytest.approx(0.4) and r1["n"] == 5
    assert r1["ci_low"] <= 0.4 <= r1["ci_high"]  # type: ignore[operator]
    x = [((i * 7) % 11) / 10 for i in range(30)]
    y = [((i * 5) % 13) / 12 for i in range(30)]
    assert paired_bootstrap(x, y, seed=1) == paired_bootstrap(x, y, seed=1)
    assert paired_bootstrap(x, y, seed=1) != paired_bootstrap(x, y, seed=2)
    assert paired_bootstrap([], [])["mean_diff"] is None
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [])
