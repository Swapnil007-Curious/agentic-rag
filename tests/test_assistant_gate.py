"""The grounding gate: follow-up queries, thresholds per source, passage packing, and the gate
thresholds of configs/assistant.yaml recomputed from the DEV retrieval outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agent_helpers import TEXTS, make_chunk
from ragsys.assistant.config import (
    GateSettings,
    derive_gate_threshold,
    load_assistant_settings,
)
from ragsys.assistant.gate import GroundingGate, gate_queries, is_follow_up, merge_hits
from ragsys.assistant.index import DenseIndex, Hit, IndexUnavailable, LockedEncoder
from ragsys.config import REPO_ROOT
from ragsys.retrieval.fakes import HashingEncoder

ENC = HashingEncoder(64)


def _index(texts: list[str], source: str, doc: str = "testact", tokens: int = 100) -> DenseIndex:
    chunks = [make_chunk(i, t, tokens, doc=doc) for i, t in enumerate(texts, start=1)]
    return DenseIndex.build(chunks, ENC, source)


def _gate(**kw: object) -> GroundingGate:
    return GroundingGate(LockedEncoder(ENC), GateSettings.model_validate({"search_k": 10, **kw}))


def test_follow_up_detection_and_queries() -> None:
    assert is_follow_up("and what about its complexity?", 12)
    assert is_follow_up("Why?", 12)
    assert not is_follow_up("What is the time complexity of quicksort in the worst case?", 12)
    assert not is_follow_up("explain " + "word " * 20 + "it", 12)  # too long to be a follow-up
    assert gate_queries("Explain heaps", "What is a tree?", 12) == ["Explain heaps"]
    q = gate_queries("and its complexity?", "How does quicksort work?", 12)
    assert q == ["and its complexity?", "How does quicksort work?\nand its complexity?"]
    assert gate_queries("why?", None, 12) == ["why?"]


def test_gate_above_and_below_threshold() -> None:
    idx = _index(TEXTS, "books")
    gate = _gate()
    hit = gate.decide(TEXTS[1], None, [(idx, 0.5)], 2048)
    assert hit.grounded and hit.checks[0].passed and hit.checks[0].top_score > 0.8
    assert (
        hit.context is not None and hit.context.passages[0].chunk_id == "structural:testact:00002"
    )
    assert hit.context.labels[0] == "C1" and hit.embed_ms >= 0
    miss = gate.decide("write a poem about the sea", None, [(idx, 0.5)], 2048)
    assert not miss.grounded and miss.context is None and not miss.checks[0].passed
    # the same message passes a lower threshold
    assert gate.decide("write a poem about the sea", None, [(idx, -1.0)], 2048).grounded
    assert not gate.decide(TEXTS[1], None, [], 2048).grounded


def test_follow_up_uses_the_previous_message() -> None:
    idx = _index(TEXTS, "books")
    gate = _gate()
    alone = gate.decide("and why is that?", None, [(idx, 0.5)], 2048)
    assert not alone.grounded
    followed = gate.decide("and why is that?", TEXTS[2], [(idx, 0.5)], 2048)
    assert followed.grounded and followed.context is not None
    assert followed.context.passages[0].chunk_id == "structural:testact:00003"


def test_packing_respects_budget_and_max_passages() -> None:
    idx = _index(TEXTS, "books", tokens=300)
    gate = _gate(max_passages=8)
    d = gate.decide(TEXTS[0], None, [(idx, -1.0)], 1024)
    assert d.context is not None and d.context.used_tokens <= 1024 and len(d.context.passages) == 3
    d = gate.decide(TEXTS[0], None, [(idx, -1.0)], 2048)
    assert d.context is not None and len(d.context.passages) == 6  # all six fit 2048
    d = _gate(max_passages=2).decide(TEXTS[0], None, [(idx, -1.0)], 2048)
    assert d.context is not None and len(d.context.passages) == 2
    # a single passage larger than the budget is still given (min_passages = 1)
    huge = _index(TEXTS[:1], "books", tokens=5000)
    d = gate.decide(TEXTS[0], None, [(huge, -1.0)], 1024)
    assert d.context is not None and len(d.context.passages) == 1


def test_sources_pass_their_own_thresholds_and_merge() -> None:
    books = _index(TEXTS, "books")
    upload = _index(
        ["The station recorded a melt rate of 4.2 centimetres per day."], "upload", doc="up_aaaa"
    )
    gate = _gate()
    d = gate.decide(
        "melt rate centimetres per day station", None, [(books, 0.5), (upload, 0.5)], 2048
    )
    assert [c.source for c in d.checks if c.passed] == ["upload"]
    assert d.context is not None and all(p.doc_id == "up_aaaa" for p in d.context.passages)
    assert d.source_of(d.context.passages[0].chunk_id) == "upload"
    both = gate.decide(TEXTS[3], None, [(books, -1.0), (upload, -1.0)], 4096)
    assert both.context is not None and {p.doc_id for p in both.context.passages} == {
        "testact",
        "up_aaaa",
    }
    scores = [h.score for h in both.hits]
    assert scores == sorted(scores, reverse=True)


def test_merge_hits_keeps_best_score_per_chunk() -> None:
    c = make_chunk(1, "text")
    merged = merge_hits([[Hit(c, 0.4, "a")], [Hit(c, 0.7, "b")]])
    assert len(merged) == 1 and merged[0].score == 0.7 and merged[0].source == "b"


def test_index_load_refuses_mismatched_embeddings(tmp_path: Path) -> None:
    from assistant_helpers import write_index

    chunks = [make_chunk(i, t) for i, t in enumerate(TEXTS, start=1)]
    write_index(tmp_path, chunks)
    idx = DenseIndex.load(tmp_path, tmp_path / "emb", "fake:hash-64", "structural", "books")
    assert len(idx) == len(TEXTS) and np.allclose(
        np.linalg.norm(idx.matrix, axis=1), 1.0, atol=1e-5
    )
    before = sorted(p.name for p in tmp_path.rglob("*"))
    write_index(tmp_path, chunks[:-1])  # chunks change...
    (tmp_path / "emb" / "fake_hash-64" / "structural.json").write_text(
        '{"chunk_ids": [], "texts_sha256": "x"}', encoding="utf-8"
    )
    with pytest.raises(IndexUnavailable):
        DenseIndex.load(tmp_path, tmp_path / "emb", "fake:hash-64", "structural", "books")
    with pytest.raises(IndexUnavailable):
        DenseIndex.load(tmp_path, tmp_path / "emb", "fake:hash-32", "structural", "books")
    assert sorted(p.name for p in tmp_path.rglob("*")) == before  # nothing was written


def test_configured_thresholds_follow_the_rule() -> None:
    s = load_assistant_settings()
    assert s.gate.default_threshold == 0.62 and s.gate.upload_threshold == 0.62
    assert s.gate.threshold("indian_law") == 0.62 and s.gate.threshold(None) == 0.62
    assert s.context_budget("hosted") == 2048 and s.context_budget("local") == 1024
    assert s.limits.max_message_chars == 4000 and s.limits.max_output_tokens == 1500
    assert [(r.max, r.window_s) for r in s.rate_limits.per_ip] == [(20, 600), (200, 86400)]
    assert s.uploads.max_files == 3 and s.uploads.max_mb == 20 and s.uploads.max_pages == 300
    assert s.uploads.ttl_s == 7200 and s.history.max_turns == 10
    checked = 0
    for name, rel in s.gate.derived_from.items():
        path = REPO_ROOT / rel
        if not path.exists():  # per-query files are git-ignored (fresh clone)
            continue
        assert derive_gate_threshold(path, s.gate.percentile) == s.gate.thresholds[name], name
        checked += 1
    if checked == 0:
        pytest.skip("DEV per-query retrieval files are not present")


def test_derive_rule_on_a_synthetic_file(tmp_path: Path) -> None:
    import json

    rows = [{"answerable": True, "top1_first_stage_score": 0.60 + i / 100} for i in range(20)]
    rows.append({"answerable": False, "top1_first_stage_score": 0.1})
    p = tmp_path / "per_query.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert derive_gate_threshold(p, 5.0) == 0.6  # p5 = 0.6095 -> rounded down
    assert derive_gate_threshold(p, 50.0) == 0.69
