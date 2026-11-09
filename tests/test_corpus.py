"""Integration test on the real PDFs in data/raw (skipped when they are absent, e.g. in CI).

Run with:  pytest -q -m corpus
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ragsys.chunking.pipeline import chunker_names, load_corpus, run_chunker
from ragsys.config import REPO_ROOT, load_chunking_config, load_ingest_config
from ragsys.ingest.pipeline import run_ingest

CFG = load_ingest_config(REPO_ROOT / "configs" / "ingest.yaml")
RAW = REPO_ROOT / CFG.paths.raw_dir
MISSING = [d.file for d in CFG.enabled_documents() if not (RAW / d.file).exists()]

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(bool(MISSING), reason=f"real corpus PDFs not present in {RAW}: {MISSING}"),
]


@pytest.fixture(scope="module")
def real(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("corpus")
    return run_ingest(CFG, RAW, out, write=True), out


def test_all_documents_parse(real) -> None:  # type: ignore[no-untyped-def]
    result, _ = real
    rep = result.report
    assert rep["skipped"] == []
    for d in CFG.enabled_documents():
        doc = rep["documents"][d.doc_id]
        assert doc["sections_or_rules"] > 0, d.doc_id
        print(d.doc_id, doc["sections_or_rules"], doc["numbering"], doc["warnings"])


def test_cpa2019_numbering(real) -> None:  # type: ignore[no-untyped-def]
    result, _ = real
    doc = result.report["documents"]["cpa2019"]
    num = doc["numbering"]
    gap_warnings = [w for w in doc["warnings"] if w.startswith("gaps in")]
    # either contiguous, or every gap is surfaced as a warning in the report
    assert num["missing"] == [] or gap_warnings, num
    print("cpa2019 numbering:", num, "warnings:", gap_warnings)
    assert num["count"] == num["max_number"] == 107


def test_no_devanagari_in_nodes(real) -> None:  # type: ignore[no-untyped-def]
    result, _ = real
    bad = [n.node_id for n in result.nodes if re.search(r"[ऀ-ॿ]", n.text)]
    assert bad == []


def test_definitions_and_xrefs(real) -> None:  # type: ignore[no-untyped-def]
    result, _ = real
    defs = {n.node_id: n.defined_term for n in result.nodes if n.defined_term}
    assert defs["cpa2019:s2(7)"] == "consumer"
    assert defs["ecom2020:r3(1)(b)"] == "e-commerce entity"
    edges = {(x.source_node, x.target_node) for x in result.xrefs}
    assert ("jurisdiction2021:r3", "cpa2019:s34(1)-prov1") in edges
    assert ("ecom2020:r3(1)(k)", "cpa2019:s2(37)") in edges


def test_all_chunkers_run(real, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, out = real
    cfg = load_chunking_config(REPO_ROOT / "configs" / "chunking.yaml")
    cfg.paths.cache_dir = REPO_ROOT / cfg.paths.cache_dir  # reuse the persistent embedding cache
    docs = load_corpus(out)
    for name in chunker_names(cfg):
        stats = run_chunker(name, cfg, docs, out / "chunks")
        assert stats["count"] > 0, name
        if name == "structural":
            assert stats["chunks_crossing_section_boundaries"] == 0
        print(name, stats["count"], stats["tokens"])
