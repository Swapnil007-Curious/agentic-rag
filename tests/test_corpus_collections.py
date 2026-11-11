"""Integration checks on the real corpora (skipped when the PDFs are absent, e.g. in CI).

Run with:  pytest -q -m corpus

* consumer_law: re-ingesting and re-chunking reproduces data/processed byte for byte, so the
  multi-collection changes leave its structure, chunks (and therefore embeddings, caches and
  results) unchanged;
* cs_ai: every selected chapter is found, 502 pages are kept, licences are recorded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragsys.chunking.pipeline import load_corpus, run_chunker
from ragsys.collections import get_collection
from ragsys.config import load_chunking_config, load_ingest_config, resolve_path
from ragsys.ingest.pipeline import run_ingest

CL = get_collection("consumer_law")
CS = get_collection("cs_ai")
CL_CFG = load_ingest_config(resolve_path(CL.ingest_config))
CS_CFG = load_ingest_config(resolve_path(CS.ingest_config))
CL_RAW, CS_RAW = resolve_path(CL.raw_dir), resolve_path(CS.raw_dir)
CL_MISSING = [d.file for d in CL_CFG.enabled_documents() if not (CL_RAW / d.file).exists()]
CS_MISSING = [d.file for d in CS_CFG.enabled_documents() if not (CS_RAW / d.file).exists()]
CL_PROCESSED = resolve_path(CL.processed_dir)

pytestmark = pytest.mark.corpus


@pytest.mark.skipif(
    bool(CL_MISSING) or not (CL_PROCESSED / "chunks" / "structural.jsonl").exists(),
    reason="consumer_law PDFs or processed outputs not present",
)
def test_consumer_law_outputs_reproduce_byte_for_byte(tmp_path: Path) -> None:
    run_ingest(CL_CFG, CL_RAW, tmp_path, write=True)
    for rel in ("structure/nodes.jsonl", "structure/xrefs.jsonl", "structure/footnotes.jsonl"):
        assert (tmp_path / rel).read_bytes() == (CL_PROCESSED / rel).read_bytes(), rel
    for p in (CL_PROCESSED / "clean").iterdir():
        assert (tmp_path / "clean" / p.name).read_bytes() == p.read_bytes(), p.name
    ccfg = load_chunking_config(resolve_path(CL.chunking_config))
    docs = load_corpus(tmp_path)
    for name in ("fixed_256", "fixed_512", "structural"):
        run_chunker(name, ccfg, docs, tmp_path / "chunks")
        new = (tmp_path / "chunks" / f"{name}.jsonl").read_bytes()
        assert new == (CL_PROCESSED / "chunks" / f"{name}.jsonl").read_bytes(), name


@pytest.mark.skipif(bool(CS_MISSING), reason=f"cs_ai PDFs not present in {CS_RAW}: {CS_MISSING}")
def test_cs_ai_chapters_pages_and_licences(tmp_path: Path) -> None:
    res = run_ingest(CS_CFG, CS_RAW, tmp_path, write=True)
    docs = res.report["documents"]
    assert sum(d["pages_kept"] for d in docs.values()) == 502
    assert all(not d["chapters_not_found"] for d in docs.values())
    assert docs["progit"]["licence"]["from_pdf"]
    assert docs["progit"]["licence"]["licence"] == "CC BY-NC-SA 3.0 Unported"
    assert not docs["d2l"]["licence"]["from_pdf"] and not docs["ods"]["licence"]["from_pdf"]
    ids = {n.node_id for n in res.nodes}
    assert {"d2l:11.5", "ods:5.2.1", "progit:h-git-branching", "progit:h-rebasing"} <= ids
    assert all(d["code_blocks"] > 0 for d in docs.values())
