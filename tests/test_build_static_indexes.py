"""The static deployment's export (scripts/build_static_indexes.py): int8 vectors with per-vector
scales, the binary format the browser reads, the worker's prompts in sync with the Python
prompts, and the committed data matching its manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from agent_helpers import TEXTS, make_chunk
from ragsys.assistant.config import load_assistant_settings
from ragsys.config import REPO_ROOT

DATA = REPO_ROOT / "web-static" / "public" / "data"
PROMPTS = REPO_ROOT / "deploy" / "worker" / "src" / "prompts.json"


@pytest.fixture(scope="module")
def bsi(script):  # type: ignore[no-untyped-def]
    return script("build_static_indexes")


def test_quantize_round_trip_and_format(bsi) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(7)
    m = rng.normal(size=(50, 384)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    q, scale = bsi.quantize(m)
    assert q.dtype == np.int8 and scale.dtype == np.float32 and np.abs(q).max() == 127
    deq = bsi.dequantize(q, scale)
    assert np.max(np.abs(deq - m)) <= scale.max() / 2 + 1e-7  # rounding error at most half a step
    blob = bsi.encode_vectors(q, scale)
    assert blob[:4] == b"RAGV" and len(blob) == 16 + 4 * 50 + 50 * 384
    q2, s2 = bsi.decode_vectors(blob)
    assert np.array_equal(q2, q) and np.array_equal(s2, scale)
    with pytest.raises(ValueError):
        bsi.decode_vectors(b"XXXX" + blob[4:])
    query = m[3]
    assert bsi.search_int8(q, scale, query, 1) == [3]
    exact = np.argsort(-(m @ query), kind="stable")[:5].tolist()
    assert bsi.search_int8(q, scale, query, 5) == exact


def test_export_collection_writes_rows_and_vectors(bsi, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from ragsys.assistant.index import DenseIndex
    from ragsys.assistant.service import AssistantCollection
    from ragsys.retrieval.fakes import HashingEncoder

    chunks = [make_chunk(i, t, 60) for i, t in enumerate(TEXTS, start=1)]
    idx = DenseIndex.build(chunks, HashingEncoder(64), "books")
    coll = AssistantCollection(
        "books", "Books", tmp_path, tmp_path, profile="consumer_law", examples=["q?"]
    )
    entry = bsi.export_collection(coll, idx, tmp_path / "out", load_assistant_settings())
    rows = json.loads((tmp_path / "out" / "books" / "passages.json").read_text(encoding="utf-8"))
    assert rows["fields"] == bsi.ROW_FIELDS and len(rows["rows"]) == len(TEXTS)
    first = rows["rows"][0]
    assert first[0] == "structural:testact:00001" and first[4] == "1-2" and first[5] == 60
    assert first[6] == TEXTS[0]  # display text, not the embedded text with its header
    q, scale = bsi.decode_vectors((tmp_path / "out" / "books" / "vectors.bin").read_bytes())
    assert q.shape == (len(TEXTS), 64) and len(scale) == len(TEXTS)
    assert entry["threshold"] == 0.62 and entry["profile"] == "consumer_law"
    assert (
        entry["grounded"]["budget_tokens"] == 2048
        and entry["files"]["vectors"]["bytes"] == 16 + 4 * 6 + 6 * 64
    )


def test_worker_prompts_match_the_python_prompts(bsi) -> None:  # type: ignore[no-untyped-def]
    if not PROMPTS.exists():
        pytest.skip("deploy/worker/src/prompts.json not generated")
    stored = json.loads(PROMPTS.read_text(encoding="utf-8"))
    assert stored == bsi.worker_prompts(), "regenerate with python scripts/build_static_indexes.py"
    assert set(stored["profiles"]) == {
        "consumer_law",
        "cs_ai",
        "indian_law",
        "human_biology",
        "physics_chemistry",
        "uploads",
    }


def test_committed_static_data_matches_its_manifest() -> None:
    path = DATA / "manifest.json"
    if not path.exists():
        pytest.skip("web-static/public/data not built")
    man = json.loads(path.read_text(encoding="utf-8"))
    settings = load_assistant_settings()
    assert man["gate"]["default_threshold"] == settings.gate.default_threshold
    assert man["context_tokens"] == 2048 and man["limits"]["max_message_chars"] == 4000
    assert {c["id"] for c in man["collections"]} == {
        "consumer_law",
        "cs_ai",
        "indian_law",
        "human_biology",
        "physics_chemistry",
    }
    for c in man["collections"]:
        assert c["threshold"] == settings.gate.threshold(c["id"])
        assert c["sources"], c["id"]
        for f in c["files"].values():
            data = (DATA / f["path"]).read_bytes()
            assert len(data) == f["bytes"] and hashlib.sha256(data).hexdigest() == f["sha256"], f[
                "path"
            ]
    assert not list(DATA.rglob("*.pdf"))
