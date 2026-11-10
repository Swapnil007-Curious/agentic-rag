"""Vector stores (NumPy vs Chroma parity) and the corpus embedding cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ragsys.retrieval.corpus import ChunkCorpus, load_chunks
from ragsys.retrieval.embed import corpus_embeddings, model_slug
from ragsys.retrieval.fakes import HashingEncoder
from ragsys.retrieval.store import ChromaStore, NumpyStore, chroma_collection_name


def test_numpy_store_exact_cosine() -> None:
    store = NumpyStore(["a", "b", "c"], np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 2.0]]))
    hits = store.search(np.array([1.0, 1.0]), 2)
    assert [h[0] for h in hits] == ["b", "a"]  # a and c tie; the earlier row wins
    assert abs(hits[0][1] - 1.4 / np.sqrt(2)) < 1e-6
    assert len(store.search(np.array([1.0, 0.0]), 10)) == 3


def test_chroma_top10_matches_numpy(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 16)).astype(np.float32)
    ids = [f"c{i:02d}" for i in range(60)]
    np_store = NumpyStore(ids, emb)
    chroma = ChromaStore(
        ids, emb, chroma_collection_name("fixture", "fake:hash-16"), tmp_path / "chroma", "fp1"
    )
    try:
        assert chroma.rebuilt and len(chroma) == 60
        for qi in range(12):
            q = rng.normal(size=16).astype(np.float32)
            a = np_store.search(q, 10)
            b = chroma.search(q, 10)
            assert {i for i, _s in a} == {i for i, _s in b}
            assert abs(a[0][1] - b[0][1]) < 1e-4, qi
    finally:
        chroma.close()
    reopened = ChromaStore(
        ids, emb, chroma_collection_name("fixture", "fake:hash-16"), tmp_path / "chroma", "fp1"
    )
    try:
        assert not reopened.rebuilt  # same fingerprint -> reused from disk
    finally:
        reopened.close()


def test_embedding_cache_detects_staleness(retrieval_dir: Path, tmp_path: Path) -> None:
    corpus = load_chunks(retrieval_dir, "structural")
    enc = HashingEncoder(32)
    emb, info = corpus_embeddings(enc, corpus, tmp_path)
    assert info.rebuilt and info.reason == "missing" and enc.calls == 1
    assert emb.shape == (len(corpus), 32)
    assert (tmp_path / model_slug(enc.name) / "structural.npy").exists()
    sidecar = (tmp_path / model_slug(enc.name) / "structural.json").read_text(encoding="utf-8")
    assert corpus.ids[0] in sidecar and corpus.texts_hash() in sidecar

    emb2, info2 = corpus_embeddings(enc, corpus, tmp_path)
    assert not info2.rebuilt and enc.calls == 1 and np.array_equal(emb, emb2)

    changed = [c.model_copy() for c in corpus.chunks]
    changed[0] = changed[0].model_copy(update={"text": changed[0].text + " amended"})
    _e, info3 = corpus_embeddings(enc, ChunkCorpus("structural", changed), tmp_path)
    assert info3.rebuilt and info3.reason == "chunk texts changed" and enc.calls == 2

    reordered = ChunkCorpus("structural", list(reversed(changed)))
    _e, info4 = corpus_embeddings(enc, reordered, tmp_path)
    assert info4.reason == "chunk ids changed"

    meta_path = tmp_path / model_slug(enc.name) / "structural.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, "model": "another/model"}), encoding="utf-8")
    _e, info5 = corpus_embeddings(enc, reordered, tmp_path)
    assert info5.rebuilt and info5.reason == "model changed"
    _e, info6 = corpus_embeddings(enc, reordered, tmp_path, force=True)
    assert info6.rebuilt and info6.reason == "forced"
