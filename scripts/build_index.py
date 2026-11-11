"""Build (or load cached) retrieval indexes for the chunked corpus.

Usage:
    python scripts/build_index.py --chunker all --models bge-small bge-base bge-large
                                  [--collection NAME] [--config FILE] [--force] [--no-chroma]
Defaults come from the collection (cs_ai: chunkers fixed_512 + structural, models bge-small +
bge-base). Embeddings are computed in batches with a progress line and an ETA; a partial result
is saved after every batch, so an interrupted run resumes where it stopped.
For every chunker: builds the BM25 index; for every model: computes or loads the corpus
embeddings (data/processed/emb/<model>/<chunker>.npy, rebuilt automatically when the chunks
changed) and builds a chromadb collection with the same vectors, then checks that Chroma's
top-10 equals exact NumPy search for every chunk used as a query. Prints timings and sizes and
writes data/processed/emb/index_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from ragsys.chunking.pipeline import chunker_names
from ragsys.collections import add_collection_arguments, collection_from_args, config_path
from ragsys.config import load_chunking_config, resolve_path
from ragsys.retrieval.config import load_retrieval_settings
from ragsys.retrieval.pipeline import RetrievalResources
from ragsys.retrieval.store import ChromaStore, chroma_collection_name
from ragsys.utils.console import emit
from ragsys.utils.runinfo import collect_run_info
from ragsys.utils.seed import set_seed


def chroma_parity(store: ChromaStore, numpy_store: Any, emb: Any, k: int = 10) -> dict[str, Any]:
    identical, overlaps = 0, []
    for i in range(emb.shape[0]):
        a = {cid for cid, _s in numpy_store.search(emb[i], k)}
        b = {cid for cid, _s in store.search(emb[i], k)}
        overlaps.append(len(a & b) / max(1, len(a)))
        identical += a == b
    n = emb.shape[0]
    return {
        "queries": n,
        "identical_top10": identical,
        "mean_overlap": sum(overlaps) / max(1, n),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--chunker", nargs="+", default=["all"], help="chunker name(s) or 'all'")
    ap.add_argument("--models", nargs="+", default=None, help="default: the collection's")
    ap.add_argument("--config", default=None, help="default: the collection's retrieval config")
    ap.add_argument("--chunking-config", default=None)
    add_collection_arguments(ap)
    ap.add_argument("--force", action="store_true", help="recompute embeddings even if cached")
    ap.add_argument("--no-chroma", action="store_true", help="skip the Chroma build/parity check")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    coll = collection_from_args(args)
    settings = load_retrieval_settings(
        resolve_path(config_path(args.config, coll.retrieval_config))
    )
    set_seed(settings.seed)
    if args.chunker == ["all"]:
        ccfg = config_path(args.chunking_config, coll.chunking_config)
        chunkers = chunker_names(load_chunking_config(resolve_path(ccfg)))
    else:
        chunkers = args.chunker
    models = args.models or list(coll.spec.pipeline.index_models)
    res = RetrievalResources(settings)
    res.embed_progress = emit
    report: dict[str, Any] = {"chunkers": {}, "models": {}}
    t_all = time.perf_counter()
    for chunker in chunkers:
        corpus = res.corpus(chunker)
        t0 = time.perf_counter()
        bm = res.bm25(chunker)
        bm_s = time.perf_counter() - t0
        emit(
            f"{chunker:11s} chunks={len(corpus):4d}  BM25 built in {bm_s * 1000:.0f} ms "
            f"(vocab {bm.vocab_size}, avg {bm.avg_doc_len:.0f} tokens/chunk)"
        )
        entry: dict[str, Any] = {
            "chunks": len(corpus),
            "bm25": {"seconds": round(bm_s, 4), "vocab": bm.vocab_size, "avg_len": bm.avg_doc_len},
            "embeddings": {},
        }
        for model in models:
            name = settings.dense_model_name(model)
            emb, info = res.embeddings(model, chunker, force=args.force)
            status = f"computed ({info.reason})" if info.rebuilt else "cached"
            line = (
                f"    {model:10s} {status:28s} {info.seconds:7.1f} s  "
                f"dim={info.dim:4d}  {info.bytes / 1e6:6.2f} MB  {info.npy_path.parent.name}/"
                f"{info.npy_path.name}"
            )
            e: dict[str, Any] = {
                "model": name,
                "rebuilt": info.rebuilt,
                "reason": info.reason,
                "seconds": round(info.seconds, 2),
                "dim": info.dim,
                "bytes": info.bytes,
                "path": str(info.npy_path),
            }
            if not args.no_chroma:
                t0 = time.perf_counter()
                store = ChromaStore(
                    corpus.ids,
                    emb,
                    chroma_collection_name(chunker, name),
                    persist_dir=resolve_path(settings.paths.chroma_dir),
                    fingerprint=f"{corpus.texts_hash()}:{info.npy_path.stat().st_mtime_ns}",
                )
                build_s = time.perf_counter() - t0
                parity = chroma_parity(store, res.vector_store(model, chunker), emb)
                store.close()
                line += (
                    f"  chroma {'built' if store.rebuilt else 'reused'} {build_s:5.1f} s, parity "
                    f"{parity['identical_top10']}/{parity['queries']} identical top-10"
                )
                e["chroma"] = {"seconds": round(build_s, 2), "rebuilt": store.rebuilt, **parity}
            emit(line)
            entry["embeddings"][model] = e
        report["chunkers"][chunker] = entry
    total = time.perf_counter() - t_all
    emit(f"total {total:.1f} s")
    report["total_seconds"] = round(total, 2)
    report["settings"] = settings.model_dump(mode="json", exclude={"grid"})
    report["run_info"] = collect_run_info()
    out = res.emb_dir / "index_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    emit(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
