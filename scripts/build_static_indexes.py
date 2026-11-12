"""Export what the static web app (web-static/) and its LLM proxy (deploy/worker/) need.

Usage:
    python scripts/build_static_indexes.py [--out web-static/public/data]
        [--worker-prompts deploy/worker/src/prompts.json] [--no-parity]

Per collection (the app index the assistant serves: structural chunks + bge-small embeddings,
read without writing anything into the processed data):

* ``<id>/passages.json``: the fields the browser needs at run time, as rows
  ``[chunk_id, doc_id, heading_path, section_ids, pages, n_tokens, display_text]``;
* ``<id>/vectors.bin``: the embeddings quantised to int8 with one float32 scale per vector
  (``x ~= scale * q``), little endian: ``b"RAGV"``, version (u8), 3 padding bytes, dim (u32),
  count (u32), then ``count`` float32 scales, then ``count * dim`` int8 values.

``manifest.json`` holds the gate thresholds, context budget, history and upload limits,
notices, example questions, SOURCES attributions and the files with their sizes and digests;
each collection is downloaded only when it is selected. The worker's prompts (chat, summary,
and each profile's synthesizer and critic prompts) are written from the Python sources so the
two deployments use the same text. With parity on (default), web-static/tests/fixtures/
parity.json gets sentence-transformers embeddings of 20 queries and, for 20 queries per
collection, the top-5 chunk ids of the same search over the published int8 index (the browser
tests compare against them). Prints per-collection download sizes (raw and gzip).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ragsys.assistant.config import AssistantSettings, load_assistant_settings
from ragsys.assistant.grounded import UPLOAD_PROMPTS
from ragsys.assistant.index import DenseIndex
from ragsys.assistant.prompts import (
    BASE_PROMPT,
    CITE_REMINDER,
    GROUNDING_RULES,
    SUMMARY_PREFIX,
    SUMMARY_PROMPT,
)
from ragsys.assistant.service import AssistantCollection
from ragsys.assistant.setup import assistant_collections
from ragsys.chunking.base import Chunk
from ragsys.collections import DEFAULT_COLLECTIONS_FILE
from ragsys.config import REPO_ROOT, resolve_path
from ragsys.profiles import PROFILES

DEFAULT_OUT = Path("web-static/public/data")
DEFAULT_PROMPTS = Path("deploy/worker/src/prompts.json")
DEFAULT_PARITY = Path("web-static/tests/fixtures/parity.json")
MAGIC = b"RAGV"
VERSION = 1
HEADER = struct.Struct("<4sB3xII")  # magic, version, padding, dim, count
ROW_FIELDS = ["chunk_id", "doc_id", "heading_path", "section_ids", "pages", "n_tokens", "text"]
SOURCE_FIELDS = ("doc_id", "title", "authors", "edition", "licence", "licence_url", "source_url")


# --------------------------------------------------------------------------- vectors


def quantize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 values and one float32 scale per row with ``row ~= scale * q`` (max |q| = 127)."""
    m = np.asarray(matrix, dtype=np.float32)
    peak = np.abs(m).max(axis=1) if m.size else np.zeros(0, dtype=np.float32)
    scale = np.where(peak > 0, peak / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(m / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale[:, None].astype(np.float32)


def encode_vectors(q: np.ndarray, scale: np.ndarray) -> bytes:
    n, dim = q.shape
    return (
        HEADER.pack(MAGIC, VERSION, dim, n)
        + scale.astype("<f4").tobytes()
        + q.astype(np.int8).tobytes()
    )


def decode_vectors(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    magic, version, dim, n = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError("not a vectors.bin file")
    off = HEADER.size
    scale = np.frombuffer(blob, dtype="<f4", count=n, offset=off)
    q = np.frombuffer(blob, dtype=np.int8, count=n * dim, offset=off + 4 * n).reshape(n, dim)
    return q, scale


def search_int8(q: np.ndarray, scale: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    """The browser's search: dequantised cosine, best first, ties by row order."""
    scores = (q.astype(np.float32) @ query.astype(np.float32)) * scale
    order = np.lexsort((np.arange(len(scores)), -scores))
    return [int(i) for i in order[:k]]


# --------------------------------------------------------------------------- passages


def pages_of(c: Chunk) -> str:
    return str(c.page_start) if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"


def passage_rows(chunks: list[Chunk]) -> list[list[Any]]:
    return [
        [
            c.chunk_id,
            c.doc_id,
            c.heading_path,
            list(c.section_ids),
            pages_of(c),
            c.n_tokens,
            c.display_text,
        ]
        for c in chunks
    ]


def dumps(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def gz_size(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9, mtime=0))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- export


def export_collection(
    coll: AssistantCollection, idx: DenseIndex, out: Path, settings: AssistantSettings
) -> dict[str, Any]:
    q, scale = quantize(idx.matrix)
    passages = dumps({"fields": ROW_FIELDS, "rows": passage_rows(idx.chunks)})
    vectors = encode_vectors(q, scale)
    d = out / coll.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "passages.json").write_bytes(passages)
    (d / "vectors.bin").write_bytes(vectors)
    return {
        "id": coll.name,
        "label": coll.label,
        "description": coll.description,
        "notice": coll.notice,
        "examples": list(coll.examples),
        "evaluated": coll.evaluated,
        "profile": coll.profile or "uploads",
        "abstention_message": coll.abstention_message,
        "threshold": settings.gate.threshold(coll.name),
        "grounded": grounded_settings(coll),
        "count": len(idx),
        "dim": int(q.shape[1]),
        "sources": [{k: str(s.get(k, "")) for k in SOURCE_FIELDS} for s in coll.sources],
        "files": {
            "passages": {
                "path": f"{coll.name}/passages.json",
                "bytes": len(passages),
                "gzip_bytes": gz_size(passages),
                "sha256": sha(passages),
            },
            "vectors": {
                "path": f"{coll.name}/vectors.bin",
                "bytes": len(vectors),
                "gzip_bytes": gz_size(vectors),
                "sha256": sha(vectors),
            },
        },
    }


def grounded_settings_default() -> dict[str, Any]:
    return {
        "budget_tokens": 2048,
        "revision_budget_tokens": 3072,
        "min_passages": 1,
        "max_synth_calls": 2,
        "max_critic_calls": 2,
        "critic_on_guard_fail": False,
        "on_final_revise": "abstain",
    }


def grounded_settings(coll: AssistantCollection) -> dict[str, Any]:
    """Context budgets (hosted backend) and loop limits of the collection's grounded pipeline."""
    gen = coll.generation
    if gen is None:
        return grounded_settings_default()
    ctx = gen.context_for("hosted")
    return {
        "budget_tokens": ctx.budget_tokens,
        "revision_budget_tokens": ctx.revision_budget_tokens,
        "min_passages": ctx.min_passages,
        "max_synth_calls": gen.loop.max_synth_calls,
        "max_critic_calls": gen.loop.max_critic_calls,
        "critic_on_guard_fail": gen.loop.critic_on_guard_fail,
        "on_final_revise": gen.loop.on_final_revise,
    }


def manifest(settings: AssistantSettings, collections: list[dict[str, Any]]) -> dict[str, Any]:
    body = {
        "format": 1,
        "model": {
            "id": "bge-small-en-v1.5",
            "source": "BAAI/bge-small-en-v1.5 (ONNX export: Xenova/bge-small-en-v1.5)",
            "pooling": "cls",
            "normalize": True,
            "max_tokens": 512,
            "query_instruction": settings.query_instruction,
        },
        "gate": {
            "default_threshold": settings.gate.default_threshold,
            "upload_threshold": settings.gate.upload_threshold,
            "follow_up_max_words": settings.gate.follow_up_max_words,
            "search_k": settings.gate.search_k,
            "max_passages": settings.gate.max_passages,
        },
        "context_tokens": settings.context_budget("hosted"),
        "uploads_grounded": grounded_settings_default(),
        "history": {
            "max_turns": settings.history.max_turns,
            "budget_tokens": settings.history.budget_tokens.get("hosted", 3000),
            "summary_target_share": settings.history.summary_target_share,
            "summary_input_chars": settings.history.summary_input_chars,
        },
        "limits": {
            "max_message_chars": settings.limits.max_message_chars,
            "max_output_tokens": settings.limits.max_output_tokens,
        },
        "uploads": {
            "max_files": settings.uploads.max_files,
            "max_mb": settings.uploads.max_mb,
            "max_pages": settings.uploads.max_pages,
            "max_chunks_per_file": settings.uploads.max_chunks_per_file,
        },
        "general_examples": list(settings.general_examples),
        "collections": collections,
    }
    body["version"] = sha(dumps(body))[:16]
    return body


def worker_prompts() -> dict[str, Any]:
    profiles = {
        name: {
            "synth": p.generation.synth_system,
            "plain": p.generation.plain_rule,
            "critic": p.generation.critic_system,
        }
        for name, p in PROFILES.items()
    }
    profiles["uploads"] = {
        "synth": UPLOAD_PROMPTS.synth_system,
        "plain": UPLOAD_PROMPTS.plain_rule,
        "critic": UPLOAD_PROMPTS.critic_system,
    }
    return {
        "chat": {
            "base": BASE_PROMPT,
            "grounding_rules": GROUNDING_RULES,
            "cite_reminder": CITE_REMINDER,
            "summary": SUMMARY_PROMPT,
            "summary_prefix": SUMMARY_PREFIX,
        },
        "profiles": profiles,
    }


# --------------------------------------------------------------------------- parity


def parity_queries(coll: AssistantCollection, idx: DenseIndex, n: int = 20) -> list[str]:
    """Realistic questions: the collection's example questions, then questions about evenly
    spaced section headings."""
    out = [q for q in coll.examples if q.strip()][:n]
    step = max(1, len(idx) // n)
    for c in idx.chunks[::step]:
        if len(out) >= n:
            break
        head = c.heading_path.split(" > ")[-1].strip()
        q = f"Explain {head}."
        if head and q not in out:
            out.append(q)
    return out[:n]


def build_parity(
    settings: AssistantSettings,
    colls: list[AssistantCollection],
    indexes: dict[str, DenseIndex],
    vectors: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    from ragsys.retrieval.embed import SentenceTransformerEncoder

    enc = SentenceTransformerEncoder(
        settings.embedding_model, "cpu", 16, settings.query_instruction
    )
    general: list[str] = list(settings.general_examples)
    for c in colls:
        general += c.examples[:4]
    general = list(dict.fromkeys(general))[:20]
    gvec = enc.encode(general, is_query=True)
    per: dict[str, Any] = {}
    for c in colls:
        queries = parity_queries(c, indexes[c.name])
        qv = enc.encode(queries, is_query=True)
        q8, scale = vectors[c.name]
        rows = []
        for text, v in zip(queries, qv, strict=True):
            top = search_int8(q8, scale, v, 6)
            scores = (q8[top].astype(np.float32) @ v) * scale[top]
            rows.append(
                {
                    "query": text,
                    "top5": [indexes[c.name].chunks[i].chunk_id for i in top[:5]],
                    "scores": [round(float(s), 6) for s in scores],
                }
            )
        per[c.name] = rows
    return {
        "instruction": settings.query_instruction,
        "general": [
            {"query": q, "embedding": [round(float(x), 7) for x in v]}
            for q, v in zip(general, gvec, strict=True)
        ],
        "collections": per,
    }


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--worker-prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--parity-out", default=str(DEFAULT_PARITY))
    ap.add_argument("--no-parity", action="store_true", help="skip the browser parity fixture")
    ap.add_argument("--collections-file", default=str(DEFAULT_COLLECTIONS_FILE))
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    settings = load_assistant_settings()
    out = resolve_path(Path(args.out))
    colls = assistant_collections(args.collections_file, settings)
    entries = []
    indexes: dict[str, DenseIndex] = {}
    vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in colls:
        idx = DenseIndex.load(
            c.processed_dir, c.emb_dir, settings.embedding_model, settings.chunker, c.name
        )
        indexes[c.name] = idx
        entry = export_collection(c, idx, out, settings)
        vectors[c.name] = decode_vectors((out / c.name / "vectors.bin").read_bytes())
        entries.append(entry)
    man = manifest(settings, entries)
    (out / "manifest.json").write_bytes(dumps(man) + b"\n")
    prompts_path = resolve_path(Path(args.worker_prompts))
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text(
        json.dumps(worker_prompts(), indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not args.no_parity:
        parity_path = resolve_path(Path(args.parity_out))
        parity_path.parent.mkdir(parents=True, exist_ok=True)
        parity_path.write_bytes(dumps(build_parity(settings, colls, indexes, vectors)) + b"\n")
        print(f"parity fixture: {parity_path.relative_to(REPO_ROOT).as_posix()}")
    print(
        f"static data in {out.relative_to(REPO_ROOT).as_posix()} (manifest version {man['version']})"
    )
    print(
        f"{'collection':<20}{'chunks':>8}{'passages':>12}{'gzip':>10}{'vectors':>11}{'gzip':>10}{'download (gzip)':>17}"
    )
    tot_raw = tot_gz = 0
    for e in entries:
        p, v = e["files"]["passages"], e["files"]["vectors"]
        raw, gz = p["bytes"] + v["bytes"], p["gzip_bytes"] + v["gzip_bytes"]
        tot_raw += raw
        tot_gz += gz
        print(
            f"{e['id']:<20}{e['count']:>8}{p['bytes'] / 1e6:>10.2f}MB{p['gzip_bytes'] / 1e6:>8.2f}MB"
            f"{v['bytes'] / 1e6:>9.2f}MB{v['gzip_bytes'] / 1e6:>8.2f}MB{gz / 1e6:>15.2f}MB"
        )
    print(
        f"{'total':<20}{'':>8}{'':>12}{'':>10}{'':>11}{'':>10}{tot_gz / 1e6:>15.2f}MB  (raw {tot_raw / 1e6:.2f} MB)"
    )
    print(f"worker prompts: {prompts_path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
