"""Draft candidate evaluation questions with an LLM (for human review; never final).

Usage:
    python scripts/draft_questions.py --n 150 [--types definition numeric ...] [--seed 0]
                                      [--llm-config configs/llm.yaml] [--fake]
Samples target nodes stratified by document and node kind, in the type mix definition 15,
factual 20, procedural 15, numeric 15, cross_reference 15, multi_section 10 (scaled to --n),
asks the configured OpenAI-compatible endpoint (Ollama by default; key from LLM_API_KEY) for
one JSON draft per node and APPENDS the drafts to data/eval/drafts.jsonl with status "pending".
It never writes questions.jsonl and never drafts unanswerable questions. Nodes already used by
a draft or a question are skipped. --fake uses the built-in deterministic FakeLLM (no network).
Next step: python scripts/review_drafts.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ragsys.collections import add_collection_arguments, collection_from_args, config_path
from ragsys.config import load_ingest_config, resolve_path
from ragsys.eval.drafting import (
    allocate,
    build_pools,
    draft_questions,
    load_drafts,
    next_draft_number,
    save_drafts,
    stratified_sample,
)
from ragsys.eval.validate import load_questions
from ragsys.ingest.pipeline import load_nodes, load_xrefs
from ragsys.llm.client import ChatModel, LLMClient, load_llm_config
from ragsys.llm.fake import FakeLLM
from ragsys.profiles import get_profile
from ragsys.retrieval.corpus import NodeGraph
from ragsys.utils.console import emit

ALL_TYPES = [
    "definition",
    "factual",
    "procedural",
    "numeric",
    "cross_reference",
    "multi_section",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--types", nargs="+", choices=ALL_TYPES, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm-config", default=None, help="default: configs/llm.yaml")
    ap.add_argument("--fake", action="store_true", help="use the deterministic FakeLLM")
    ap.add_argument("--out", default=None, help="default: <eval_dir>/drafts.jsonl")
    ap.add_argument("--questions", default=None, help="default: <eval_dir>/questions.jsonl")
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--ingest-config", default=None)
    add_collection_arguments(ap)
    ap.add_argument(
        "--backend", default=None, help="LLM backend of configs/llm.yaml (default: local)"
    )
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    coll = collection_from_args(args)
    profile = get_profile(coll.profile)
    processed = resolve_path(config_path(args.processed_dir, coll.processed_dir))
    nodes_path = processed / "structure" / "nodes.jsonl"
    if not nodes_path.exists():
        emit(f"{nodes_path} not found; run scripts/ingest.py first")
        return 1
    nodes = load_nodes(nodes_path)
    xrefs = load_xrefs(processed / "structure" / "xrefs.jsonl")
    graph = NodeGraph(nodes)
    titles = {n.doc_id: n.doc_title for n in nodes}
    ing = resolve_path(config_path(args.ingest_config, coll.ingest_config))
    if ing.exists():
        titles.update({d.doc_id: d.display_title for d in load_ingest_config(ing).documents})

    llm: ChatModel
    base_url: str | None = None
    if args.fake:
        llm = FakeLLM()
        emit("using FakeLLM (deterministic canned drafts, no network) -- not real drafts")
    else:
        cfg = (
            load_llm_config(resolve_path(Path(args.llm_config)), args.backend)
            if args.llm_config
            else coll.llm(args.backend)
        )
        client = LLMClient(cfg)
        if not client.is_reachable():
            emit(
                f"LLM endpoint {cfg.base_url} is not reachable.\n"
                "  Ollama: install it, run `ollama pull " + cfg.model + "` and keep it running;\n"
                "  other providers: set base_url/model in configs/llm.yaml and $env:LLM_API_KEY;\n"
                "  or run with --fake to try the workflow without an LLM."
            )
            return 2
        llm, base_url = client, cfg.base_url
        emit(f"using {cfg.model} at {cfg.base_url}")

    out_path = resolve_path(config_path(args.out, coll.drafts_file))
    existing = load_drafts(out_path)
    qpath = resolve_path(config_path(args.questions, coll.questions_file))
    questions, _errs = load_questions(qpath) if qpath.exists() else ([], [])
    used = {g for d in existing for g in d.get("gold_node_ids", [])}
    used |= {g for q in questions for g in q.gold_node_ids}

    mix = {t: w for t, w in profile.type_mix.items() if args.types is None or t in args.types}
    alloc = allocate(args.n, mix)
    pools = build_pools(nodes, xrefs, graph, exclude=used, profile=profile)
    sample, shortfall = stratified_sample(
        pools, alloc, seed=args.seed, order=profile.sampling_order
    )
    emit(
        "plan: "
        + ", ".join(f"{t}={alloc[t]}" for t in mix)
        + "   pool sizes: "
        + ", ".join(f"{t}={len(pools[t])}" for t in mix)
    )
    for t, k in shortfall.items():
        emit(f"WARNING: only {alloc[t] - k} of {alloc[t]} {t} targets available")

    drafts, failures = draft_questions(
        llm, sample, titles, next_draft_number(existing), args.seed, base_url, profile=profile
    )
    save_drafts(out_path, existing + drafts)
    by_type = Counter(d["type"] for d in drafts)
    by_doc = Counter(d["gold_node_ids"][0].split(":", 1)[0] for d in drafts)
    flagged = [d for d in drafts if d["overlap_flag"]]
    emit(f"drafted {len(drafts)} question(s): " + ", ".join(f"{t}={n}" for t, n in by_type.items()))
    emit("by document: " + ", ".join(f"{d}={n}" for d, n in sorted(by_doc.items())))
    emit(f"copy-like (lexical overlap > 0.6): {len(flagged)}; failed drafts: {len(failures)}")
    for f in failures:
        emit(f"  failed {f['type']} {f['node_ids']}: {f['error']}")
    calls = getattr(llm, "calls", [])
    if calls and isinstance(calls[0], dict):
        cached = sum(1 for c in calls if c.get("cached"))
        ms = sum(float(c.get("latency_ms", 0.0)) for c in calls)
        emit(f"LLM calls: {len(calls)} ({cached} from cache), {ms / 1000:.1f} s total")
    for d in drafts:
        flag = "  [copy-like]" if d["overlap_flag"] else ""
        emit(f"  {d['draft_id']} {d['type']:15s} {', '.join(d['gold_node_ids'])}{flag}")
        emit(f"       Q: {d['question']}")
        emit(f"       A: {d['gold_answer']}")
    emit(f"appended to {out_path}. Review with: python scripts/review_drafts.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
