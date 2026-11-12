"""Steps of scripts/run_app_collections.py: the app indexes and the generator-scale ablation.

Order:

1. for every collection that is not evaluated (indian_law, human_biology, physics_chemistry):
   ingest -> chunk -> build_index (structural chunks, bge-small, no Chroma);
2. the cs_ai app index (evaluation documents + Python Tutorial + Think Python) into
   ``collections/cs_ai/processed_app``: ingest -> chunk -> build_index (bge-small); the frozen
   evaluation corpus in ``collections/cs_ai/processed`` is never touched;
3. the hosted generator-scale ablation on the TEST split: ``eval_generation.py --systems G2H
   G4H`` for consumer_law and cs_ai (rows appended to their generation_test tables);
4. SUMMARY.md of consumer_law and cs_ai, rewritten with the ablation section.

Every step is skipped when complete with unchanged inputs and resumes from its own caches
(embedding batches, the LLM response cache, per-question records), so after a crash, a reboot
or an exhausted API quota the same command continues.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragsys.collections import Collection, all_collections, get_collection
from ragsys.config import REPO_ROOT, load_chunking_config, load_ingest_config, resolve_path
from ragsys.llm.client import EXIT_QUOTA
from ragsys.pipeline_plan import DEFAULTS, _measured, _read_json
from ragsys.pipeline_runner import Step, combine, file_digest, stat_digest

STATE_FILE = Path("collections/app_pipeline_state.json")
LOG_FILE = Path("collections/app_pipeline_log.txt")
# --wait-for-quota: longest Retry-After waited out inside a call, pause before restarting a step
# stopped by the quota, and how long the runner keeps trying
QUOTA_RETRY_WAIT_S = 21600
QUOTA_RESTART_S = 1800
QUOTA_MAX_DAYS = 5
# API tokens per call of the hosted generator, measured on 2026-11-12 (openai/gpt-oss-120b,
# low reasoning effort): a synthesis prompt is its passages (bge tokens ~ API tokens) plus about
# 800 tokens of instructions, with a short completion; a critic call is about 500 tokens
SYNTH_OVERHEAD_TOKENS = 800
SYNTH_COMPLETION_TOKENS = 150
CRITIC_TOKENS = 650
REVISION_OVERHEAD_TOKENS = 1000
DEFAULT_REVISION_SHARE = 0.35


def _py(script: str, *args: str) -> list[str]:
    return [sys.executable, str(REPO_ROOT / "scripts" / script), *args]


@dataclass
class IndexTarget:
    """One index to build: a collection's only index, or an evaluated collection's app index."""

    name: str  # step prefix, e.g. "indian_law" or "cs_ai-app"
    coll: Collection
    ingest: Path
    chunking: Path
    retrieval: Path
    processed: Path
    models: list[str]

    @property
    def config_args(self) -> list[str]:
        return ["--collection", self.coll.name]


def index_targets(collections_file: str) -> list[IndexTarget]:
    out: list[IndexTarget] = []
    for c in all_collections(collections_file):
        if c.evaluated and not c.has_app_index:
            continue
        out.append(
            IndexTarget(
                name=f"{c.name}-app" if c.evaluated else c.name,
                coll=c,
                ingest=c.app_ingest_config,
                chunking=c.app_chunking_config,
                retrieval=c.app_retrieval_config,
                processed=c.app_processed_dir,
                models=c.app_index_models,
            )
        )
    # app-only collections first, then app indexes of evaluated collections
    return sorted(out, key=lambda t: t.coll.evaluated)


def ablation_collections(collections_file: str) -> list[Collection]:
    return [
        c
        for c in all_collections(collections_file)
        if c.evaluated and c.spec.pipeline.ablation_systems
    ]


def _chunkers(t: IndexTarget) -> list[str]:
    from ragsys.chunking.pipeline import chunker_names

    return chunker_names(load_chunking_config(resolve_path(t.chunking)))


def index_steps(t: IndexTarget, collections_file: str) -> list[Step]:
    a = resolve_path
    c_args = ["--collection", t.coll.name, "--collections-file", collections_file]
    ing = load_ingest_config(a(t.ingest))
    raw = a(ing.paths.raw_dir)
    raw_files = [raw / d.file for d in ing.enabled_documents()]
    nodes = a(t.processed) / "structure" / "nodes.jsonl"
    chunkers = _chunkers(t)
    chunk_files = [a(t.processed) / "chunks" / f"{c}.jsonl" for c in chunkers]
    explicit = t.coll.has_app_index  # the app configs are not the collection's defaults

    def fp_ingest() -> str:
        return combine(file_digest(a(t.ingest)), *[stat_digest(f) for f in raw_files])

    def fp_chunks() -> str:
        return combine(file_digest(a(t.chunking)), file_digest(nodes))

    def fp_index() -> str:
        return combine(
            file_digest(a(t.retrieval)), *t.models, *[file_digest(f) for f in chunk_files]
        )

    def index_done() -> bool:
        from ragsys.retrieval.config import load_retrieval_settings
        from ragsys.retrieval.corpus import load_chunks
        from ragsys.retrieval.embed import model_slug

        rs = load_retrieval_settings(a(t.retrieval))
        for chunker in chunkers:
            try:
                corpus = load_chunks(a(t.processed), chunker)
            except FileNotFoundError:
                return False
            for model in t.models:
                side = (
                    a(rs.paths.emb_dir) / model_slug(rs.dense_model_name(model)) / f"{chunker}.json"
                )
                meta = _read_json(side)
                if meta is None or meta.get("texts_sha256") != corpus.texts_hash():
                    return False
        return True

    return [
        Step(
            f"{t.name}:ingest",
            _py("ingest.py", *c_args, *(["--config", str(t.ingest)] if explicit else [])),
            done=lambda: nodes.exists() and (nodes.parent / "structure_report.json").exists(),
            fingerprint=fp_ingest,
        ),
        Step(
            f"{t.name}:chunk",
            _py(
                "chunk.py",
                "--chunker",
                "all",
                *c_args,
                *(["--config", str(t.chunking)] if explicit else []),
            ),
            done=lambda: all(f.exists() for f in chunk_files),
            fingerprint=fp_chunks,
        ),
        Step(
            f"{t.name}:index",
            _py(
                "build_index.py",
                "--chunker",
                "all",
                "--models",
                *t.models,
                "--no-chroma",
                *c_args,
                *(
                    ["--config", str(t.retrieval), "--chunking-config", str(t.chunking)]
                    if explicit
                    else []
                ),
            ),
            done=index_done,
            fingerprint=fp_index,
        ),
    ]


def ablation_steps(
    coll: Collection, collections_file: str, *, wait_for_quota: bool = False
) -> list[Step]:
    """The ablation step of one collection. With ``wait_for_quota`` the evaluation waits out
    long Retry-After answers itself (up to QUOTA_RETRY_WAIT_S per call) and, should it still
    stop on the quota (exit code 75), the runner starts it again every QUOTA_RESTART_S for up
    to QUOTA_MAX_DAYS days; it resumes from its per-question records and the LLM cache."""
    systems = list(coll.spec.pipeline.ablation_systems)
    marker = resolve_path(coll.generation_results_dir) / "test" / "_complete_ablation.json"
    wait = ["--max-retry-wait", str(QUOTA_RETRY_WAIT_S)] if wait_for_quota else []
    return [
        Step(
            f"{coll.name}:ablation",
            _py(
                "eval_generation.py",
                "--split",
                "test",
                "--systems",
                *systems,
                "--collection",
                coll.name,
                "--collections-file",
                collections_file,
                *wait,
            ),
            done=marker.exists,
            accept_codes=(0, 3),
            note="exit 3 = the ablation already completed",
            retry_codes=(EXIT_QUOTA,) if wait_for_quota else (),
            retry_wait_s=float(QUOTA_RESTART_S),
            max_retries=int(QUOTA_MAX_DAYS * 86400 / QUOTA_RESTART_S) if wait_for_quota else 0,
        )
    ]


def summary_steps(colls: list[Collection], collections_file: str) -> list[Step]:
    return [
        Step(
            f"{c.name}:summary",
            _py(
                "run_collection_pipeline.py",
                "--summary-only",
                "--collection",
                c.name,
                "--collections-file",
                collections_file,
            ),
            always=True,
        )
        for c in colls
    ]


def build_app_steps(
    collections_file: str, *, ablation: bool = True, wait_for_quota: bool = False
) -> list[Step]:
    steps: list[Step] = []
    for t in index_targets(collections_file):
        steps += index_steps(t, collections_file)
    if ablation:
        colls = ablation_collections(collections_file)
        for c in colls:
            steps += ablation_steps(c, collections_file, wait_for_quota=wait_for_quota)
        steps += summary_steps(colls, collections_file)
    return steps


# --------------------------------------------------------------------------- estimates


def _pages_and_chunks(t: IndexTarget) -> tuple[int, dict[str, tuple[int, float]]]:
    a = resolve_path
    rep = _read_json(a(t.processed) / "structure" / "structure_report.json")
    pages = 0
    if rep:
        pages = sum(
            int(d.get("pages_kept") or d.get("pages") or 0) for d in rep["documents"].values()
        )
    crep = _read_json(a(t.processed) / "chunks" / "chunk_report.json")
    chunks: dict[str, tuple[int, float]] = {}
    if crep:
        for name, st in crep.get("chunkers", {}).items():
            chunks[name] = (int(st["count"]), float(st["tokens"].get("mean") or 0))
    if not pages:
        from ragsys.pipeline_plan import corpus_size

        pages = int(corpus_size(t.coll)["pages"]) if not t.coll.has_app_index else 0
        if not pages:  # cs_ai app: the evaluation documents' pages plus the two books
            pages = 900
    return pages, chunks


def revision_share(coll: Collection, mirrored: str = "G4") -> float:
    """Share of test questions that needed a revision round in the local run of the mirrored
    system (``DEFAULT_REVISION_SHARE`` when its records are not available)."""
    from ragsys.eval.generation import read_jsonl

    path = resolve_path(coll.generation_results_dir) / "test" / mirrored / "per_question.jsonl"
    recs = read_jsonl(path)
    if not recs:
        return DEFAULT_REVISION_SHARE
    revised = sum(
        1 for r in recs if sum(1 for st in r.get("steps", []) if st.get("name") == "synthesis") > 1
    )
    return revised / len(recs)


def ablation_tokens(coll: Collection) -> dict[str, Any]:
    """Expected API usage of the ablation: per-call tokens measured on the hosted model, the
    hosted context budgets and the share of revisions of the local G4 test run. G4H's first
    synthesis has the same prompt as G2H's and comes from the cache."""
    from ragsys.agents.config import load_generation_settings
    from ragsys.eval.split import split_of
    from ragsys.eval.validate import load_questions

    gen = load_generation_settings(resolve_path(coll.generation_config))
    systems = list(coll.spec.pipeline.ablation_systems)
    backend = gen.backend_of(systems[0])
    ctx = gen.context_for(backend)
    spec = gen.backends().spec(backend)
    questions, _err = load_questions(resolve_path(coll.questions_file))
    n_test = sum(1 for q in questions if split_of(q.id) == "test")
    p_rev = revision_share(coll)
    synth = ctx.budget_tokens + SYNTH_OVERHEAD_TOKENS + SYNTH_COMPLETION_TOKENS
    revision = ctx.revision_budget_tokens + REVISION_OVERHEAD_TOKENS + SYNTH_COMPLETION_TOKENS
    per_q = {"G2H": synth, "G4H": CRITIC_TOKENS + p_rev * (revision + CRITIC_TOKENS)}
    calls_q = {"G2H": 1.0, "G4H": 1.0 + 2.0 * p_rev}
    tokens = sum(per_q.get(s, synth) for s in systems) * n_test
    calls = sum(calls_q.get(s, 1.0) for s in systems) * n_test
    tpm = spec.tokens_per_minute or 1e9
    rpm = spec.requests_per_minute or 1e9
    minutes = max(tokens / tpm, calls / rpm)
    days = max(
        tokens / spec.tokens_per_day if spec.tokens_per_day else 0.0,
        calls / spec.requests_per_day if spec.requests_per_day else 0.0,
    )
    return {
        "n_test": n_test,
        "tokens": int(tokens),
        "calls": int(calls),
        "revision_share": round(p_rev, 3),
        "api_minutes": minutes,
        "quota_days": days,
        "tpm": spec.tokens_per_minute,
        "rpm": spec.requests_per_minute,
        "tpd": spec.tokens_per_day,
        "rpd": spec.requests_per_day,
        "model": spec.model,
    }


def ablation_wall_s(tok: dict[str, Any], quota_days_used: float = 0.0) -> float:
    """Wall time of an ablation step that starts after ``quota_days_used`` days of the daily
    quota were spent: the API time, or the time for the daily quota to renew, whichever is
    longer (the first day's quota is available at once)."""
    api = float(tok["api_minutes"]) * 60.0
    end = quota_days_used + float(tok["quota_days"])
    waiting_days = max(0.0, end - max(1.0, quota_days_used))
    return max(api, waiting_days * 86400.0)


def estimate_app_steps(
    steps: list[Step], collections_file: str
) -> tuple[dict[str, float], list[str], dict[str, Any]]:
    try:
        default_coll: Collection | None = get_collection(None, collections_file)
    except KeyError:
        default_coll = None
    m, basis = _measured(default_coll)
    est: dict[str, float] = {}
    info: dict[str, Any] = {"targets": {}, "ablation": {}}
    for t in index_targets(collections_file):
        pages, chunks = _pages_and_chunks(t)
        tokens = pages * m["tokens_per_page"]
        if not chunks:
            chunks = {c: (int(tokens / 220) + 1, 220.0) for c in _chunkers(t)}
        est[f"{t.name}:ingest"] = pages * 0.015 + 10
        est[f"{t.name}:chunk"] = pages * 0.01 + 15
        emb = 0.0
        for model in t.models:
            ov = m.get(
                f"emb_overhead_s_per_chunk:{model}",
                DEFAULTS.get(f"emb_overhead_s_per_chunk:{model}", 0.1),
            )
            per_tok = m.get(
                f"emb_s_per_token:{model}", DEFAULTS.get(f"emb_s_per_token:{model}", 0.0005)
            )
            for n, mean_tok in chunks.values():
                emb += n * (ov + per_tok * mean_tok)
            emb += m["model_load_s"]
        est[f"{t.name}:index"] = emb
        info["targets"][t.name] = {"pages": pages, "chunks": chunks}
    used = 0.0  # days of the daily quota spent by earlier ablation steps
    for c in ablation_collections(collections_file):
        tok = ablation_tokens(c)
        nli = tok["n_test"] * len(c.spec.pipeline.ablation_systems) * m["nli_s_per_answer"]
        wall = ablation_wall_s(tok, used)
        used += float(tok["quota_days"])
        api = float(tok["api_minutes"]) * 60.0
        est[f"{c.name}:ablation"] = max(wall, api * 1.15) + nli + 2 * m["model_load_s"]
        est[f"{c.name}:summary"] = 5
        info["ablation"][c.name] = tok
    info["quota_days"] = used
    return est, basis, info
