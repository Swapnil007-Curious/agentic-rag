"""The steps of a collection's end-to-end pipeline, their completion checks, time estimates
and the final SUMMARY.md (used by scripts/run_collection_pipeline.py).

Order: ingest -> chunk -> build_index -> build_auto_eval -> validate_eval (stops on failure) ->
ollama stop -> retrieval dev grid -> select_configs -> eval_retrieval test -> run_generation dev
-> derive_systems dev -> eval_generation test (which also derives G3 on test) -> summary.

Time estimates use the corpus size (pages kept, chunk counts and tokens once known) and speeds
measured on this machine and recorded in the repository's own logs: embedding times in the
consumer_law index report, LLM call latencies per phase in its eval_construction.json, reranking
time per question in its retrieval summaries and end-to-end latency per question in its
generation records. When a log is missing, the values measured when this runner was written are
used instead (listed in DEFAULTS).
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragsys.collections import Collection, get_collection
from ragsys.config import REPO_ROOT, load_chunking_config, load_ingest_config, resolve_path
from ragsys.eval.schema import AUTO_ORIGIN
from ragsys.pipeline_runner import Step, combine, file_digest, read_state, stat_digest
from ragsys.profiles import get_profile
from ragsys.utils.memory import fmt_duration

# Speeds measured on the development laptop (i3-1115G4, 4 threads, qwen2.5:3b via Ollama),
# used only when the repository's logs are missing.
DEFAULTS: dict[str, float] = {
    "ingest_s_per_page": 0.03,
    "chunk_s_per_page": 0.01,
    "tokens_per_page": 440.0,
    "emb_overhead_s_per_chunk:bge-base": 0.35,
    "emb_s_per_token:bge-base": 0.00179,
    "emb_overhead_s_per_chunk:bge-small": 0.10,
    "emb_s_per_token:bge-small": 0.00050,
    "llm_s:draft": 10.0,
    "llm_s:paraphrase": 7.0,
    "llm_s:support": 6.1,
    "llm_s:answer_from_source": 8.3,
    "llm_s:out_of_scope": 7.5,
    "llm_s:perturbation": 6.0,
    "llm_s:retrieval_check": 20.6,
    "rerank_s_per_question": 1.8,
    "gen_s_per_question:G1": 25.0,
    "gen_s_per_question:G2": 33.0,
    "gen_s_per_question:G4": 58.0,
    "nli_s_per_answer": 1.5,
    "model_load_s": 30.0,
}


def _py(script: str, *args: str) -> list[str]:
    return [sys.executable, str(REPO_ROOT / "scripts" / script), *args]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def has_auto_items(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(line).get("origin") == AUTO_ORIGIN:
                return True
        except (json.JSONDecodeError, AttributeError):
            continue
    return False


# --------------------------------------------------------------------------- steps


@dataclass
class PlanContext:
    coll: Collection
    collections_file: str
    chunkers: list[str]
    grid_size: int
    raw_files: list[Path]
    llm_model: str

    def a(self, p: Path) -> Path:
        return resolve_path(p)


def plan_context(coll: Collection, collections_file: str) -> PlanContext:
    from ragsys.chunking.pipeline import chunker_names
    from ragsys.retrieval.config import load_retrieval_settings

    ing = load_ingest_config(resolve_path(coll.ingest_config))
    ch = load_chunking_config(resolve_path(coll.chunking_config))
    rs = load_retrieval_settings(resolve_path(coll.retrieval_config))
    raw = resolve_path(ing.paths.raw_dir)
    return PlanContext(
        coll=coll,
        collections_file=collections_file,
        chunkers=chunker_names(ch),
        grid_size=len(rs.grid_configs()),
        raw_files=[raw / d.file for d in ing.enabled_documents()],
        llm_model=str(coll.llm().model),
    )


def build_steps(ctx: PlanContext) -> list[Step]:
    coll = ctx.coll
    a = ctx.a
    c_args = ["--collection", coll.name, "--collections-file", ctx.collections_file]
    pipe = coll.spec.pipeline
    tables = a(coll.tables_dir)
    questions = a(coll.questions_file)
    chunk_files = [a(coll.chunks_dir) / f"{c}.jsonl" for c in ctx.chunkers]
    selected = a(coll.selected_configs_file)
    test_log = a(coll.test_access_log)

    def fp_ingest() -> str:
        return combine(file_digest(a(coll.ingest_config)), *[stat_digest(f) for f in ctx.raw_files])

    def fp_chunks() -> str:
        return combine(
            file_digest(a(coll.chunking_config)),
            file_digest(a(coll.nodes_file)),
            file_digest(a(coll.structure_dir) / "xrefs.jsonl"),
        )

    def fp_index() -> str:
        return combine(
            file_digest(a(coll.retrieval_config)),
            *pipe.index_models,
            *[file_digest(f) for f in chunk_files],
        )

    def index_done() -> bool:
        from ragsys.retrieval.config import load_retrieval_settings
        from ragsys.retrieval.corpus import load_chunks
        from ragsys.retrieval.embed import model_slug

        rs = load_retrieval_settings(a(coll.retrieval_config))
        for chunker in ctx.chunkers:
            try:
                corpus = load_chunks(a(coll.processed_dir), chunker)
            except FileNotFoundError:
                return False
            for model in pipe.index_models:
                side = a(coll.emb_dir) / model_slug(rs.dense_model_name(model)) / f"{chunker}.json"
                meta = _read_json(side)
                if meta is None or meta.get("texts_sha256") != corpus.texts_hash():
                    return False
        return True

    def fp_eval_inputs() -> str:
        return combine(file_digest(questions), file_digest(a(coll.retrieval_config)), fp_index())

    def grid_done() -> bool:
        meta = _read_json(tables / "retrieval_dev.meta.json")
        return bool(meta) and meta.get("n_configs") == ctx.grid_size  # type: ignore[union-attr]

    def retrieval_test_prepare(argv: list[str]) -> list[str]:
        from ragsys.eval.split import read_access_log

        logged = [
            e for e in read_access_log(test_log) if str(e.get("note", "")).startswith("selection:")
        ]
        return [*argv, "--allow-repeat"] if logged else argv

    def fp_generation() -> str:
        return combine(
            file_digest(questions),
            file_digest(selected),
            file_digest(a(coll.generation_config)),
            file_digest(a(coll.retrieval_config)),
        )

    def gen_dev_done(systems: list[str]) -> bool:
        rows = {r.get("system") for r in _csv_rows(tables / "generation_dev.csv")}
        return all(s in rows for s in systems)

    from ragsys.agents.config import load_generation_settings

    gen = load_generation_settings(a(coll.generation_config))
    dev_systems = [s for s in pipe.dev_systems if not gen.is_derived(s)]
    derived_dev = [
        s for s in gen.systems if gen.is_derived(s) and gen.runnable_base(s) in dev_systems
    ]
    complete_marker = a(coll.generation_results_dir) / "test" / "_complete.json"

    def gen_test_prepare(argv: list[str]) -> list[str]:
        return [*argv, "--allow-repeat"] if complete_marker.exists() else argv

    steps = [
        Step(
            "ingest",
            _py("ingest.py", *c_args),
            done=lambda: (
                a(coll.nodes_file).exists()
                and (a(coll.structure_dir) / "structure_report.json").exists()
            ),
            fingerprint=fp_ingest,
        ),
        Step(
            "chunk",
            _py("chunk.py", "--chunker", "all", *c_args),
            done=lambda: all(f.exists() for f in chunk_files),
            fingerprint=fp_chunks,
        ),
        Step(
            "build_index",
            _py(
                "build_index.py",
                "--chunker",
                "all",
                "--models",
                *pipe.index_models,
                *([] if pipe.chroma else ["--no-chroma"]),
                *c_args,
            ),
            done=index_done,
            fingerprint=fp_index,
        ),
        Step(
            "build_auto_eval",
            _py("build_auto_eval.py", *c_args),
            done=lambda: has_auto_items(questions) and a(coll.eval_construction_file).exists(),
            accept_codes=(0, 3),
            note="exit 3 = the evaluation set already holds automatic items",
        ),
        Step("validate_eval", _py("validate_eval.py", *c_args), always=True),
        Step(
            "ollama_stop",
            ["ollama", "stop", ctx.llm_model],
            always=True,
            fatal=False,
            note="frees the LLM's memory before the retrieval grid",
        ),
        Step(
            "retrieval_dev_grid",
            _py("run_retrieval_grid.py", "--split", "dev", *c_args),
            done=grid_done,
            fingerprint=fp_eval_inputs,
        ),
        Step(
            "select_configs",
            _py("select_configs.py", *c_args),
            done=selected.exists,
            fingerprint=lambda: file_digest(tables / "retrieval_dev.csv"),
        ),
        Step(
            "eval_retrieval_test",
            _py("eval_retrieval.py", "--split", "test", *c_args),
            done=lambda: (tables / "retrieval_test.csv").exists() and test_log.exists(),
            fingerprint=lambda: file_digest(selected),
            prepare=retrieval_test_prepare,
        ),
        Step(
            "generation_dev",
            _py("run_generation.py", "--split", "dev", "--systems", *dev_systems, *c_args),
            done=lambda: gen_dev_done(dev_systems),
            fingerprint=fp_generation,
        ),
        Step(
            "derive_dev",
            _py("derive_systems.py", "--split", "dev", "--systems", *derived_dev, *c_args)
            if derived_dev
            else _py("derive_systems.py", "--split", "dev", *c_args),
            done=lambda: gen_dev_done([*dev_systems, *derived_dev]),
            fingerprint=lambda: combine(
                fp_generation(),
                *[
                    stat_digest(a(coll.generation_results_dir) / "dev" / s / "per_question.jsonl")
                    for s in dev_systems
                ],
            ),
        ),
        Step(
            "generation_test",
            _py("eval_generation.py", "--split", "test", "--systems", *pipe.test_systems, *c_args),
            done=complete_marker.exists,
            fingerprint=fp_generation,
            accept_codes=(0, 3),
            prepare=gen_test_prepare,
            note="also derives G3 from G2 on the test split",
        ),
        Step(
            "summary",
            _py("run_collection_pipeline.py", "--summary-only", *c_args),
            always=True,
        ),
    ]
    if not derived_dev:
        steps = [s for s in steps if s.name != "derive_dev"]
    return steps


# --------------------------------------------------------------------------- estimates


def _measured(default_coll: Collection | None) -> tuple[dict[str, float], list[str]]:
    """Speeds from the repository's logs (the default collection's), else DEFAULTS."""
    m = dict(DEFAULTS)
    used: list[str] = []
    if default_coll is None:
        return m, ["built-in defaults"]
    a = resolve_path
    idx = _read_json(a(default_coll.emb_dir) / "index_report.json")
    crep = _read_json(a(default_coll.chunks_dir) / "chunk_report.json")
    if idx and crep:
        pts: dict[str, list[tuple[float, float]]] = {}
        for chunker, entry in idx.get("chunkers", {}).items():
            st = crep.get("chunkers", {}).get(chunker)
            n = entry.get("chunks") or 0
            if not st or not n:
                continue
            mean_tok = float(st["tokens"].get("mean") or 0)
            for model, e in entry.get("embeddings", {}).items():
                if e.get("rebuilt") and e.get("seconds"):
                    pts.setdefault(model, []).append((mean_tok, float(e["seconds"]) / n))
        for model, ps in pts.items():
            if len(ps) >= 2:
                ps.sort()
                (x0, y0), (x1, y1) = ps[0], ps[-1]
                b = (
                    max((y1 - y0) / (x1 - x0), 1e-5)
                    if x1 > x0
                    else DEFAULTS.get(f"emb_s_per_token:{model}", 0.001)
                )
                m[f"emb_s_per_token:{model}"] = b
                m[f"emb_overhead_s_per_chunk:{model}"] = max(y0 - b * x0, 0.02)
        used.append(f"embedding times: {default_coll.emb_dir.as_posix()}/index_report.json")
    ec = _read_json(a(default_coll.eval_construction_file))
    if ec:
        for phase, v in ec.get("llm_calls", {}).get("by_phase", {}).items():
            if v.get("calls") and v.get("latency_s"):
                m[f"llm_s:{phase}"] = float(v["latency_s"]) / float(v["calls"])
        used.append(f"LLM latency per phase: {default_coll.eval_construction_file.as_posix()}")
    gdir = a(default_coll.generation_results_dir)
    for system in ("G1", "G2", "G4"):
        vals: list[float] = []
        for split in ("dev", "test"):
            p = gdir / split / system / "per_question.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    try:
                        cost = json.loads(line).get("cost", {})
                    except json.JSONDecodeError:
                        continue
                    if cost.get("e2e_ms"):
                        vals.append(float(cost["e2e_ms"]) / 1000.0)
        if vals:
            m[f"gen_s_per_question:{system}"] = statistics.fmean(vals)
    if (gdir / "dev").exists():
        used.append(
            f"generation latency per question: {default_coll.generation_results_dir.as_posix()}"
        )
    rr: list[float] = []
    rdir = a(default_coll.retrieval_results_dir) / "dev"
    if rdir.exists():
        for sp in rdir.glob("*rerank-minilm/summary.json"):
            s = _read_json(sp) or {}
            n = (s.get("counts") or {}).get("n_questions") or s.get("n_questions")
            if s.get("wall_time_s") and n:
                rr.append(float(s["wall_time_s"]) / float(n))
        if rr:
            m["rerank_s_per_question"] = max(rr)
            used.append(
                f"reranking time per question: {default_coll.retrieval_results_dir.as_posix()}"
            )
    return m, used or ["built-in defaults"]


def corpus_size(coll: Collection) -> dict[str, Any]:
    """Pages kept and chunk statistics of the collection (estimated when not built yet)."""
    a = resolve_path
    rep = _read_json(a(coll.structure_dir) / "structure_report.json")
    pages = 0
    if rep:
        pages = sum(
            int(d.get("pages_kept") or d.get("pages") or 0) for d in rep["documents"].values()
        )
    else:
        from ragsys.ingest.chapters import resolve_chapters

        ing = load_ingest_config(a(coll.ingest_config))
        for d in ing.enabled_documents():
            path = a(ing.paths.raw_dir) / d.file
            if path.exists() and d.chapters:
                pages += sum(len(r.pages) for r in resolve_chapters(path, d.chapters))
    crep = _read_json(a(coll.chunks_dir) / "chunk_report.json")
    chunks: dict[str, tuple[int, float]] = {}
    if crep:
        for name, st in crep.get("chunkers", {}).items():
            chunks[name] = (int(st["count"]), float(st["tokens"].get("mean") or 0))
    return {"pages": pages, "chunks": chunks}


def estimate_steps(ctx: PlanContext) -> tuple[dict[str, float], list[str], dict[str, Any]]:
    coll = ctx.coll
    try:
        default_coll: Collection | None = get_collection(None, ctx.collections_file)
    except KeyError:
        default_coll = None
    m, basis = _measured(default_coll)
    size = corpus_size(coll)
    pages = max(size["pages"], 1)
    tokens = pages * m["tokens_per_page"]
    chunks = dict(size["chunks"])
    for name in ctx.chunkers:  # estimated before chunking
        if name not in chunks:
            if name.startswith("fixed_"):
                n_tok = int(name.split("_")[1])
                chunks[name] = (int(tokens / max(n_tok - n_tok // 8, 1)) + 1, float(n_tok))
            else:
                chunks[name] = (int(tokens / 220) + 1, 220.0)
    est: dict[str, float] = {}
    est["ingest"] = pages * m["ingest_s_per_page"] + 5
    est["chunk"] = pages * m["chunk_s_per_page"] + 15
    emb = 0.0
    for model in coll.spec.pipeline.index_models:
        ov = m.get(f"emb_overhead_s_per_chunk:{model}", 0.35)
        per_tok = m.get(f"emb_s_per_token:{model}", 0.0018)
        for n, mean_tok in chunks.values():
            emb += n * (ov + per_tok * mean_tok)
        emb += m["model_load_s"]
    est["build_index"] = emb
    ae = coll.spec.auto_eval
    n, nu = ae.n_candidates, ae.n_unanswerable_candidates
    est["build_auto_eval"] = (
        n * m["llm_s:draft"]
        + 0.35 * n * m["llm_s:paraphrase"]
        + 0.8 * n * m["llm_s:support"]
        + 0.6 * n * m["llm_s:answer_from_source"]
        + (nu / 2 / 3) * m["llm_s:out_of_scope"]
        + (nu / 2) * m["llm_s:perturbation"]
        + 0.8 * nu * m["llm_s:retrieval_check"]
        + 2 * m["model_load_s"]
    )
    est["validate_eval"] = 5
    est["ollama_stop"] = 3
    total_q = ae.target_answerable + ae.target_unanswerable
    n_dev, n_test = round(0.3 * total_q), total_q - round(0.3 * total_q)
    rer = m["rerank_s_per_question"]
    est["retrieval_dev_grid"] = n_dev * rer * 2 * len(ctx.chunkers) + 4 * m["model_load_s"]
    est["select_configs"] = 3
    est["eval_retrieval_test"] = n_test * rer * 3 + 3 * m["model_load_s"]
    g1, g2, g4 = (m[f"gen_s_per_question:{s}"] for s in ("G1", "G2", "G4"))
    # G4's first synthesis prompt equals G2's (same retrieval and context): a cache hit
    per_q = {"G1": g1, "G2": g2, "G4": max(g4 - g2, 0.3 * g4)}
    dev = [s for s in coll.spec.pipeline.dev_systems if s in per_q]
    test = [s for s in coll.spec.pipeline.test_systems if s in per_q]
    est["generation_dev"] = (
        n_dev * sum(per_q[s] for s in dev)
        + n_dev * len(dev) * m["nli_s_per_answer"]
        + 2 * m["model_load_s"]
    )
    est["derive_dev"] = 60
    est["generation_test"] = (
        n_test * sum(per_q[s] for s in test)
        + n_test * (len(test) + 1) * m["nli_s_per_answer"]
        + 2 * m["model_load_s"]
    )
    est["summary"] = 5
    info = {"pages": pages, "chunks": chunks, "n_dev": n_dev, "n_test": n_test}
    return est, basis, info


# --------------------------------------------------------------------------- summary


def _fmt(v: str | None, nd: int = 3) -> str:
    if v in (None, ""):
        return "n/a"
    try:
        return f"{float(v):.{nd}f}"
    except ValueError:
        return str(v)


def _table(rows: list[list[str]], header: list[str]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def write_summary(coll: Collection) -> Path:
    a = resolve_path
    tables = a(coll.tables_dir)
    lines = [
        f"# {coll.title}: pipeline summary",
        "",
        f"Collection `{coll.name}` (profile `{coll.profile}`), written "
        f"{_dt.datetime.now(_dt.UTC).isoformat(timespec='seconds')} by "
        "scripts/run_collection_pipeline.py from the tables in this folder.",
        "",
    ]
    size = corpus_size(coll)
    lines += ["## Corpus", "", f"Pages kept: {size['pages']}", ""]
    if size["chunks"]:
        lines += _table(
            [[k, str(n), f"{t:.0f}"] for k, (n, t) in size["chunks"].items()],
            ["chunker", "chunks", "mean tokens"],
        )
        lines.append("")
    ec = _read_json(tables / "eval_construction.json")
    if ec:
        fc = ec.get("final_counts", {}).get("whole_file", {})
        lines += [
            "## Evaluation set",
            "",
            f"{fc.get('answerable', 'n/a')} answerable + {fc.get('unanswerable', 'n/a')} "
            f"unanswerable questions (model {ec.get('model')}, automatically filtered).",
            "",
            "By type: " + ", ".join(f"{k}={v}" for k, v in (fc.get("by_type") or {}).items()),
            "",
            "By split: "
            + ", ".join(
                f"{sp} {v['answerable']}+{v['unanswerable']}"
                for sp, v in (fc.get("by_split") or {}).items()
            ),
            "",
        ]
        xr = ec.get("cross_reference_decision")
        if xr:
            lines += [f"cross_reference questions: {xr.get('reason')}", ""]
        for note in ec.get("notes", []):
            lines += [f"Note: {note}", ""]
    dev_rows = _csv_rows(tables / "retrieval_dev.csv")
    if dev_rows:
        dev_rows.sort(key=lambda r: -float(r.get("coverage@1024") or -1))
        lines += ["## Retrieval (dev grid)", ""]
        lines += _table(
            [
                [
                    r["config_id"],
                    _fmt(r.get("recall@5")),
                    _fmt(r.get("mrr@10")),
                    _fmt(r.get("ndcg@10")),
                    _fmt(r.get("coverage@1024")),
                    _fmt(r.get("latency_total_p50_ms"), 1),
                ]
                for r in dev_rows
            ],
            ["config", "R@5", "MRR@10", "nDCG@10", "cov@1024", "p50 ms"],
        )
        lines.append("")
    sel = _read_json(tables / "selected_configs.json")
    if sel:
        lines += ["Selected for test (" + str(sel.get("rule", "")) + "):", ""]
        lines += [f"- `{s['config_id']}`: {', '.join(s['roles'])}" for s in sel.get("selected", [])]
        lines.append("")
    test_rows = _csv_rows(tables / "retrieval_test.csv")
    if test_rows:
        lines += ["## Retrieval (test)", ""]
        lines += _table(
            [
                [
                    r["config_id"],
                    _fmt(r.get("recall@5")),
                    _fmt(r.get("mrr@10")),
                    _fmt(r.get("coverage@1024")),
                ]
                for r in test_rows
            ],
            ["config", "R@5", "MRR@10", "cov@1024"],
        )
        lines.append("")
        boot = _csv_rows(tables / "retrieval_test_bootstrap.csv")
        if boot:
            lines += ["Paired bootstrap vs the naive baseline:", ""]
            lines += _table(
                [
                    [
                        b["config_id"],
                        b["metric"],
                        _fmt(b["mean_diff"]),
                        f"[{_fmt(b['ci_low'])}, {_fmt(b['ci_high'])}]",
                    ]
                    for b in boot
                ],
                ["config", "metric", "diff", "95% CI"],
            )
            lines.append("")
    gen_cols = [
        ("overall_accuracy", "accuracy"),
        ("faithfulness", "faithful"),
        ("correct_rate_answerable", "correct (ans.)"),
        ("citation_precision", "cit. prec."),
        ("correct_refusal_rate", "correct refusal"),
        ("false_abstention_rate", "false abst."),
        ("latency_p50_ms", "p50 ms"),
    ]
    for split in ("dev", "test"):
        rows = _csv_rows(tables / f"generation_{split}.csv")
        if not rows:
            continue
        lines += [f"## Generation ({split})", ""]
        lines += _table(
            [
                [
                    r["system"],
                    *[_fmt(r.get(k), 0 if k.endswith("_ms") else 3) for k, _l in gen_cols],
                ]
                for r in rows
            ],
            ["system", *[lbl for _k, lbl in gen_cols]],
        )
        lines.append("")
        boot = _csv_rows(tables / f"generation_{split}_bootstrap.csv")
        if boot and split == "test":
            keep = [b for b in boot if b.get("metric") in ("overall_accuracy", "faithfulness")]
            if keep:
                lines += ["Paired bootstrap vs G1:", ""]
                lines += _table(
                    [
                        [
                            b["system"],
                            b["metric"],
                            b.get("scope", ""),
                            _fmt(b["mean_diff"]),
                            f"[{_fmt(b['ci_low'])}, {_fmt(b['ci_high'])}]",
                        ]
                        for b in keep
                    ],
                    ["system", "metric", "scope", "diff", "95% CI"],
                )
                lines.append("")
    gen = _read_json(tables / "generation_test.meta.json") or _read_json(
        tables / "generation_dev.meta.json"
    )
    skipped = ((gen or {}).get("settings") or {}).get("skipped_systems") or {}
    if skipped:
        lines += ["Systems not run for this collection:", ""]
        lines += [f"- {k}: {v}" for k, v in skipped.items()]
        lines.append("")
    state = read_state(a(coll.pipeline_state))
    if state["steps"]:
        lines += ["## Pipeline steps", ""]
        lines += _table(
            [
                [
                    name,
                    fmt_duration(float(v.get("seconds", 0))),
                    f"{v.get('peak_rss_mb', 0):,}",
                    str(v.get("completed_utc", "")),
                ]
                for name, v in state["steps"].items()
            ],
            ["step", "time", "peak RSS MB", "completed (UTC)"],
        )
        lines.append("")
    out = tables / "SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


def profile_targets(coll: Collection) -> tuple[int, int]:
    return get_profile(coll.profile).targets
