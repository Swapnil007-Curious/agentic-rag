"""Retrieval evaluation harness: per-question records, per-config summaries, resumable runs
over many configs, CSV tables and paired bootstrap comparisons.

Questions are always selected through :func:`ragsys.eval.split.questions_for_split`, so the
test split cannot be evaluated without a grant from ``scripts/eval_retrieval.py``.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from ragsys.eval.metrics import (
    auroc,
    budget_coverage,
    hit_at_k,
    mean,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    percentile,
    recall_at_k,
)
from ragsys.eval.relevance import RelevanceJudge
from ragsys.eval.schema import EvalQuestion
from ragsys.eval.split import Split, TestAccess, questions_for_split, split_of
from ragsys.retrieval.config import RetrievalConfig, RetrievalSettings, config_dir_name
from ragsys.retrieval.pipeline import STAGES, RetrievalResources, Retriever
from ragsys.utils.memory import fmt_duration, fmt_mb, rss_bytes
from ragsys.utils.runinfo import collect_run_info

SMOKE_LABEL = "smoke - not results"


def metric_names(ks: Sequence[int], budgets: Sequence[int]) -> list[str]:
    return (
        [f"recall@{k}" for k in ks]
        + [f"hit@{k}" for k in ks]
        + ["mrr@10", "ndcg@10"]
        + [f"coverage@{b}" for b in budgets]
    )


def questions_fingerprint(questions: Sequence[EvalQuestion]) -> str:
    h = hashlib.sha256()
    for q in sorted(questions, key=lambda x: x.id):
        h.update(q.model_dump_json().encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class Evaluator:
    """Evaluates retrievers built from one :class:`RetrievalResources`."""

    def __init__(self, settings: RetrievalSettings, resources: RetrievalResources) -> None:
        self.settings = settings
        self.resources = resources
        self.ks = list(settings.eval.ks)
        self.budgets = list(settings.eval.budgets)
        self._judges: dict[str, RelevanceJudge] = {}
        self._n_rel: dict[tuple[str, tuple[str, ...]], int] = {}

    def judge(self, chunker: str) -> RelevanceJudge:
        if chunker not in self._judges:
            self._judges[chunker] = RelevanceJudge(
                self.resources.graph(), self.resources.corpus(chunker)
            )
        return self._judges[chunker]

    def documents_of(self, q: EvalQuestion) -> list[str]:
        graph = self.resources.graph()
        docs = {graph.nodes[g].doc_id for g in q.gold_node_ids if g in graph}
        return sorted(docs) or ["(none)"]

    def evaluate_question(
        self, retriever: Retriever, q: EvalQuestion, split: str
    ) -> dict[str, Any]:
        res = retriever.search(q.question)
        judge = self.judge(retriever.cfg.chunker)
        golds = list(q.gold_node_ids) if q.answerable else []
        ranked_ids = [r.chunk_id for r in res.ranked]
        prof = judge.profile(ranked_ids, golds) if golds else [set() for _ in ranked_ids]
        toks = [r.chunk.n_tokens for r in res.ranked]
        metrics: dict[str, float] | None = None
        n_rel = 0
        if q.answerable:
            key = (retriever.cfg.chunker, tuple(golds))
            if key not in self._n_rel:
                self._n_rel[key] = judge.n_relevant_chunks(golds)
            n_rel = self._n_rel[key]
            metrics = {f"recall@{k}": recall_at_k(prof, golds, k) for k in self.ks}
            metrics.update({f"hit@{k}": hit_at_k(prof, k) for k in self.ks})
            metrics["mrr@10"] = mrr_at_k(prof, 10)
            metrics["ndcg@10"] = ndcg_at_k(prof, 10, n_rel)
            metrics.update(
                {f"coverage@{b}": budget_coverage(prof, toks, golds, b) for b in self.budgets}
            )
        return {
            "qid": q.id,
            "split": split,
            "question": q.question,
            "type": q.type,
            "origin": q.origin,
            "difficulty": q.difficulty,
            "answerable": q.answerable,
            "documents": self.documents_of(q),
            "gold_node_ids": golds,
            "n_relevant_chunks": n_rel,
            "metrics": metrics,
            "top1_first_stage_score": res.top1_first_stage_score,
            "top1_rerank_score": res.top1_rerank_score if retriever.cfg.rerank else None,
            "timings_ms": res.timings_ms,
            "n_expanded": len(res.expansions),
            "ranked": [
                {
                    "chunk_id": r.chunk_id,
                    "n_tokens": r.chunk.n_tokens,
                    "provenance": r.provenance,
                    "first_stage_rank": r.first_stage_rank,
                    "first_stage_score": r.first_stage_score,
                    "rerank_score": r.rerank_score,
                    "relevant_to": sorted(p),
                }
                for r, p in zip(res.ranked, prof, strict=True)
            ],
        }

    def summarise(
        self,
        cfg: RetrievalConfig,
        records: Sequence[dict[str, Any]],
        split: str,
        questions_sha: str,
        label: str,
    ) -> dict[str, Any]:
        names = metric_names(self.ks, self.budgets)
        ans = [r for r in records if r["answerable"]]
        unans = [r for r in records if not r["answerable"]]

        def avg(rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
            return {m: mean([r["metrics"][m] for r in rows if r["metrics"]]) for m in names}

        latency: dict[str, dict[str, float | None]] = {}
        for stage in STAGES:
            vals = [r["timings_ms"][stage] for r in records if stage in r["timings_ms"]]
            if vals:
                latency[stage] = {"p50": percentile(vals, 50), "p95": percentile(vals, 95)}

        def dist(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
            vals = [float(r[key]) for r in rows if r[key] is not None]
            return {
                "n": len(vals),
                "mean": mean(vals),
                "median": percentile(vals, 50),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
            }

        scores: dict[str, Any] = {}
        aurocs: dict[str, float | None] = {}
        for key in ("top1_first_stage_score", "top1_rerank_score"):
            if key == "top1_rerank_score" and not cfg.rerank:
                continue
            name = key.removesuffix("_score")
            scores[name] = {"answerable": dist(ans, key), "unanswerable": dist(unans, key)}
            aurocs[name] = auroc(
                [float(r[key]) for r in ans if r[key] is not None],
                [float(r[key]) for r in unans if r[key] is not None],
            )

        def breakdown(keyfn: Callable[[dict[str, Any]], Iterable[str]]) -> dict[str, Any]:
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in records:
                for g in keyfn(r):
                    groups[g].append(r)
            return {
                g: {
                    "n": len(rows),
                    "n_answerable": sum(1 for r in rows if r["answerable"]),
                    "metrics": avg([r for r in rows if r["answerable"]]),
                }
                for g, rows in sorted(groups.items())
            }

        return {
            "config_id": cfg.config_id(),
            "label": label,
            "split": split,
            "questions_sha256": questions_sha,
            "n_questions": len(records),
            "n_answerable": len(ans),
            "n_unanswerable": len(unans),
            "metrics": avg(ans),
            "latency_ms": latency,
            "top1_scores": scores,
            "auroc": aurocs,
            "breakdown": {
                "type": breakdown(lambda r: [r["type"]]),
                "document": breakdown(lambda r: r["documents"]),
                "origin": breakdown(lambda r: [r["origin"]]),
            },
            "config": cfg.model_dump(mode="json"),
            "models": {
                "dense": self.settings.dense_model_name(cfg.dense_model)
                if cfg.uses_dense
                else None,
                "reranker": self.settings.reranker_name(cfg.rerank_model) if cfg.rerank else None,
            },
            "settings": self.settings.model_dump(mode="json", exclude={"grid"}),
            "run_info": collect_run_info(),
        }

    def evaluate_config(
        self, cfg: RetrievalConfig, questions: Sequence[EvalQuestion], split: str, label: str = ""
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        retriever = Retriever(cfg, self.resources)
        retriever.warm_up()
        records = [self.evaluate_question(retriever, q, split) for q in questions]
        summary = self.summarise(cfg, records, split, questions_fingerprint(questions), label)
        return records, summary


# --------------------------------------------------------------------------- runs on disk


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def run_configs(
    evaluator: Evaluator,
    configs: Sequence[RetrievalConfig],
    all_questions: Sequence[EvalQuestion],
    split: Split,
    out_root: Path,
    *,
    access: TestAccess | None = None,
    overwrite: bool = False,
    label: str = "",
    log: Callable[[str], None] = print,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Evaluate ``configs`` on one split; per config writes ``<out_root>/<config_dir>/``
    ``per_query.jsonl`` and ``summary.json``. Existing outputs for the same question set are
    reused unless ``overwrite``. Returns summaries (in the order of ``configs``) and per-query
    records by config id.

    Configs are executed grouped by dense model (:func:`execution_groups`): configs without a
    dense encoder first, then one group per encoder, which is released (``del`` +
    ``gc.collect()``) before the next group starts, so at most one encoder is in memory. The
    process RSS is logged at group boundaries and an ETA after each evaluated config."""
    questions = questions_for_split(all_questions, split, access)
    sha = questions_fingerprint(questions)
    total = len(configs)
    reuse: dict[int, tuple[dict[str, Any], Path]] = {}
    status_of: dict[int, str] = {}
    for i, cfg in enumerate(configs):
        d = out_root / config_dir_name(cfg.config_id())
        sp, pq = d / "summary.json", d / "per_query.jsonl"
        status_of[i] = "ran"
        if not overwrite and sp.exists() and pq.exists():
            old = json.loads(sp.read_text(encoding="utf-8"))
            if old.get("questions_sha256") == sha and old.get("label", "") == label:
                reuse[i] = (old, pq)
            else:
                status_of[i] = "rerun (question set changed)"
    n_to_run = total - len(reuse)
    results: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    resources = evaluator.resources
    ran_seconds: list[float] = []
    pos = 0
    groups = execution_groups(configs, evaluator.settings)
    for model_name, idxs in groups:
        n_run_here = sum(1 for i in idxs if i not in reuse)
        what = f"dense encoder {model_name}" if model_name else "no dense encoder"
        log(
            f"== group: {what}: {len(idxs)} config(s), {n_run_here} to evaluate | "
            f"RSS {fmt_mb(rss_bytes())}"
        )
        for i in idxs:
            pos += 1
            cfg = configs[i]
            cid = cfg.config_id()
            if i in reuse:
                old, pq = reuse[i]
                results[i] = (old, _read_jsonl(pq))
                log(f"[{pos:3d}/{total}] skip (exists) {cid}")
                continue
            d = out_root / config_dir_name(cid)
            t0 = time.perf_counter()
            records, summary = evaluator.evaluate_config(cfg, questions, split, label)
            summary["wall_time_s"] = round(time.perf_counter() - t0, 2)
            _write_jsonl(d / "per_query.jsonl", records)
            (d / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            results[i] = (summary, records)
            ran_seconds.append(summary["wall_time_s"])
            left = n_to_run - len(ran_seconds)
            eta = fmt_duration(left * sum(ran_seconds) / len(ran_seconds))
            m = summary["metrics"]
            lat = summary["latency_ms"].get("total", {}).get("p50")
            log(
                f"[{pos:3d}/{total}] {status_of[i]:4s} {cid}  n={summary['n_questions']} "
                f"R@5={_fmt(m.get('recall@5'))} cov@1024={_fmt(m.get('coverage@1024'))} "
                f"p50={_fmt(lat, 1)}ms ({summary['wall_time_s']}s)  ETA {eta}"
            )
        if model_name is not None:
            released = resources.release_encoders()
            if released:
                log(f"   released {', '.join(released)} | RSS {fmt_mb(rss_bytes())}")
    summaries = [results[i][0] for i in range(total)]
    records_by = {configs[i].config_id(): results[i][1] for i in range(total)}
    return summaries, records_by


def execution_groups(
    configs: Sequence[RetrievalConfig], settings: RetrievalSettings
) -> list[tuple[str | None, list[int]]]:
    """Indices of ``configs`` grouped by dense model name: configs without a dense encoder
    (BM25 only) first, then one group per model in order of first appearance."""
    groups: dict[str | None, list[int]] = {None: []}
    for i, cfg in enumerate(configs):
        key = settings.dense_model_name(cfg.dense_model) if cfg.uses_dense else None
        groups.setdefault(key, []).append(i)
    return [(k, v) for k, v in groups.items() if v]


def load_existing(
    configs: Sequence[RetrievalConfig],
    all_questions: Sequence[EvalQuestion],
    split: Split,
    out_root: Path,
    *,
    access: TestAccess | None = None,
    label: str = "",
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Summaries/records already on disk for ``configs`` that match the current question set
    (used to rebuild tables over the whole grid after running a subset)."""
    sha = questions_fingerprint(questions_for_split(all_questions, split, access))
    summaries: list[dict[str, Any]] = []
    records_by: dict[str, list[dict[str, Any]]] = {}
    for cfg in configs:
        cid = cfg.config_id()
        d = out_root / config_dir_name(cid)
        sp, pq = d / "summary.json", d / "per_query.jsonl"
        if not (sp.exists() and pq.exists()):
            continue
        old = json.loads(sp.read_text(encoding="utf-8"))
        if old.get("questions_sha256") == sha and old.get("label", "") == label:
            summaries.append(old)
            records_by[cid] = _read_jsonl(pq)
    return summaries, records_by


def _fmt(v: Any, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- tables


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return v


def summary_row(s: dict[str, Any]) -> dict[str, Any]:
    cfg = s["config"]
    row: dict[str, Any] = {
        "config_id": s["config_id"],
        "label": s.get("label", ""),
        "chunker": cfg["chunker"],
        "retriever": cfg["retriever"],
        "dense_model": cfg["dense_model"] if cfg["retriever"] != "bm25" else "",
        "xref_expand": cfg["xref_expand"],
        "rerank": cfg["rerank_model"] if cfg["rerank"] else "",
        "n_questions": s["n_questions"],
        "n_answerable": s["n_answerable"],
        "n_unanswerable": s["n_unanswerable"],
    }
    row.update(s["metrics"])
    for stage in STAGES:
        lat = s["latency_ms"].get(stage)
        row[f"latency_{stage}_p50_ms"] = lat["p50"] if lat else None
        row[f"latency_{stage}_p95_ms"] = lat["p95"] if lat else None
    for name in ("top1_first_stage", "top1_rerank"):
        row[f"auroc_{name}"] = s["auroc"].get(name)
        sc = s["top1_scores"].get(name, {})
        row[f"mean_{name}_answerable"] = sc.get("answerable", {}).get("mean")
        row[f"mean_{name}_unanswerable"] = sc.get("unanswerable", {}).get("mean")
    return row


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in fields})


def breakdown_rows(summaries: Sequence[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    rows = []
    for s in summaries:
        for group, b in s["breakdown"][kind].items():
            row: dict[str, Any] = {
                "config_id": s["config_id"],
                "label": s.get("label", ""),
                kind: group,
                "n": b["n"],
                "n_answerable": b["n_answerable"],
            }
            row.update(b["metrics"])
            rows.append(row)
    return rows


def bootstrap_rows(
    records_by: dict[str, list[dict[str, Any]]],
    baseline_id: str,
    metrics: Sequence[str] = ("recall@5", "coverage@1024"),
    n_resamples: int = 1000,
    seed: int = 42,
    label: str = "",
) -> list[dict[str, Any]]:
    """Paired bootstrap of (config - baseline) over answerable questions present in both."""
    base = {r["qid"]: r for r in records_by.get(baseline_id, []) if r["metrics"]}
    rows = []
    for cid, recs in records_by.items():
        if cid == baseline_id:
            continue
        pairs = [(r, base[r["qid"]]) for r in recs if r["metrics"] and r["qid"] in base]
        for m in metrics:
            res = paired_bootstrap(
                [a["metrics"][m] for a, _b in pairs],
                [b["metrics"][m] for _a, b in pairs],
                n_resamples=n_resamples,
                seed=seed,
            )
            rows.append(
                {
                    "config_id": cid,
                    "baseline_id": baseline_id,
                    "label": label,
                    "metric": m,
                    **res,
                    "n_resamples": n_resamples,
                    "seed": seed,
                }
            )
    return rows


def write_tables(
    tables_dir: Path,
    prefix: str,
    summaries: Sequence[dict[str, Any]],
    records_by: dict[str, list[dict[str, Any]]],
    settings: RetrievalSettings,
    baseline_id: str | None,
    label: str = "",
    extra_meta: dict[str, Any] | None = None,
) -> list[Path]:
    """Write ``<prefix>.csv``, ``<prefix>_by_type.csv``, ``_by_document``, ``_by_origin``,
    ``_bootstrap`` (when the baseline was run) and ``<prefix>.meta.json``."""
    paths = []
    p = tables_dir / f"{prefix}.csv"
    write_csv(p, [summary_row(s) for s in summaries])
    paths.append(p)
    for kind in ("type", "document", "origin"):
        p = tables_dir / f"{prefix}_by_{kind}.csv"
        write_csv(p, breakdown_rows(summaries, kind))
        paths.append(p)
    if baseline_id and baseline_id in records_by:
        p = tables_dir / f"{prefix}_bootstrap.csv"
        write_csv(
            p,
            bootstrap_rows(
                records_by,
                baseline_id,
                n_resamples=settings.eval.bootstrap_resamples,
                seed=settings.eval.seed,
                label=label,
            ),
        )
        paths.append(p)
    meta = {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "label": label,
        "n_configs": len(summaries),
        "baseline_id": baseline_id,
        "questions_sha256": sorted({s["questions_sha256"] for s in summaries}),
        "settings": settings.model_dump(mode="json"),
        "run_info": collect_run_info(),
        **(extra_meta or {}),
    }
    p = tables_dir / f"{prefix}.meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths.append(p)
    return paths


def split_counts(questions: Sequence[EvalQuestion]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {"dev": {"answerable": 0, "unanswerable": 0}}
    out["test"] = {"answerable": 0, "unanswerable": 0}
    for q in questions:
        out[split_of(q.id)]["answerable" if q.answerable else "unanswerable"] += 1
    return out
