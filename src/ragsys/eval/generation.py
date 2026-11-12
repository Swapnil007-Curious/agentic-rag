"""Generation runs on one split: resumable per question, derived systems, and the answer
evaluation that writes the tables.

Per system, ``results/generation/<split>/<system>/per_question.jsonl`` gets one record per
question as soon as it is answered, so an interrupted run resumes where it stopped. A record is
reused only when its question digest (id, question, type, answerability, gold nodes and answer)
and the system fingerprint (system definition, retrieval config, context budgets, loop
settings, prompt texts, LLM model and parameters) still match; stale records are recomputed
(the LLM response cache makes unchanged calls free). ``eval.jsonl`` holds the per-question
answer metrics, including the NLI score of every sentence.

Questions always come from :func:`ragsys.eval.split.questions_for_split` in the calling script,
so the test split needs a grant from ``scripts/eval_generation.py``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ragsys.agents.config import GenerationSettings
from ragsys.agents.derive import derive_records
from ragsys.agents.engine import AnswerEngine
from ragsys.config import resolve_path
from ragsys.eval.answers import (
    AnswerEvaluator,
    NLICache,
    NLIScorer,
    make_nli_model,
    write_answer_tables,
)
from ragsys.eval.schema import EvalQuestion
from ragsys.llm.client import LLMConfig
from ragsys.profiles import CONSUMER_LAW, GenerationPrompts, get_profile
from ragsys.retrieval.pipeline import RetrievalResources
from ragsys.utils.memory import fmt_duration, fmt_mb, rss_bytes

Log = Callable[[str], None]


def question_digest(q: EvalQuestion) -> str:
    blob = json.dumps(
        {
            "id": q.id,
            "question": q.question,
            "type": q.type,
            "answerable": q.answerable,
            "gold_node_ids": q.gold_node_ids,
            "gold_answer": q.gold_answer,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def questions_digest(questions: Sequence[EvalQuestion]) -> str:
    h = hashlib.sha256()
    for q in sorted(questions, key=lambda x: x.id):
        h.update(question_digest(q).encode())
    return h.hexdigest()


def prompt_digest(prompts: GenerationPrompts | None = None) -> str:
    """Digest of the synthesizer and critic prompts (default: the consumer_law profile)."""
    p = prompts or CONSUMER_LAW.generation
    text = "\n".join(
        [p.synth_version, p.synth_system, p.plain_rule, p.critic_version, p.critic_system]
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


SYSTEM_KEYS_ADDED_LATER = ("backend", "ablation_of")  # left out of the hash when unset


def system_fingerprint(
    settings: GenerationSettings, system: str, llm: LLMConfig | None = None
) -> str:
    """Hash of everything that determines a runnable system's outputs. ``llm`` defaults to the
    LLM config of the system's backend. Keys added to the system definition after the first
    runs are hashed only when set, so existing systems keep their fingerprints."""
    base = settings.runnable_base(system)
    backend = settings.backend_of(base)
    if llm is None:
        llm = settings.llm(backend)
    spec = settings.system(base).model_dump(mode="json")
    for key in SYSTEM_KEYS_ADDED_LATER:
        if spec.get(key) is None:
            spec.pop(key, None)
    blob = json.dumps(
        {
            "system": spec,
            "retrieval": settings.retrieval_cfg(base).model_dump(mode="json"),
            "context": settings.context_for(backend).model_dump(mode="json"),
            "loop": settings.loop.model_dump(mode="json"),
            "prompts": prompt_digest(get_profile(settings.profile).generation),
            "llm": {
                "base_url": llm.base_url,
                "model": llm.model,
                "temperature": llm.temperature,
                "max_tokens": llm.max_tokens,
                "seed": llm.seed,
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line cut off by an interruption
    return out


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def make_record(
    q: EvalQuestion, result_record: dict[str, Any], split: str, fingerprint: str
) -> dict[str, Any]:
    return {
        "qid": q.id,
        "split": split,
        "type": q.type,
        "answerable": q.answerable,
        "origin": q.origin,
        "difficulty": q.difficulty,
        "gold_node_ids": list(q.gold_node_ids),
        "question_digest": question_digest(q),
        "fingerprint": fingerprint,
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        **result_record,
    }


def run_system(
    engine: AnswerEngine,
    system: str,
    questions: Sequence[EvalQuestion],
    split: str,
    out_dir: Path,
    fingerprint: str,
    *,
    overwrite: bool = False,
    log: Log = print,
) -> list[dict[str, Any]]:
    """Answer every question with ``system`` (resumable); returns records in question order."""
    if getattr(engine.retrieval_settings, "index", "eval") == "app":
        from ragsys.collections import AppIndexRefused

        raise AppIndexRefused("generation evaluations run on the frozen evaluation index only")
    path = out_dir / system / "per_question.jsonl"
    wanted = {q.id: question_digest(q) for q in questions}
    done: dict[str, dict[str, Any]] = {}
    if not overwrite:
        for r in read_jsonl(path):
            if r.get("fingerprint") == fingerprint and wanted.get(r.get("qid", "")) == r.get(
                "question_digest"
            ):
                done[r["qid"]] = r
    write_jsonl(path, [done[q.id] for q in questions if q.id in done])  # drop stale lines
    todo = [q for q in questions if q.id not in done]
    log(
        f"== {system}: {len(questions)} question(s), {len(done)} done, {len(todo)} to run "
        f"| RSS {fmt_mb(rss_bytes())}"
    )
    spent: list[float] = []
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for i, q in enumerate(todo, start=1):
            t0 = time.perf_counter()
            result = engine.ask(q.question, system)
            rec = make_record(q, result.to_record(), split, fingerprint)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[q.id] = rec
            spent.append(time.perf_counter() - t0)
            eta = fmt_duration((len(todo) - i) * sum(spent) / len(spent))
            cost = rec["cost"]
            log(
                f"[{system} {len(done):3d}/{len(questions)}] {q.id} {rec['outcome']:9s}"
                f"{' (' + str(rec['reason']) + ')' if rec['reason'] else ''} "
                f"calls={cost['llm_calls']} cached={cost['cached_calls']} "
                f"e2e={cost['e2e_ms'] / 1000:.1f}s wall={spent[-1]:.1f}s ETA {eta}"
            )
    return [done[q.id] for q in questions]


def derive_system(
    settings: GenerationSettings,
    system: str,
    base_records: list[dict[str, Any]],
    out_dir: Path,
    threshold: float | None,
) -> list[dict[str, Any]]:
    recs = derive_records(settings, system, base_records, threshold)
    write_jsonl(out_dir / system / "per_question.jsonl", recs)
    return recs


def load_system_records(
    out_dir: Path, system: str, questions: Sequence[EvalQuestion]
) -> list[dict[str, Any]] | None:
    """Records of ``system`` covering exactly ``questions`` (current digests), else None."""
    wanted = {q.id: question_digest(q) for q in questions}
    recs = {
        r["qid"]: r
        for r in read_jsonl(out_dir / system / "per_question.jsonl")
        if wanted.get(r.get("qid", "")) == r.get("question_digest")
    }
    if set(recs) != set(wanted):
        return None
    return [recs[q.id] for q in questions]


def make_scorer(settings: GenerationSettings) -> tuple[NLIScorer, NLICache]:
    nli = settings.nli
    model = make_nli_model(nli.model, nli.fallback_model, nli.batch_size)
    cache = NLICache(resolve_path(nli.cache_dir) / "nli_scores.sqlite")
    return NLIScorer(model, cache, nli.window_tokens, nli.window_stride), cache


def evaluate_systems(
    settings: GenerationSettings,
    resources: RetrievalResources,
    records_by_system: dict[str, list[dict[str, Any]]],
    questions: Sequence[EvalQuestion],
    split: str,
    out_dir: Path,
    tables_dir: Path,
    *,
    bootstrap: bool,
    log: Log = print,
    scorer: NLIScorer | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> list[Path]:
    """NLI-based answer metrics for every system; writes ``eval.jsonl`` per system and the
    ``generation_<split>`` tables."""
    own_cache: NLICache | None = None
    if scorer is None:
        scorer, own_cache = make_scorer(settings)
    rows_by_system, nli_meta = score_systems(
        settings, resources, records_by_system, questions, out_dir, scorer, log=log
    )
    meta = {
        "split": split,
        "questions_digest": questions_digest(questions),
        "nli": nli_meta,
        "systems": {s: settings.system(s).model_dump(mode="json") for s in records_by_system},
        "settings": settings.model_dump(mode="json"),
        **(extra_meta or {}),
    }
    paths = write_answer_tables(
        tables_dir,
        f"generation_{split}",
        rows_by_system,
        reference=settings.eval.reference_system,
        n_resamples=settings.eval.bootstrap_resamples,
        seed=settings.eval.seed,
        bootstrap=bootstrap,
        extra_meta=meta,
    )
    if own_cache is not None:
        own_cache.close()
    return paths


def score_systems(
    settings: GenerationSettings,
    resources: RetrievalResources,
    records_by_system: dict[str, list[dict[str, Any]]],
    questions: Sequence[EvalQuestion],
    out_dir: Path,
    scorer: NLIScorer,
    *,
    log: Log = print,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Answer metrics per system (writes ``<out_dir>/<system>/eval.jsonl``) and the NLI
    settings actually used."""
    evaluator = AnswerEvaluator(resources, scorer, settings.nli.entailment_threshold)
    qmap = {q.id: q for q in questions}
    rows_by_system: dict[str, list[dict[str, Any]]] = {}
    for system, recs in records_by_system.items():
        t0 = time.perf_counter()
        rows = evaluator.evaluate(recs, qmap)
        write_jsonl(out_dir / system / "eval.jsonl", rows)
        rows_by_system[system] = rows
        log(
            f"   evaluated {system}: {len(rows)} question(s) in {time.perf_counter() - t0:.1f}s "
            f"(NLI pairs computed so far: {scorer.computed_pairs}) | RSS {fmt_mb(rss_bytes())}"
        )
    model = scorer.model
    nli_meta = {
        "model": model.name,
        "requested_model": getattr(model, "requested", model.name),
        "fallback_reason": getattr(model, "fallback_reason", None),
        "entailment_threshold": settings.nli.entailment_threshold,
        "window_tokens": settings.nli.window_tokens,
        "window_stride": settings.nli.window_stride,
    }
    return rows_by_system, nli_meta


def ordered_systems(settings: GenerationSettings, names: Sequence[str]) -> list[str]:
    """``names`` in the order of the configuration (unknown names raise)."""
    for n in names:
        settings.system(n)
    return [s for s in settings.systems if s in set(names)]
