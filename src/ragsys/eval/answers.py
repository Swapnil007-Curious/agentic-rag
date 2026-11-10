"""Answer-quality evaluation, independent of the generator LLM.

NLI model: ``cross-encoder/nli-deberta-v3-base`` on CPU (``cross-encoder/nli-distilroberta-base``
if it cannot be loaded; the reason is recorded). P(entailment) of (premise, hypothesis) pairs is
cached on disk in SQLite, keyed by (model, sha256(premise), sha256(hypothesis), window
settings). Long premises are split into ``window_tokens`` windows (stride ``window_stride``,
model tokenizer) and the maximum entailment over windows is used.

Per question (definitions; see also the README):

* **faithfulness** (answered items): the answer is split into sentences (citation tags removed);
  a sentence is supported when the maximum P(entailment) over its cited passages (heading path
  + text) is >= the threshold (0.5). A sentence without a valid citation is unsupported.
  Faithfulness = supported / sentences; unsupported-claim rate = 1 - faithfulness.
* **correctness** (answerable items that were answered): P(entailment | premise = answer,
  hypothesis = gold answer) >= 0.5, or, for ``numeric`` questions whose gold answer contains
  numbers, every gold number occurs in the answer (compared by value, digits or words).
* **citation precision** = cited passages relevant to a gold node (the retrieval relevance
  definition) / cited labels (labels missing from the context count as not relevant; no
  citation gives 0); **citation recall** = gold nodes covered by the cited passages / gold nodes.
  Both over answerable items that were answered.
* **abstention**: correct refusal = abstained on an unanswerable question; false abstention =
  abstained on an answerable one; precision / recall / F1 with "unanswerable" as positive.
* **overall accuracy**: answerable and answered correctly, or unanswerable and abstained.
* **cost**: LLM calls, prompt / completion tokens, end-to-end latency (stored latencies for
  cached LLM responses and query embeddings).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ragsys.agents.abstention import prf
from ragsys.agents.guards import cited_labels
from ragsys.agents.text import split_sentences, strip_citations
from ragsys.eval.checks import extract_numbers, number_forms
from ragsys.eval.harness import write_csv
from ragsys.eval.metrics import mean, paired_bootstrap, percentile
from ragsys.eval.relevance import RelevanceJudge
from ragsys.eval.schema import EvalQuestion
from ragsys.retrieval.bm25 import STOPWORDS
from ragsys.retrieval.pipeline import RetrievalResources
from ragsys.utils.runinfo import collect_run_info

REASONS = ("synth_abstain", "guard_fail", "critic_abstain", "low_retrieval_score")


# --------------------------------------------------------------------------- NLI models


class NLIModel(Protocol):
    name: str

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...

    def windows(self, text: str, size: int, stride: int) -> list[str]: ...


_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS}


def _word_windows(text: str, size: int, stride: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    out = []
    for start in range(0, len(words), stride):
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return out


class OverlapNLI:
    """Deterministic stand-in (tests): P(entailment) = share of the hypothesis' content words
    found in the premise."""

    name = "fake:overlap-nli"

    def __init__(self) -> None:
        self.calls = 0

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls += 1
        out = []
        for premise, hypothesis in pairs:
            hw = _content_words(hypothesis)
            out.append(len(hw & _content_words(premise)) / len(hw) if hw else 0.0)
        return out

    def windows(self, text: str, size: int, stride: int) -> list[str]:
        return _word_windows(text, size, stride)


class CrossEncoderNLI:
    """sentence-transformers ``CrossEncoder`` NLI model on CPU, loaded lazily; falls back to a
    second model (recording why) when the first cannot be loaded."""

    def __init__(
        self, model: str, fallback: str | None = None, batch_size: int = 8, device: str = "cpu"
    ) -> None:
        self.name = model
        self.requested = model
        self.fallback = fallback
        self.batch_size = batch_size
        self.device = device
        self.fallback_reason: str | None = None
        self._model: Any = None
        self._entail_idx = 1

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        from sentence_transformers import CrossEncoder

        from ragsys.retrieval.embed import quiet_transformers

        quiet_transformers()
        try:
            self._model = CrossEncoder(self.requested, device=self.device)
        except Exception as exc:
            if not self.fallback:
                raise
            self.fallback_reason = (
                f"{self.requested} could not be loaded: {type(exc).__name__}: {exc}"
            )
            self.name = self.fallback
            self._model = CrossEncoder(self.fallback, device=self.device)
        labels = getattr(self._model.model.config, "id2label", {}) or {}
        idx = [int(i) for i, lab in labels.items() if "entail" in str(lab).lower()]
        self._entail_idx = idx[0] if idx else 1
        return self._model

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        model = self._load()
        probs = model.predict(
            list(pairs),
            apply_softmax=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        arr = np.asarray(probs, dtype=np.float64).reshape(len(pairs), -1)
        return [float(x) for x in arr[:, self._entail_idx]]

    def windows(self, text: str, size: int, stride: int) -> list[str]:
        tok = self._load().tokenizer
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        if len(offsets) <= size:
            return [text]
        out = []
        for start in range(0, len(offsets), stride):
            end = min(start + size, len(offsets))
            out.append(text[offsets[start][0] : offsets[end - 1][1]])
            if end == len(offsets):
                break
        return out

    def unload(self) -> None:
        self._model = None


def make_nli_model(name: str, fallback: str | None, batch_size: int) -> NLIModel:
    if name.startswith("fake:"):
        return OverlapNLI()
    return CrossEncoderNLI(name, fallback, batch_size)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NLICache:
    """(model, premise sha, hypothesis sha, windowing) -> max P(entailment) over windows."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.hits = 0
        self.misses = 0
        self._con = sqlite3.connect(str(path))
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS nli (model TEXT NOT NULL, premise_sha TEXT NOT NULL,"
            " hypothesis_sha TEXT NOT NULL, windowing TEXT NOT NULL, p_entail REAL NOT NULL,"
            " created_utc TEXT NOT NULL,"
            " PRIMARY KEY (model, premise_sha, hypothesis_sha, windowing))"
        )
        self._con.commit()

    def get(
        self, model: str, keys: Sequence[tuple[str, str]], windowing: str
    ) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for p, h in keys:
            row = self._con.execute(
                "SELECT p_entail FROM nli WHERE model=? AND premise_sha=? AND hypothesis_sha=?"
                " AND windowing=?",
                (model, p, h, windowing),
            ).fetchone()
            if row is not None:
                out[(p, h)] = float(row[0])
        self.hits += len(out)
        self.misses += len(keys) - len(out)
        return out

    def put(self, model: str, rows: Sequence[tuple[str, str, float]], windowing: str) -> None:
        now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        self._con.executemany(
            "INSERT OR REPLACE INTO nli VALUES (?, ?, ?, ?, ?, ?)",
            [(model, p, h, windowing, v, now) for p, h, v in rows],
        )
        self._con.commit()

    def close(self) -> None:
        self._con.close()


class NLIScorer:
    """Max-over-windows P(entailment) for (premise, hypothesis) items, batched and cached."""

    def __init__(
        self,
        model: NLIModel,
        cache: NLICache | None = None,
        window_tokens: int = 400,
        window_stride: int = 300,
    ) -> None:
        self.model = model
        self.cache = cache
        self.window_tokens = window_tokens
        self.window_stride = window_stride
        self.computed_pairs = 0

    @property
    def windowing(self) -> str:
        return f"w{self.window_tokens}s{self.window_stride}"

    def score(self, items: Sequence[tuple[str, str]]) -> list[float]:
        uniq = list(dict.fromkeys(items))
        keys = [(_sha(p), _sha(h)) for p, h in uniq]
        found: dict[tuple[str, str], float] = {}
        if self.cache is not None:
            found = self.cache.get(self.model.name, keys, self.windowing)
        todo = [
            (i, item) for i, (item, k) in enumerate(zip(uniq, keys, strict=True)) if k not in found
        ]
        if todo:
            flat: list[tuple[str, str]] = []
            spans: list[tuple[int, int]] = []
            for _i, (premise, hyp) in todo:
                wins = self.model.windows(premise, self.window_tokens, self.window_stride) or [
                    premise
                ]
                spans.append((len(flat), len(flat) + len(wins)))
                flat.extend((w, hyp) for w in wins)
            probs = self.model.entailment(flat)
            self.computed_pairs += len(flat)
            rows = []
            for (i, _item), (a, b) in zip(todo, spans, strict=True):
                v = max(probs[a:b]) if b > a else 0.0
                found[keys[i]] = v
                rows.append((keys[i][0], keys[i][1], v))
            if self.cache is not None:
                self.cache.put(self.model.name, rows, self.windowing)
        by_item = {item: found[k] for item, k in zip(uniq, keys, strict=True)}
        return [by_item[it] for it in items]


# --------------------------------------------------------------------------- per question


def _bare_one(text: str) -> bool:
    return text.strip() == "one"


def gold_numbers_in_answer(gold: str, answer: str) -> bool | None:
    """True when every number of ``gold`` occurs (by value) in ``answer``; None if ``gold`` has
    no numbers (the numeric rule does not apply)."""
    golds = [m for m in extract_numbers(gold) if not _bare_one(m.text)]
    if not golds:
        return None
    forms = number_forms([strip_citations(answer)])
    return all(round(m.value, 4) in forms for m in golds)


def chunker_of(record: dict[str, Any]) -> str:
    return str(record["retrieval"]["config_id"]).split("|", 1)[0]


@dataclass
class _Plan:
    record: dict[str, Any]
    question: EvalQuestion
    sentences: list[tuple[str, list[str]]]  # (text, valid labels)
    premises: dict[str, str]  # label -> premise
    cited: list[str]  # all cited labels


class AnswerEvaluator:
    def __init__(
        self,
        resources: RetrievalResources,
        scorer: NLIScorer,
        entailment_threshold: float = 0.5,
    ) -> None:
        self.resources = resources
        self.scorer = scorer
        self.threshold = entailment_threshold
        self._judges: dict[str, RelevanceJudge] = {}

    def judge(self, chunker: str) -> RelevanceJudge:
        if chunker not in self._judges:
            self._judges[chunker] = RelevanceJudge(
                self.resources.graph(), self.resources.corpus(chunker)
            )
        return self._judges[chunker]

    def _plan(self, record: dict[str, Any], q: EvalQuestion) -> _Plan:
        corpus = self.resources.corpus(chunker_of(record))
        premises: dict[str, str] = {}
        for c in record.get("context", []):
            chunk = corpus.by_id.get(c["chunk_id"])
            if chunk is not None:
                premises[c["label"]] = f"{chunk.heading_path}\n{chunk.display_text}"
        sentences = []
        answer = record.get("answer", "") if record["outcome"] == "answered" else ""
        for s in split_sentences(answer):
            sentences.append((s.text, [lab for lab in s.labels if lab in premises]))
        cited = cited_labels(answer, record.get("citations", [])) if answer else []
        return _Plan(record, q, sentences, premises, cited)

    def evaluate(
        self, records: Sequence[dict[str, Any]], questions: dict[str, EvalQuestion]
    ) -> list[dict[str, Any]]:
        plans = [self._plan(r, questions[r["qid"]]) for r in records]
        items: list[tuple[str, str]] = []
        for pl in plans:
            for text, labels in pl.sentences:
                items.extend((pl.premises[lab], text) for lab in labels)
            if pl.question.answerable and pl.record["outcome"] == "answered":
                items.append((strip_citations(pl.record["answer"]), pl.question.gold_answer))
        scores = dict(zip(items, self.scorer.score(items), strict=True)) if items else {}
        return [self._row(pl, scores) for pl in plans]

    def _row(self, pl: _Plan, scores: dict[tuple[str, str], float]) -> dict[str, Any]:
        rec, q = pl.record, pl.question
        answered = rec["outcome"] == "answered"
        row: dict[str, Any] = {
            "qid": q.id,
            "system": rec["system"],
            "type": q.type,
            "origin": q.origin,
            "answerable": q.answerable,
            "outcome": rec["outcome"],
            "reason": rec.get("reason"),
            "answered": answered,
            "abstained": not answered,
        }
        sent_rows = []
        if answered:
            for text, labels in pl.sentences:
                p = max((scores[(pl.premises[lab], text)] for lab in labels), default=0.0)
                sent_rows.append(
                    {
                        "text": text,
                        "labels": labels,
                        "p_entail": round(p, 4),
                        "supported": p >= self.threshold,
                    }
                )
            n_sup = sum(1 for s in sent_rows if s["supported"])
            row.update(
                n_sentences=len(sent_rows),
                n_supported=n_sup,
                faithfulness=n_sup / len(sent_rows) if sent_rows else 0.0,
                any_unsupported=n_sup < len(sent_rows) or not sent_rows,
            )
        else:
            row.update(n_sentences=0, n_supported=0, faithfulness=None, any_unsupported=None)
        row["sentences"] = sent_rows

        correct: bool | None = None
        if q.answerable and answered:
            p_corr = scores[(strip_citations(rec["answer"]), q.gold_answer)]
            numeric = (
                gold_numbers_in_answer(q.gold_answer, rec["answer"])
                if q.type == "numeric"
                else None
            )
            correct = p_corr >= self.threshold or bool(numeric)
            row.update(p_correct=round(p_corr, 4), numeric_match=numeric)
            judge = self.judge(chunker_of(rec))
            golds = list(q.gold_node_ids)
            label_to_chunk = {c["label"]: c["chunk_id"] for c in rec.get("context", [])}
            relevant = 0
            covered: set[str] = set()
            for lab in pl.cited:
                cid = label_to_chunk.get(lab)
                if cid is None or cid not in judge.corpus.by_id:
                    continue
                rg = judge.relevant_golds(cid, golds)
                relevant += 1 if rg else 0
                covered |= rg
            row["n_cited"] = len(pl.cited)
            row["citation_precision"] = relevant / len(pl.cited) if pl.cited else 0.0
            row["citation_recall"] = len(covered) / len(set(golds)) if golds else 0.0
        else:
            row.update(p_correct=None, numeric_match=None, n_cited=len(pl.cited))
            row.update(citation_precision=None, citation_recall=None)
        row["correct"] = correct
        row["correct_and_faithful"] = (
            bool(correct) and row["faithfulness"] == 1.0 if q.answerable else None
        )
        row["accurate"] = bool(correct) if q.answerable else not answered
        cost = rec.get("cost", {})
        row.update(
            llm_calls=int(cost.get("llm_calls", 0)),
            prompt_tokens=int(cost.get("prompt_tokens", 0)),
            completion_tokens=int(cost.get("completion_tokens", 0)),
            e2e_ms=float(cost.get("e2e_ms", 0.0)),
        )
        return row


# --------------------------------------------------------------------------- summaries


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def summarise_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if r["answered"]]
    ans_answered = [r for r in ans if r["answered"]]
    abst = [r for r in rows if r["abstained"]]
    tp = sum(1 for r in unans if r["abstained"])
    p, rc, f = prf(tp, len(abst) - tp, len(unans) - tp)
    faith = [float(r["faithfulness"]) for r in answered]
    n_sent = sum(int(r["n_sentences"]) for r in answered)
    n_sup = sum(int(r["n_supported"]) for r in answered)
    lat = [float(r["e2e_ms"]) for r in rows]
    out: dict[str, Any] = {
        "n_questions": len(rows),
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "n_answered": len(answered),
        "answered_rate": _rate(len(answered), len(rows)),
        "overall_accuracy": _rate(sum(1 for r in rows if r["accurate"]), len(rows)),
        "faithfulness": mean(faith),
        "faithfulness_micro": _rate(n_sup, n_sent),
        "unsupported_claim_rate": None if not faith else 1.0 - float(np.mean(faith)),
        "answers_with_unsupported_share": _rate(
            sum(1 for r in answered if r["any_unsupported"]), len(answered)
        ),
        "correct_rate_answered": _rate(
            sum(1 for r in ans_answered if r["correct"]), len(ans_answered)
        ),
        "correct_rate_answerable": _rate(sum(1 for r in ans if r["correct"]), len(ans)),
        "correct_and_faithful_rate": _rate(
            sum(1 for r in ans if r["correct_and_faithful"]), len(ans)
        ),
        "citation_precision": mean([float(r["citation_precision"]) for r in ans_answered]),
        "citation_recall": mean([float(r["citation_recall"]) for r in ans_answered]),
        "correct_refusal_rate": _rate(tp, len(unans)),
        "false_abstention_rate": _rate(sum(1 for r in ans if r["abstained"]), len(ans)),
        "abstention_precision": p if abst else None,
        "abstention_recall": rc if unans else None,
        "abstention_f1": f if (abst or unans) else None,
        "llm_calls_mean": mean([float(r["llm_calls"]) for r in rows]),
        "prompt_tokens_mean": mean([float(r["prompt_tokens"]) for r in rows]),
        "completion_tokens_mean": mean([float(r["completion_tokens"]) for r in rows]),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
    }
    reasons = Counter(r["reason"] for r in abst)
    for reason in REASONS:
        out[f"abstained_{reason}"] = reasons.get(reason, 0)
    return out


def summary_rows(
    rows_by_system: dict[str, list[dict[str, Any]]], label: str = ""
) -> list[dict[str, Any]]:
    out = []
    for system, rows in rows_by_system.items():
        out.append({"system": system, "label": label, **summarise_rows(rows)})
    return out


def by_type_rows(
    rows_by_system: dict[str, list[dict[str, Any]]], label: str = ""
) -> list[dict[str, Any]]:
    out = []
    for system, rows in rows_by_system.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[r["type"]].append(r)
        for t, grp in sorted(groups.items()):
            out.append({"system": system, "label": label, "type": t, **summarise_rows(grp)})
    return out


BOOTSTRAP_METRICS = ("overall_accuracy", "faithfulness", "correct_refusal_rate")


def bootstrap_rows(
    rows_by_system: dict[str, list[dict[str, Any]]],
    reference: str,
    n_resamples: int = 1000,
    seed: int = 42,
    label: str = "",
) -> list[dict[str, Any]]:
    """Paired bootstrap of (system - reference): overall accuracy over all questions,
    faithfulness over questions both systems answered, correct refusal over unanswerable
    questions."""
    base = {r["qid"]: r for r in rows_by_system.get(reference, [])}
    out = []
    for system, rows in rows_by_system.items():
        if system == reference:
            continue
        mine = {r["qid"]: r for r in rows}
        common = [q for q in mine if q in base]
        for metric in BOOTSTRAP_METRICS:
            if metric == "overall_accuracy":
                qs = common
                a = [float(mine[q]["accurate"]) for q in qs]
                b = [float(base[q]["accurate"]) for q in qs]
                scope = "all questions"
            elif metric == "faithfulness":
                qs = [q for q in common if mine[q]["answered"] and base[q]["answered"]]
                a = [float(mine[q]["faithfulness"]) for q in qs]
                b = [float(base[q]["faithfulness"]) for q in qs]
                scope = "questions answered by both"
            else:
                qs = [q for q in common if not mine[q]["answerable"]]
                a = [float(mine[q]["abstained"]) for q in qs]
                b = [float(base[q]["abstained"]) for q in qs]
                scope = "unanswerable questions"
            res = paired_bootstrap(a, b, n_resamples=n_resamples, seed=seed)
            out.append(
                {
                    "system": system,
                    "reference": reference,
                    "label": label,
                    "metric": metric,
                    "scope": scope,
                    **res,
                    "n_resamples": n_resamples,
                    "seed": seed,
                }
            )
    return out


def write_answer_tables(
    tables_dir: Path,
    prefix: str,
    rows_by_system: dict[str, list[dict[str, Any]]],
    *,
    reference: str | None,
    n_resamples: int,
    seed: int,
    label: str = "",
    bootstrap: bool = True,
    extra_meta: dict[str, Any] | None = None,
) -> list[Path]:
    """``<prefix>.csv`` (one row per system), ``<prefix>_by_type.csv``, ``<prefix>_bootstrap.csv``
    (when the reference system is present) and ``<prefix>.meta.json``."""
    paths = []
    p = tables_dir / f"{prefix}.csv"
    write_csv(p, summary_rows(rows_by_system, label))
    paths.append(p)
    p = tables_dir / f"{prefix}_by_type.csv"
    write_csv(p, by_type_rows(rows_by_system, label))
    paths.append(p)
    if bootstrap and reference and reference in rows_by_system and len(rows_by_system) > 1:
        p = tables_dir / f"{prefix}_bootstrap.csv"
        write_csv(p, bootstrap_rows(rows_by_system, reference, n_resamples, seed, label))
        paths.append(p)
    meta = {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "label": label,
        "systems": list(rows_by_system),
        "reference": reference,
        "run_info": collect_run_info(),
        **(extra_meta or {}),
    }
    p = tables_dir / f"{prefix}.meta.json"
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    paths.append(p)
    return paths
