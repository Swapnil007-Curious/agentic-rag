"""Answer metrics on hand-computed cases (faithfulness, correctness incl. the numeric rule,
citation precision/recall with the retrieval relevance definition, abstention, accuracy, cost),
NLI windows and cache, and bootstrap reproducibility."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from ragsys.eval.answers import (
    AnswerEvaluator,
    NLICache,
    NLIScorer,
    OverlapNLI,
    bootstrap_rows,
    gold_numbers_in_answer,
    summarise_rows,
)
from ragsys.eval.metrics import paired_bootstrap
from ragsys.eval.relevance import RelevanceJudge
from ragsys.eval.schema import EvalQuestion
from ragsys.retrieval.pipeline import RetrievalResources
from retrieval_helpers import fake_settings


class KeyedNLI:
    """P(entailment) from a function of (premise, hypothesis); counts model calls."""

    name = "fake:keyed"

    def __init__(self, fn: Callable[[str, str], float]) -> None:
        self.fn = fn
        self.calls = 0

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls += 1
        return [self.fn(p, h) for p, h in pairs]

    def windows(self, text: str, size: int, stride: int) -> list[str]:
        return OverlapNLI().windows(text, size, stride)


@pytest.fixture()
def resources(retrieval_dir: Path, tmp_path: Path) -> RetrievalResources:
    s = fake_settings(retrieval_dir, tmp_path / "results", tmp_path / "q.jsonl")
    return RetrievalResources(s)


def _q(qid: str, qtype: str, golds: list[str], gold: str) -> EvalQuestion:
    return EvalQuestion.model_validate(
        {
            "id": qid,
            "question": f"question {qid}?",
            "type": qtype,
            "answerable": qtype != "unanswerable",
            "gold_node_ids": golds,
            "gold_answer": gold,
            "difficulty": "easy",
            "author": "Swapnil",
        }
    )


def _rec(
    qid: str,
    outcome: str,
    answer: str,
    ctx: list[str],
    calls: int,
    ms: float,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "qid": qid,
        "system": "GX",
        "outcome": outcome,
        "reason": reason,
        "answer": answer,
        "citations": [],
        "context": [{"label": f"C{i}", "chunk_id": c} for i, c in enumerate(ctx, start=1)],
        "retrieval": {"config_id": "structural|dense|fake-a"},
        "cost": {
            "llm_calls": calls,
            "prompt_tokens": 10 * calls,
            "completion_tokens": calls,
            "e2e_ms": ms,
        },
    }


def test_answer_metrics_by_hand(resources: RetrievalResources) -> None:
    corpus = resources.corpus("structural")
    judge = RelevanceJudge(resources.graph(), corpus)
    g1, g2 = "testact:s3(1)", "testact:s2(2)"
    a = next(c for c in corpus.chunks if judge.relevant_golds(c.chunk_id, [g1, g2]) == {g1})
    c2 = next(c for c in corpus.chunks if judge.relevant_golds(c.chunk_id, [g1, g2]) == {g2})
    b = next(c for c in corpus.chunks if c.doc_id == "report")
    assert not judge.is_relevant(b.chunk_id, [g1, g2])
    a_marker = a.display_text.split()[0]

    def fn(premise: str, hyp: str) -> float:
        if hyp == "Alpha is required.":
            return 0.9 if a.display_text in premise else 0.0
        return {
            "Beta is allowed.": 0.2,
            "GOLD1": 0.7,
            "It is thirty days.": 0.95,
            "Delta.": 0.6,
        }.get(hyp, 0.1)

    questions = {
        "q101": _q("q101", "factual", [g1, g2], "GOLD1"),
        "q102": _q("q102", "numeric", [g2], "within 30 days"),
        "q103": _q("q103", "factual", [g1], "x"),
        "q104": _q("q104", "unanswerable", [], "not covered"),
        "q105": _q("q105", "unanswerable", [], "not covered"),
    }
    records = [
        _rec(
            "q101",
            "answered",
            "Alpha is required [C1]. Beta is allowed [C2]. Gamma holds.",
            [a.chunk_id, b.chunk_id],
            2,
            1000,
        ),
        _rec("q102", "answered", "It is thirty days [C1].", [c2.chunk_id], 4, 3000),
        _rec("q103", "abstained", "", [a.chunk_id], 1, 2000, "synth_abstain"),
        _rec("q104", "abstained", "", [b.chunk_id], 1, 500, "low_retrieval_score"),
        _rec("q105", "answered", "Delta [C1].", [b.chunk_id], 2, 4000),
    ]
    nli = KeyedNLI(fn)
    rows = AnswerEvaluator(resources, NLIScorer(nli, None, 400, 300), 0.5).evaluate(
        records, questions
    )
    r1, r2, r3, r4, r5 = rows
    assert (r1["n_sentences"], r1["n_supported"], r1["any_unsupported"]) == (3, 1, True)
    assert r1["faithfulness"] == pytest.approx(1 / 3)
    assert [s["supported"] for s in r1["sentences"]] == [True, False, False]
    assert r1["sentences"][2]["labels"] == []  # no citation -> unsupported
    assert (r1["correct"], r1["p_correct"], r1["numeric_match"]) == (True, 0.7, None)
    assert (r1["citation_precision"], r1["citation_recall"]) == (0.5, 0.5)
    assert r1["correct_and_faithful"] is False and r1["accurate"] is True
    assert (r2["correct"], r2["numeric_match"], r2["p_correct"]) == (
        True,
        True,
        0.1,
    )  # numeric rule
    assert r2["correct_and_faithful"] is True and r2["citation_precision"] == 1.0
    assert (r3["accurate"], r3["correct"], r3["faithfulness"]) == (False, None, None)
    assert r4["accurate"] is True and r4["citation_precision"] is None
    assert (r5["accurate"], r5["faithfulness"], r5["citation_recall"]) == (False, 1.0, None)
    assert a_marker  # the relevant chunk has text

    s = summarise_rows(rows)
    assert (s["n_questions"], s["n_answerable"], s["n_unanswerable"], s["n_answered"]) == (
        5,
        3,
        2,
        3,
    )
    assert s["overall_accuracy"] == pytest.approx(3 / 5)
    assert s["faithfulness"] == pytest.approx((1 / 3 + 1 + 1) / 3)
    assert s["faithfulness_micro"] == pytest.approx(3 / 5)
    assert s["unsupported_claim_rate"] == pytest.approx(1 - (1 / 3 + 2) / 3)
    assert s["answers_with_unsupported_share"] == pytest.approx(1 / 3)
    assert s["correct_rate_answered"] == 1.0 and s["correct_rate_answerable"] == pytest.approx(
        2 / 3
    )
    assert s["correct_and_faithful_rate"] == pytest.approx(1 / 3)
    assert s["citation_precision"] == pytest.approx(0.75) and s["citation_recall"] == pytest.approx(
        0.75
    )
    assert s["correct_refusal_rate"] == 0.5 and s["false_abstention_rate"] == pytest.approx(1 / 3)
    assert (s["abstention_precision"], s["abstention_recall"], s["abstention_f1"]) == (
        0.5,
        0.5,
        0.5,
    )
    assert s["llm_calls_mean"] == 2.0 and s["prompt_tokens_mean"] == 20.0
    assert (s["latency_p50_ms"], s["latency_p95_ms"]) == (2000.0, pytest.approx(3800.0))
    assert (
        s["abstained_synth_abstain"],
        s["abstained_low_retrieval_score"],
        s["abstained_guard_fail"],
    ) == (1, 1, 0)


def test_gold_numbers_rule() -> None:
    assert gold_numbers_in_answer("within 30 days", "thirty days [C1]") is True
    assert gold_numbers_in_answer("fifty lakh rupees", "Rs. 50,00,000") is True
    assert gold_numbers_in_answer("within 30 days", "three months") is False
    assert gold_numbers_in_answer("the District Commission", "anything") is None


def test_nli_windows_and_cache(tmp_path: Path) -> None:
    filler = " ".join(f"word{i}" for i in range(300))
    premise = f"{filler} alpha beta gamma"

    def short_only(p: str, h: str) -> float:  # a model that only "sees" short inputs
        return 1.0 if len(p.split()) <= 60 and set(h.split()) <= set(p.split()) else 0.0

    nli = KeyedNLI(short_only)
    assert NLIScorer(nli, None, 1000, 900).score([(premise, "alpha beta gamma")]) == [0.0]
    cache = NLICache(tmp_path / "nli.sqlite")
    scorer = NLIScorer(nli, cache, 50, 40)
    assert scorer.score([(premise, "alpha beta gamma"), (premise, "alpha beta gamma")]) == [
        1.0,
        1.0,
    ]
    calls = nli.calls
    again = NLIScorer(nli, NLICache(tmp_path / "nli.sqlite"), 50, 40)
    assert again.score([(premise, "alpha beta gamma")]) == [1.0] and nli.calls == calls
    assert again.cache is not None and again.cache.hits == 1
    other = NLIScorer(nli, NLICache(tmp_path / "nli.sqlite"), 60, 50)  # other windowing: no reuse
    other.score([(premise, "alpha beta gamma")])
    assert nli.calls == calls + 1
    assert OverlapNLI().entailment([("the goods were defective", "defective goods")]) == [1.0]


def _rows(
    accurate: list[bool], answered: list[bool], answerable: list[bool]
) -> list[dict[str, Any]]:
    return [
        {
            "qid": f"q{i}",
            "accurate": acc,
            "answered": ans,
            "abstained": not ans,
            "answerable": ab,
            "faithfulness": 1.0 if acc else 0.5,
        }
        for i, (acc, ans, ab) in enumerate(zip(accurate, answered, answerable, strict=True))
    ]


def test_bootstrap_is_paired_and_reproducible() -> None:
    ref = _rows(
        [False, True, False, True, False, False],
        [True, True, True, True, True, True],
        [True] * 4 + [False] * 2,
    )
    sys2 = _rows(
        [True, True, False, True, True, True],
        [True, True, False, True, False, False],
        [True] * 4 + [False] * 2,
    )
    by = {"G1": ref, "G2": sys2}
    r1 = bootstrap_rows(by, "G1", 300, 11)
    r2 = bootstrap_rows(by, "G1", 300, 11)
    assert r1 == r2 and [r["metric"] for r in r1] == [
        "overall_accuracy",
        "faithfulness",
        "correct_refusal_rate",
    ]
    acc = r1[0]
    direct = paired_bootstrap(
        [float(x["accurate"]) for x in sys2], [float(x["accurate"]) for x in ref], 300, 11
    )
    assert {k: acc[k] for k in direct} == direct and acc["mean_diff"] == pytest.approx(3 / 6)
    assert r1[1]["n"] == 3  # faithfulness: questions both systems answered
    assert r1[2]["n"] == 2 and r1[2]["mean_diff"] == 1.0  # both unanswerable refused vs none
    assert bootstrap_rows({"G1": ref}, "G1") == []
