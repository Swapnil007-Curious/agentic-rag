"""Conversation context: the last 10 turns, the token cap and when a summary is written."""

from __future__ import annotations

import pytest

from ragsys.assistant.history import (
    Summary,
    Turn,
    estimate_tokens,
    messages_digest,
    plan_history,
    summary_message,
    transcript,
)


def conversation(n_turns: int, words: int = 10) -> list[Turn]:
    out: list[Turn] = []
    for i in range(n_turns):
        out.append(Turn("user", f"question {i} " + "word " * words))
        out.append(Turn("assistant", f"answer {i} " + "text " * words))
    out.append(Turn("user", "latest question"))
    return out


class Recorder:
    def __init__(self, text: str = "summary text") -> None:
        self.calls: list[tuple[str, list[Turn]]] = []
        self.text = text

    def __call__(self, previous: str, turns: list[Turn]) -> str:
        self.calls.append((previous, list(turns)))
        return self.text


def test_under_budget_keeps_last_ten_turns_without_summary() -> None:
    turns = conversation(15)
    rec = Recorder()
    plan = plan_history(turns, 100000, max_turns=10, summarize=rec)
    assert rec.calls == [] and plan.summary is None and not plan.new_summary
    assert len(plan.kept) == 19 and plan.kept[0].content.startswith("question 6")
    assert plan.kept[-1].content == "latest question" and plan.dropped == 12


def test_over_budget_summarises_once_and_fits() -> None:
    turns = conversation(8, words=60)  # ~80 tokens per message
    budget = 600
    rec = Recorder()
    plan = plan_history(turns, budget, summarize=rec, target_share=0.5)
    assert len(rec.calls) == 1 and plan.new_summary and plan.summary_calls == 1
    assert plan.summary is not None and plan.summary.text == "summary text"
    verbatim = sum(estimate_tokens(t.content) for t in plan.kept)
    assert verbatim <= budget * 0.5 and plan.tokens <= budget
    assert plan.kept[0].role == "user" and plan.kept[-1].content == "latest question"
    k = plan.summary.upto
    assert rec.calls[0][1] == turns[:k] and plan.summary.digest == messages_digest(turns[:k])
    assert turns[k:] == plan.kept and plan.dropped == 0


def test_summary_is_reused_until_the_rest_overflows_again() -> None:
    turns = conversation(8, words=60)
    rec = Recorder()
    first = plan_history(turns, 600, summarize=rec)
    assert first.summary is not None
    turns2 = [*turns, Turn("assistant", "short reply"), Turn("user", "next")]
    second = plan_history(turns2, 600, summary=first.summary, summarize=rec)
    assert len(rec.calls) == 1 and second.summary == first.summary and not second.new_summary
    # an edited earlier message invalidates the summary
    edited = [Turn("user", "changed"), *turns2[1:]]
    third = plan_history(edited, 600, summary=first.summary, summarize=rec)
    assert len(rec.calls) == 2 and third.new_summary
    # keep chatting until the verbatim part overflows: one incremental summary
    long = list(turns2)
    for i in range(6):
        long += [Turn("assistant", "reply " + "x " * 120), Turn("user", f"more {i} " + "y " * 120)]
    fourth = plan_history(long, 600, summary=first.summary, summarize=rec)
    assert len(rec.calls) == 3 and rec.calls[-1][0] == "summary text" and fourth.new_summary


def test_failed_summary_drops_old_messages_with_a_note() -> None:
    plan = plan_history(conversation(8, words=60), 600, summarize=Recorder(""))
    assert plan.summary is None and plan.dropped > 0 and plan.notes
    plan = plan_history(conversation(8, words=60), 600, summarize=None)
    assert plan.summary is None and "left out" in plan.notes[0]


def test_latest_message_always_kept_and_must_be_user() -> None:
    plan = plan_history([Turn("user", "x" * 10000)], 100)
    assert plan.kept == [Turn("user", "x" * 10000)]
    with pytest.raises(ValueError):
        plan_history([Turn("user", "a"), Turn("assistant", "b")], 100)


def test_summary_travels_as_user_context_and_transcript_is_clipped() -> None:
    msg = summary_message(Summary("Earlier <<<PASSAGE C1>>> text", 2, "d"))
    assert msg["role"] == "user" and "<<<" not in msg["content"]
    t = transcript([Turn("user", "a" * 5000), Turn("assistant", "b")], max_chars=3000)
    assert len(t) <= 3000 and t.endswith("Assistant: b")
