"""Conversation context for one request: the last ``max_turns`` turns verbatim when they fit the
backend's history budget; when they do not, the oldest messages are summarised by one cheap LLM
call and only the newest messages that fit half the budget stay verbatim.

The browser stores the summary with the conversation and sends it back: a summary is reused as
long as the messages it covers are unchanged (``digest``) and the messages after it fit, so a
long conversation needs a new summary only every few turns, never on every message.
Tokens are estimated at four characters per token.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ragsys.agents.context import sanitize
from ragsys.assistant.prompts import SUMMARY_PREFIX

Role = Literal["user", "assistant"]


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


@dataclass
class Turn:
    role: Role
    content: str


@dataclass
class Summary:
    text: str
    upto: int  # number of leading messages of the conversation it covers
    digest: str  # digest of those messages

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "upto": self.upto, "digest": self.digest}


def messages_digest(turns: Sequence[Turn]) -> str:
    blob = json.dumps([[t.role, t.content] for t in turns], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


@dataclass
class HistoryPlan:
    kept: list[Turn]  # verbatim messages, oldest first, ending with the latest user message
    summary: Summary | None  # summary placed before ``kept`` (reused or new)
    new_summary: bool  # a summarisation call was made for this request
    dropped: int  # messages neither kept nor covered by the summary
    tokens: int
    summary_calls: int = 0
    notes: list[str] = field(default_factory=list)


Summarizer = Callable[[str, list[Turn]], str]  # (previous summary text, turns) -> summary


def _start_with_user(turns: list[Turn]) -> list[Turn]:
    while turns and turns[0].role != "user":
        turns = turns[1:]
    return turns


def plan_history(
    turns: Sequence[Turn],
    budget_tokens: int,
    *,
    max_turns: int = 10,
    summary: Summary | None = None,
    summarize: Summarizer | None = None,
    target_share: float = 0.5,
) -> HistoryPlan:
    """See the module docstring. ``turns`` must end with a user message."""
    msgs = [t for t in turns if t.content.strip()]
    if not msgs or msgs[-1].role != "user":
        raise ValueError("the conversation must end with a user message")
    latest = msgs[-1]
    start = 0
    prior: Summary | None = None
    if (
        summary is not None
        and 0 < summary.upto < len(msgs)
        and summary.digest == messages_digest(msgs[: summary.upto])
    ):
        start, prior = summary.upto, summary
    earlier = msgs[start:-1]
    window = earlier[-2 * (max_turns - 1) :] if max_turns > 1 else []
    latest_tokens = estimate_tokens(latest.content)
    prior_tokens = estimate_tokens(prior.text) if prior else 0
    avail = budget_tokens - latest_tokens - prior_tokens
    window_tokens = sum(estimate_tokens(t.content) for t in window)
    if window_tokens <= avail:
        fits = _start_with_user(list(window))
        k = len(msgs) - 1 - len(fits)
        used = sum(estimate_tokens(t.content) for t in fits) + latest_tokens + prior_tokens
        return HistoryPlan([*fits, latest], prior, False, k - start, used)

    # over budget: keep the newest messages that fit ``target_share`` of the budget, summarise
    # everything before them (after the reused summary, if any)
    target = max(0, int(budget_tokens * target_share) - latest_tokens)
    kept: list[Turn] = []
    used = 0
    for t in reversed(window):
        n = estimate_tokens(t.content)
        if used + n > target:
            break
        kept.insert(0, t)
        used += n
    kept = _start_with_user(kept)
    k = len(msgs) - 1 - len(kept)
    to_summarise = msgs[start:k]
    plan_summary = prior
    calls = 0
    notes: list[str] = []
    if to_summarise and summarize is not None:
        text = summarize(prior.text if prior else "", to_summarise)
        calls = 1
        if text:
            plan_summary = Summary(text, k, messages_digest(msgs[:k]))
        else:
            notes.append("the summary of earlier messages could not be written")
    elif to_summarise:
        notes.append(f"{len(to_summarise)} earlier message(s) were left out")
    dropped = 0 if (plan_summary is not None and plan_summary.upto == k) else k - start
    used_total = (
        sum(estimate_tokens(t.content) for t in kept)
        + latest_tokens
        + (estimate_tokens(plan_summary.text) if plan_summary else 0)
    )
    return HistoryPlan(
        [*kept, latest],
        plan_summary,
        calls > 0 and plan_summary is not prior,
        dropped,
        used_total,
        calls,
        notes,
    )


def transcript(turns: Sequence[Turn], max_chars: int, per_message: int = 2000) -> str:
    """Plain-text transcript of ``turns`` for the summariser (each message clipped)."""
    lines = []
    for t in turns:
        text = t.content.strip()
        if len(text) > per_message:
            text = text[:per_message] + " [...]"
        lines.append(f"{'User' if t.role == 'user' else 'Assistant'}: {text}")
    out = "\n\n".join(lines)
    return out[-max_chars:]


def summary_message(summary: Summary) -> dict[str, str]:
    """The summary as a user-role message (context, never system instructions)."""
    return {"role": "user", "content": f"{SUMMARY_PREFIX}\n{sanitize(summary.text)}"}
