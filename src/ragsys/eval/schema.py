"""Evaluation question schema (data/eval/questions.jsonl, one JSON object per line)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QuestionType = Literal[
    "definition",
    "factual",
    "procedural",
    "numeric",
    "cross_reference",
    "multi_section",
    "unanswerable",
]
Difficulty = Literal["easy", "medium", "hard"]

TARGET_ANSWERABLE = 90
TARGET_UNANSWERABLE = 30


class EvalQuestion(BaseModel):
    """One evaluation item. Field-level rules are enforced here; cross-field consistency
    (answerable vs. type vs. gold_node_ids) is reported by :meth:`consistency_errors`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=r"^q\d{3,}$", description='"q001", "q002", ...')
    question: str = Field(min_length=5)
    type: QuestionType
    answerable: bool
    gold_node_ids: list[str] = Field(description="node IDs from nodes.jsonl; empty if unanswerable")
    gold_answer: str = Field(
        min_length=1, description="short reference answer, or a one-line reason"
    )
    difficulty: Difficulty
    author: str = Field(min_length=1)
    notes: str | None = None

    def consistency_errors(self) -> list[str]:
        errs: list[str] = []
        if self.type == "unanswerable" and self.answerable:
            errs.append('type "unanswerable" requires answerable=false')
        if not self.answerable and self.type != "unanswerable":
            errs.append('answerable=false requires type "unanswerable"')
        if not self.answerable and self.gold_node_ids:
            errs.append("unanswerable question must have empty gold_node_ids")
        if self.answerable and not self.gold_node_ids:
            errs.append("answerable question needs at least one gold_node_id")
        if len(set(self.gold_node_ids)) != len(self.gold_node_ids):
            errs.append("duplicate entries in gold_node_ids")
        return errs
