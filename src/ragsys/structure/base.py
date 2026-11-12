"""Common node model and the pluggable StructureParser protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from ragsys.config import DocumentConfig, StructureConfig
from ragsys.ingest.clean import CleanDocument

Level = Literal[
    "chapter",
    "section",
    "rule",
    "subsection",
    "clause",
    "subclause",
    "explanation",
    "proviso",
    "schedule",
    "heading",
    "code",
]
DocType = Literal["act", "rules", "generic", "textbook", "constitution"]

# Levels that form the top-level retrieval units of a document ("section-level").
UNIT_LEVELS: frozenset[str] = frozenset({"section", "rule", "schedule"})


class Node(BaseModel):
    """One structural unit of a document.

    ``text`` is the full text of the node *including* its descendants and always equals
    ``clean_text[char_start:char_end]`` of the document's cleaned text file.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    doc_title: str
    doc_type: DocType
    node_id: str
    level: Level
    number: str | None = None
    heading: str | None = None
    text: str
    parent_id: str | None = None
    children: list[str] = []
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    defined_term: str | None = None
    order: int = 0

    @property
    def is_unit(self) -> bool:
        """Top-level retrieval unit: a section/rule/schedule, or a top-level heading."""
        if self.level in UNIT_LEVELS:
            return True
        return self.level == "heading" and self.parent_id is None


@dataclass
class ParseResult:
    nodes: list[Node]
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class StructureParser(Protocol):
    """Turns a cleaned document into a tree of :class:`Node` objects."""

    name: str

    def parse(self, doc: CleanDocument, cfg: DocumentConfig) -> ParseResult: ...


def get_parser(name: str, structure_cfg: StructureConfig) -> StructureParser:
    """Factory for configured parsers ("statute", "constitution" or "generic")."""
    from ragsys.structure.constitution import ConstitutionParser
    from ragsys.structure.generic import GenericParser
    from ragsys.structure.statute import IndianStatuteParser

    if name == "statute":
        return IndianStatuteParser(structure_cfg)
    if name == "constitution":
        return ConstitutionParser(structure_cfg)
    if name == "generic":
        return GenericParser()
    raise ValueError(f"unknown parser {name!r}")


def slugify(text: str, max_words: int = 8) -> str:
    words = [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]
    return "-".join(words[:max_words]) or "untitled"
