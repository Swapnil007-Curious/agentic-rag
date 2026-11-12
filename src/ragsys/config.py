"""Pydantic v2 configuration models loaded from YAML (extra fields forbidden)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


def _check_regex(value: str | None) -> str | None:
    if value is not None:
        re.compile(value)
    return value


# --------------------------------------------------------------------------- ingest


class IngestPaths(StrictModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    # the SOURCES.csv this ingest writes, relative to raw_dir (an app index of an evaluated
    # collection writes its own file, so the evaluation corpus's record is never rewritten)
    sources_file: str = Field(default="SOURCES.csv", pattern=r"^[A-Za-z0-9_.-]+\.csv$")

    def sources_path(self) -> Path:
        return self.raw_dir / self.sources_file


class DocumentConfig(StrictModel):
    file: str
    doc_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str
    short_title: str | None = None
    doc_type: Literal["act", "rules", "generic", "textbook", "constitution"]
    # "constitution": the statute parser with Parts, Chapters inside Parts and Articles
    parser: Literal["statute", "generic", "constitution"] = "statute"
    parent_act: str | None = None
    source_url: str = ""
    start_marker: str | None = None
    end_marker: str | None = None
    back_matter_markers: list[str] = Field(default_factory=list)
    enabled: bool = True
    # ---- textbook documents (doc_type "textbook", parser "generic")
    authors: str = ""
    edition: str = ""
    downloaded_on: str = ""
    # chapters to keep, by title; page ranges come from the PDF outline (or heading detection)
    chapters: list[str] = Field(default_factory=list)
    # alternatively: every level-1 outline entry whose title matches this regex is a chapter
    # to keep (e.g. '^Chapter\s+\d+' keeps all numbered chapters and skips the front and back
    # matter); used only when ``chapters`` is empty
    outline_chapters: str | None = None
    # licence used when the PDF does not state one with a version (with where it comes from)
    licence: str = ""
    licence_url: str = ""
    licence_note: str = ""
    # extra running-header/footer patterns for this document only
    header_footer_patterns: list[str] = Field(default_factory=list)
    # pseudocode set in the text font: short lines indented past the text margin are code lines
    pseudocode_by_indent: bool = False
    # ---- statute layouts (all off by default; the consumer-law documents use none of them)
    # a bold section heading printed on its own line above "12. (1) ..." is folded into the
    # numbered line as an inline heading ("12. Heading.—(1) ..."), the form the parser expects
    section_heading_above: bool = False
    # private-use glyphs (symbol-font stars and bullets that mark amendment notes) are removed
    strip_private_use: bool = False
    # footnotes set in the body font (no size difference to detect them by): on each page, the
    # first line matching this pattern below a vertical gap of at least two line heights starts
    # the footnote area, which runs to the bottom of the page (e.g. '^\d{1,3}\.\s+(?:Sub|Ins)\.')
    footnote_start_pattern: str | None = None

    _v_start = field_validator(
        "start_marker", "end_marker", "outline_chapters", "footnote_start_pattern"
    )(_check_regex)

    @field_validator("back_matter_markers", "header_footer_patterns")
    @classmethod
    def _v_back(cls, value: list[str]) -> list[str]:
        for v in value:
            re.compile(v)
        return value

    @model_validator(mode="after")
    def _textbook_parser(self) -> DocumentConfig:
        if self.doc_type == "textbook" and self.parser != "generic":
            raise ValueError("textbook documents use parser: generic")
        return self

    @property
    def display_title(self) -> str:
        return self.short_title or self.title


class CleaningConfig(StrictModel):
    header_footer_min_page_fraction: float = Field(default=0.5, gt=0, le=1)
    header_footer_y_tolerance: float = Field(default=8.0, gt=0)
    header_footer_min_pages: int = Field(default=3, ge=2)
    header_footer_band: float = Field(default=0.15, gt=0, lt=0.5)
    page_number_band: float = Field(default=0.12, gt=0, lt=0.5)
    devanagari_line_threshold: float = Field(default=0.30, ge=0, le=1)
    devanagari_page_threshold: float = Field(default=0.50, ge=0, le=1)
    footnote_max_size_ratio: float = Field(default=0.90, gt=0, le=1)
    marker_max_size_ratio: float = Field(default=0.80, gt=0, le=1)
    paragraph_gap_ratio: float = Field(default=0.35, gt=0)
    toc_markers: list[str] = Field(
        default_factory=lambda: [
            r"^ARRANGEMENT OF (SECTIONS|RULES|CLAUSES|REGULATIONS)$",
            r"^TABLE OF CONTENTS$",
            r"^CONTENTS$",
        ]
    )
    header_footer_patterns: list[str] = Field(default_factory=list)
    drop_line_patterns: list[str] = Field(default_factory=list)

    @field_validator("toc_markers", "header_footer_patterns", "drop_line_patterns")
    @classmethod
    def _v_patterns(cls, value: list[str]) -> list[str]:
        for v in value:
            re.compile(v)
        return value


class StructureConfig(StrictModel):
    definitions_heading_pattern: str = r"^(Definitions?|Interpretations?)\b"
    act_aliases: dict[str, str] = Field(default_factory=dict)

    _v_def = field_validator("definitions_heading_pattern")(_check_regex)


class TextbookConfig(StrictModel):
    """Layout rules for textbook PDFs (doc_type "textbook")."""

    # a line is code when at least this fraction of its characters is set in a monospace font
    monospace_font_patterns: list[str] = Field(
        default_factory=lambda: [
            "mono",
            "courier",
            "cour",
            "consol",
            "inconsolata",
            "mplus1mn",
            "menlo",
            "typewriter",
            "cmtt",
            "sfmono",
            "code",
        ]
    )
    code_min_mono_fraction: float = Field(default=0.8, gt=0, le=1)
    # headings: font size >= body + delta, or set in a heading-only font family, or bold
    heading_size_delta: float = Field(default=0.8, gt=0)
    max_heading_chars: int = Field(default=160, gt=10)
    # sections dropped with all their subsections (matched against the heading title)
    drop_sections: list[str] = Field(
        default_factory=lambda: [
            r"(?i)^exercises?$",
            r"(?i)^problems?$",
            r"(?i)^(?:exercise\s+)?solutions?(?:\s+to\s+(?:the\s+)?exercises)?$",
            r"(?i)^references?$",
            r"(?i)^bibliography$",
            r"(?i)^index$",
            r"(?i)^further\s+reading$",
            r"(?i)^(?:table\s+of\s+)?contents$",
        ]
    )
    # paragraphs dropped from the first match to the next heading (exercise lists inside a
    # "Discussion and Exercises" section)
    drop_from_patterns: list[str] = Field(
        default_factory=lambda: [r"^Exercise\s+\d+(?:\.\d+)*\.?(?:\s|$)"]
    )
    # a heading matching one of these starts end-of-chapter matter: everything up to the next
    # chapter heading is dropped (OpenStax "Key Terms", "Review Questions", ...); empty: off
    drop_to_chapter_end: list[str] = Field(default_factory=list)
    # a numbered heading followed directly by another heading whose number is printed again as
    # a heading later (a chapter-opening list of its sections) is dropped; off by default
    drop_outline_headings: bool = False
    # standalone numbers in a font smaller than this fraction of the body size are margin markers
    small_number_ratio: float = Field(default=0.85, gt=0, le=1)

    @field_validator("drop_sections", "drop_from_patterns", "drop_to_chapter_end")
    @classmethod
    def _v_patterns(cls, value: list[str]) -> list[str]:
        for v in value:
            re.compile(v)
        return value


class IngestConfig(StrictModel):
    paths: IngestPaths = Field(default_factory=IngestPaths)
    documents: list[DocumentConfig]
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    textbook: TextbookConfig = Field(default_factory=TextbookConfig)
    # SOURCES.csv columns: "statute" (the original eight) or "textbook" (adds authors, edition,
    # licence, licence source, pages kept and chapters)
    sources_format: Literal["statute", "textbook"] = "statute"

    @model_validator(mode="after")
    def _unique_doc_ids(self) -> IngestConfig:
        ids = [d.doc_id for d in self.documents]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate doc_id(s): {dupes}")
        return self

    def enabled_documents(self) -> list[DocumentConfig]:
        return [d for d in self.documents if d.enabled]


# --------------------------------------------------------------------------- chunking


class ChunkPaths(StrictModel):
    processed_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/processed/cache")


class FixedChunkerConfig(StrictModel):
    name: str
    size: int = Field(gt=8)
    overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> FixedChunkerConfig:
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        return self


class SemanticChunkerConfig(StrictModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    batch_size: int = Field(default=32, gt=0)
    breakpoint_percentile: float = Field(default=90.0, gt=0, lt=100)
    min_tokens: int = Field(default=64, gt=0)
    max_tokens: int = Field(default=512, gt=0)

    @model_validator(mode="after")
    def _min_lt_max(self) -> SemanticChunkerConfig:
        if self.min_tokens >= self.max_tokens:
            raise ValueError("min_tokens must be smaller than max_tokens")
        return self


class StructuralChunkerConfig(StrictModel):
    max_tokens: int = Field(default=384, gt=0)
    min_tokens: int = Field(default=48, ge=0)
    contextual_header: bool = True
    # also chunk text that lies outside every unit (a textbook chapter's introduction before its
    # first section); off for statutes, whose chapters hold only their headings
    orphan_text: bool = False


class ChunkingConfig(StrictModel):
    paths: ChunkPaths = Field(default_factory=ChunkPaths)
    tokenizer_model: str = "BAAI/bge-small-en-v1.5"
    seed: int = 42
    fixed: list[FixedChunkerConfig] = Field(
        default_factory=lambda: [
            FixedChunkerConfig(name="fixed_256", size=256, overlap=32),
            FixedChunkerConfig(name="fixed_512", size=512, overlap=64),
        ]
    )
    semantic: SemanticChunkerConfig = Field(default_factory=SemanticChunkerConfig)
    structural: StructuralChunkerConfig = Field(default_factory=StructuralChunkerConfig)
    # chunkers run by "--chunker all" (null: every fixed chunker, semantic and structural)
    chunkers: list[str] | None = None


# --------------------------------------------------------------------------- loading

M = TypeVar("M", bound=BaseModel)


def load_yaml_config(path: str | Path, model: type[M]) -> M:
    """Load and validate a YAML file into ``model`` (unknown keys raise)."""
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return model.model_validate(data)


def load_ingest_config(path: str | Path) -> IngestConfig:
    return load_yaml_config(path, IngestConfig)


def load_chunking_config(path: str | Path) -> ChunkingConfig:
    return load_yaml_config(path, ChunkingConfig)


def resolve_path(path: Path, base: Path = REPO_ROOT) -> Path:
    """Resolve a config path relative to the repository root (absolute paths unchanged)."""
    return path if path.is_absolute() else (base / path)
