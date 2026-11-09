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


class DocumentConfig(StrictModel):
    file: str
    doc_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str
    short_title: str | None = None
    doc_type: Literal["act", "rules", "generic"]
    parser: Literal["statute", "generic"] = "statute"
    parent_act: str | None = None
    source_url: str = ""
    start_marker: str | None = None
    end_marker: str | None = None
    back_matter_markers: list[str] = Field(default_factory=list)
    enabled: bool = True

    _v_start = field_validator("start_marker", "end_marker")(_check_regex)

    @field_validator("back_matter_markers")
    @classmethod
    def _v_back(cls, value: list[str]) -> list[str]:
        for v in value:
            re.compile(v)
        return value

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


class IngestConfig(StrictModel):
    paths: IngestPaths = Field(default_factory=IngestPaths)
    documents: list[DocumentConfig]
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)

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
