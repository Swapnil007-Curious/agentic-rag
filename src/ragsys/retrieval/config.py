"""Retrieval configuration: one :class:`RetrievalConfig` per retriever, plus the repository-wide
:class:`RetrievalSettings` (paths, model registry, evaluation and grid) from YAML."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ragsys.config import StrictModel, load_yaml_config

RetrieverType = Literal["bm25", "dense", "hybrid_rrf", "hybrid_linear"]
RETRIEVER_TYPES: tuple[str, ...] = ("bm25", "dense", "hybrid_rrf", "hybrid_linear")

DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class RetrievalConfig(StrictModel):
    """Everything that determines one retriever's ranking (unknown keys raise)."""

    chunker: str = "structural"
    retriever: RetrieverType = "hybrid_rrf"
    dense_model: str = "bge-base"  # key of RetrievalSettings.dense_models, or a model name
    query_instruction: bool = True  # bge query prefix, applied to queries only
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    bm25_stem: bool = False
    rrf_k: int = Field(default=60, gt=0)
    alpha: float = Field(default=0.5, ge=0, le=1)  # weight of dense in linear fusion
    first_stage_k: int = Field(default=50, gt=0)
    xref_expand: bool = False
    xref_top_m: int = Field(default=5, gt=0)
    xref_max_added: int = Field(default=10, ge=0)
    xref_per_target: int = Field(default=1, gt=0)
    rerank: bool = False
    rerank_model: str = "minilm"  # key of RetrievalSettings.rerankers, or a model name
    rerank_n: int = Field(default=20, gt=0)
    final_k: int = Field(default=10, gt=0)

    @property
    def uses_bm25(self) -> bool:
        return self.retriever in ("bm25", "hybrid_rrf", "hybrid_linear")

    @property
    def uses_dense(self) -> bool:
        return self.retriever in ("dense", "hybrid_rrf", "hybrid_linear")

    def effective(self) -> dict[str, Any]:
        """Only the fields that influence the ranking of this retriever type."""
        d = self.model_dump()
        keep = {"chunker", "retriever", "first_stage_k", "final_k", "xref_expand", "rerank"}
        if self.uses_bm25:
            keep |= {"bm25_k1", "bm25_b", "bm25_stem"}
        if self.uses_dense:
            keep |= {"dense_model", "query_instruction"}
        if self.retriever == "hybrid_rrf":
            keep.add("rrf_k")
        if self.retriever == "hybrid_linear":
            keep.add("alpha")
        if self.xref_expand:
            keep |= {"xref_top_m", "xref_max_added", "xref_per_target"}
        if self.rerank:
            keep |= {"rerank_model", "rerank_n"}
        return {k: v for k, v in d.items() if k in keep}

    def config_id(self) -> str:
        """Short deterministic id, e.g. ``structural|hybrid_rrf|bge-base|xref|rerank-minilm``.

        Parameters that are not spelled out (BM25 k1/b, RRF k, first-stage depth, ...) are
        appended as ``cfg-<hash>`` when they differ from the defaults, so distinct rankings
        never share an id.
        """
        parts = [self.chunker, self.retriever]
        if self.uses_dense:
            parts.append(self.dense_model)
        if self.retriever == "hybrid_linear":
            parts.append(f"a{self.alpha:g}")
        if self.xref_expand:
            parts.append("xref")
        if self.rerank:
            parts.append(f"rerank-{self.rerank_model}")
        spelled = {"chunker", "retriever", "dense_model", "alpha", "xref_expand", "rerank"}
        spelled.add("rerank_model")
        defaults = RetrievalConfig(chunker=self.chunker, retriever=self.retriever).model_dump()
        extra = {
            k: v for k, v in self.effective().items() if k not in spelled and v != defaults.get(k)
        }
        if extra:
            digest = hashlib.sha256(json.dumps(extra, sort_keys=True).encode()).hexdigest()[:6]
            parts.append(f"cfg-{digest}")
        return "|".join(parts)

    @property
    def dir_name(self) -> str:
        """Filesystem-safe form of :meth:`config_id` ('|' is not allowed in Windows paths)."""
        return config_dir_name(self.config_id())


def config_dir_name(config_id: str) -> str:
    return config_id.replace("|", "__")


def parse_config_id(config_id: str, **overrides: Any) -> RetrievalConfig:
    """Inverse of :meth:`RetrievalConfig.config_id` for ids without a ``cfg-`` hash."""
    parts = config_id.replace("__", "|").split("|")
    if len(parts) < 2:
        raise ValueError(f"not a config id: {config_id!r}")
    data: dict[str, Any] = {"chunker": parts[0], "retriever": parts[1]}
    if data["retriever"] not in RETRIEVER_TYPES:
        raise ValueError(f"unknown retriever {data['retriever']!r} in {config_id!r}")
    rest = parts[2:]
    if data["retriever"] != "bm25":
        if not rest:
            raise ValueError(f"{config_id!r}: dense model missing")
        data["dense_model"] = rest.pop(0)
    for p in rest:
        if p == "xref":
            data["xref_expand"] = True
        elif p.startswith("rerank-"):
            data["rerank"] = True
            data["rerank_model"] = p[len("rerank-") :]
        elif p.startswith("a") and data["retriever"] == "hybrid_linear":
            data["alpha"] = float(p[1:])
        elif p.startswith("cfg-"):
            raise ValueError(
                f"{config_id!r} has non-default parameters (hash {p}); "
                "load its full config from the run's summary.json or pass flags instead"
            )
        else:
            raise ValueError(f"unknown part {p!r} in config id {config_id!r}")
    data.update(overrides)
    return RetrievalConfig.model_validate(data)


# --------------------------------------------------------------------------- settings


class RetrievalPaths(StrictModel):
    processed_dir: Path = Path("data/processed")  # reads chunks/ and structure/
    emb_dir: Path = Path("data/processed/emb")
    chroma_dir: Path = Path("data/processed/chroma")
    results_dir: Path = Path("results")
    questions: Path = Path("data/eval/questions.jsonl")


class GridBlock(StrictModel):
    """Cartesian product chunkers x retrievers x xref_expand x rerank."""

    chunkers: list[str]
    retrievers: list[dict[str, Any]]  # partial RetrievalConfig, e.g. {retriever: dense, ...}
    xref_expand: list[bool] = Field(default_factory=lambda: [False])
    rerank: list[bool] = Field(default_factory=lambda: [False])
    rerank_model: str = "minilm"

    @field_validator("retrievers")
    @classmethod
    def _v_retrievers(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in value:
            RetrievalConfig.model_validate(r)
        return value


class EvalSettings(StrictModel):
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    budgets: list[int] = Field(default_factory=lambda: [512, 1024, 2048])
    bootstrap_resamples: int = Field(default=1000, gt=0)
    seed: int = 42


class SelectionSettings(StrictModel):
    # Baselines are always carried to the test evaluation.
    baselines: dict[str, str] = Field(
        default_factory=lambda: {
            "naive_rag": "fixed_512|dense|bge-base",
            "bm25": "fixed_512|bm25",
        }
    )
    reference_baseline: str = "naive_rag"  # bootstrap differences are computed against this
    min_answerable_for_test: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def _ref_known(self) -> SelectionSettings:
        if self.reference_baseline not in self.baselines:
            raise ValueError("reference_baseline must be one of the baselines")
        return self


class RetrievalSettings(StrictModel):
    # "eval": the frozen evaluation index (every evaluation script requires it); "app": an index
    # served by ask.py and the web UI only (documents that are not evaluated); evaluation
    # scripts refuse it
    index: Literal["eval", "app"] = "eval"
    paths: RetrievalPaths = Field(default_factory=RetrievalPaths)
    seed: int = 42
    device: str = "cpu"
    embed_batch_size: int = Field(default=16, gt=0)
    rerank_batch_size: int = Field(default=16, gt=0)
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    dense_models: dict[str, str] = Field(
        default_factory=lambda: {
            "bge-small": "BAAI/bge-small-en-v1.5",
            "bge-base": "BAAI/bge-base-en-v1.5",
            "bge-large": "BAAI/bge-large-en-v1.5",
        }
    )
    rerankers: dict[str, str] = Field(
        default_factory=lambda: {
            "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "bge-reranker-base": "BAAI/bge-reranker-base",
        }
    )
    defaults: RetrievalConfig = Field(default_factory=RetrievalConfig)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    grid: list[GridBlock] = Field(default_factory=list)
    selection: SelectionSettings = Field(default_factory=SelectionSettings)

    def dense_model_name(self, key: str) -> str:
        return self.dense_models.get(key, key)

    def reranker_name(self, key: str) -> str:
        return self.rerankers.get(key, key)

    def grid_configs(self) -> list[RetrievalConfig]:
        """Expand the grid blocks in order, dropping duplicate config ids."""
        out: list[RetrievalConfig] = []
        seen: set[str] = set()
        for block in self.grid:
            for chunker, retr, xref, rr in itertools.product(
                block.chunkers, block.retrievers, block.xref_expand, block.rerank
            ):
                data = self.defaults.model_dump()
                data.update(retr)
                data.update(
                    chunker=chunker, xref_expand=xref, rerank=rr, rerank_model=block.rerank_model
                )
                cfg = RetrievalConfig.model_validate(data)
                cid = cfg.config_id()
                if cid not in seen:
                    seen.add(cid)
                    out.append(cfg)
        return out


def load_retrieval_settings(path: str | Path) -> RetrievalSettings:
    return load_yaml_config(path, RetrievalSettings)
