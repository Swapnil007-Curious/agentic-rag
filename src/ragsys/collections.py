"""Collections: named document sets, each with its own configs, data, caches and results.

``configs/collections.yaml`` maps a collection name to its four config files (ingest, chunking,
retrieval, generation), the LLM endpoint config, a domain profile (prompt wording, question-type
mix, out-of-scope topics; see :mod:`ragsys.eval.profiles`) and four root directories::

    raw_dir        PDFs + SOURCES.csv
    processed_dir  clean text, structure, chunks, embeddings and every cache
    eval_dir       questions.jsonl, drafts.jsonl
    results_dir    retrieval/, generation/, tables/, test_access_log.json

Every other location is derived from these roots (see :class:`Collection`), so two collections
never share an output or a cache. The paths written inside the config files must agree with the
roots; :func:`consistency_errors` checks that (and a test runs it for every collection).

The collection ``consumer_law`` maps to the original locations (``data/raw``, ``data/processed``,
``data/eval``, ``results``) and the original config files, so every command that existed before
collections were introduced resolves exactly the same paths.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field, model_validator

from ragsys.config import StrictModel, load_yaml_config, resolve_path

DEFAULT_COLLECTIONS_FILE = Path("configs/collections.yaml")
DEFAULT_COLLECTION = "consumer_law"


class CollectionConfigFiles(StrictModel):
    ingest: Path
    chunking: Path
    retrieval: Path
    generation: Path
    llm: Path = Path("configs/llm.yaml")


class CollectionRoots(StrictModel):
    raw_dir: Path
    processed_dir: Path
    eval_dir: Path
    results_dir: Path
    feedback_file: Path | None = None  # default: <parent of eval_dir>/feedback.jsonl


class CollectionUI(StrictModel):
    """Texts of the web UI for one collection."""

    label: str  # selector entry, e.g. "Indian Consumer Law"
    eyebrow: str
    heading: str
    lede: str
    placeholder: str
    disclaimer: str
    abstention_message: str = "Not found in these documents"
    n_examples: int = Field(default=5, ge=0, le=20)


class AutoEvalDefaults(StrictModel):
    n_candidates: int = Field(default=200, gt=0)
    target_answerable: int = Field(default=90, gt=0)
    n_unanswerable_candidates: int = Field(default=90, ge=0)
    target_unanswerable: int = Field(default=30, ge=0)


class PipelineDefaults(StrictModel):
    """What scripts/run_collection_pipeline.py runs for this collection."""

    index_models: list[str] = Field(default_factory=lambda: ["bge-small", "bge-base"])
    chroma: bool = False
    dev_systems: list[str] = Field(default_factory=lambda: ["G2", "G4"])
    test_systems: list[str] = Field(default_factory=lambda: ["G1", "G2", "G3", "G4"])


class CollectionSpec(StrictModel):
    title: str
    profile: str = "consumer_law"
    configs: CollectionConfigFiles
    roots: CollectionRoots
    ui: CollectionUI
    auto_eval: AutoEvalDefaults = Field(default_factory=AutoEvalDefaults)
    pipeline: PipelineDefaults = Field(default_factory=PipelineDefaults)
    notes: dict[str, str] = Field(default_factory=dict)


class CollectionsFile(StrictModel):
    default: str = DEFAULT_COLLECTION
    ui_default: str | None = None  # collection selected first in the web UI
    collections: dict[str, CollectionSpec]

    @model_validator(mode="after")
    def _known_defaults(self) -> CollectionsFile:
        for name in (self.default, self.ui_default):
            if name is not None and name not in self.collections:
                raise ValueError(f"unknown collection {name!r} (known: {sorted(self.collections)})")
        return self


def _posix(path: Path) -> str:
    return PurePosixPath(*path.parts).as_posix() if not path.is_absolute() else path.as_posix()


@dataclass(frozen=True)
class Collection:
    """A resolved collection. Paths are repository-relative (as written in the YAML); use
    :func:`ragsys.config.resolve_path` (or :meth:`abs`) for absolute paths."""

    name: str
    spec: CollectionSpec

    # ---------------------------------------------------------------- basics

    @property
    def title(self) -> str:
        return self.spec.title

    @property
    def profile(self) -> str:
        return self.spec.profile

    @property
    def ingest_config(self) -> Path:
        return self.spec.configs.ingest

    @property
    def chunking_config(self) -> Path:
        return self.spec.configs.chunking

    @property
    def retrieval_config(self) -> Path:
        return self.spec.configs.retrieval

    @property
    def generation_config(self) -> Path:
        return self.spec.configs.generation

    @property
    def llm_config(self) -> Path:
        return self.spec.configs.llm

    # ---------------------------------------------------------------- roots

    @property
    def raw_dir(self) -> Path:
        return self.spec.roots.raw_dir

    @property
    def processed_dir(self) -> Path:
        return self.spec.roots.processed_dir

    @property
    def eval_dir(self) -> Path:
        return self.spec.roots.eval_dir

    @property
    def results_dir(self) -> Path:
        return self.spec.roots.results_dir

    # ---------------------------------------------------------------- derived locations

    @property
    def sources_csv(self) -> Path:
        return self.raw_dir / "SOURCES.csv"

    @property
    def clean_dir(self) -> Path:
        return self.processed_dir / "clean"

    @property
    def structure_dir(self) -> Path:
        return self.processed_dir / "structure"

    @property
    def nodes_file(self) -> Path:
        return self.structure_dir / "nodes.jsonl"

    @property
    def chunks_dir(self) -> Path:
        return self.processed_dir / "chunks"

    @property
    def emb_dir(self) -> Path:
        return self.processed_dir / "emb"

    @property
    def chroma_dir(self) -> Path:
        return self.processed_dir / "chroma"

    @property
    def semantic_cache_dir(self) -> Path:
        return self.processed_dir / "cache"

    @property
    def llm_cache_dir(self) -> Path:
        return self.processed_dir / "llm_cache"

    @property
    def rerank_cache_dir(self) -> Path:
        return self.processed_dir / "rerank_cache"

    @property
    def nli_cache_dir(self) -> Path:
        return self.processed_dir / "nli_cache"

    @property
    def questions_file(self) -> Path:
        return self.eval_dir / "questions.jsonl"

    @property
    def drafts_file(self) -> Path:
        return self.eval_dir / "drafts.jsonl"

    @property
    def tables_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def retrieval_results_dir(self) -> Path:
        return self.results_dir / "retrieval"

    @property
    def generation_results_dir(self) -> Path:
        return self.results_dir / "generation"

    @property
    def test_access_log(self) -> Path:
        return self.results_dir / "test_access_log.json"

    @property
    def threshold_file(self) -> Path:
        return self.tables_dir / "abstention_threshold.json"

    @property
    def selected_configs_file(self) -> Path:
        return self.tables_dir / "selected_configs.json"

    @property
    def eval_construction_file(self) -> Path:
        return self.tables_dir / "eval_construction.json"

    @property
    def summary_file(self) -> Path:
        return self.tables_dir / "SUMMARY.md"

    @property
    def pipeline_log(self) -> Path:
        return self.results_dir / "pipeline_log.txt"

    @property
    def pipeline_state(self) -> Path:
        return self.results_dir / "pipeline_state.json"

    @property
    def feedback_file(self) -> Path:
        explicit = self.spec.roots.feedback_file
        return explicit if explicit is not None else self.eval_dir.parent / "feedback.jsonl"

    PATH_NAMES = (
        "ingest_config",
        "chunking_config",
        "retrieval_config",
        "generation_config",
        "llm_config",
        "raw_dir",
        "processed_dir",
        "eval_dir",
        "results_dir",
        "sources_csv",
        "clean_dir",
        "structure_dir",
        "nodes_file",
        "chunks_dir",
        "emb_dir",
        "chroma_dir",
        "semantic_cache_dir",
        "llm_cache_dir",
        "rerank_cache_dir",
        "nli_cache_dir",
        "questions_file",
        "drafts_file",
        "tables_dir",
        "retrieval_results_dir",
        "generation_results_dir",
        "test_access_log",
        "threshold_file",
        "selected_configs_file",
        "eval_construction_file",
        "summary_file",
        "pipeline_log",
        "pipeline_state",
        "feedback_file",
    )

    def paths(self) -> dict[str, str]:
        """Every location of this collection as a repository-relative POSIX string."""
        return {name: _posix(getattr(self, name)) for name in self.PATH_NAMES}

    @staticmethod
    def abs(path: Path) -> Path:
        return resolve_path(path)

    # ---------------------------------------------------------------- config loading

    def llm(self) -> Any:
        """configs/llm.yaml (or the collection's LLM config) with this collection's cache."""
        from ragsys.llm.client import load_llm_config

        cfg = load_llm_config(resolve_path(self.llm_config))
        return cfg.model_copy(update={"cache_dir": self.llm_cache_dir})


def load_collections(path: str | Path = DEFAULT_COLLECTIONS_FILE) -> CollectionsFile:
    return load_yaml_config(resolve_path(Path(path)), CollectionsFile)


def get_collection(
    name: str | None = None, path: str | Path = DEFAULT_COLLECTIONS_FILE
) -> Collection:
    reg = load_collections(path)
    key = name or reg.default
    if key not in reg.collections:
        raise KeyError(f"unknown collection {key!r}; known: {', '.join(sorted(reg.collections))}")
    return Collection(key, reg.collections[key])


def all_collections(path: str | Path = DEFAULT_COLLECTIONS_FILE) -> list[Collection]:
    reg = load_collections(path)
    return [Collection(k, v) for k, v in reg.collections.items()]


# --------------------------------------------------------------------------- argparse helpers


def add_collection_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--collection",
        default=None,
        help=f"collection name from {DEFAULT_COLLECTIONS_FILE.as_posix()} (default: {DEFAULT_COLLECTION})",
    )
    ap.add_argument(
        "--collections-file",
        default=str(DEFAULT_COLLECTIONS_FILE),
        help="collection registry (default: %(default)s)",
    )


def collection_from_args(args: argparse.Namespace) -> Collection:
    try:
        return get_collection(args.collection, args.collections_file)
    except KeyError as exc:
        raise SystemExit(str(exc.args[0])) from exc


def config_path(explicit: str | None, default: Path) -> Path:
    """An explicitly passed --config path, else the collection's config file."""
    return Path(explicit) if explicit else default


# --------------------------------------------------------------------------- consistency


def consistency_errors(coll: Collection) -> list[str]:
    """Paths inside the collection's config files that disagree with its roots."""
    from ragsys.agents.config import load_generation_settings
    from ragsys.config import load_chunking_config, load_ingest_config
    from ragsys.retrieval.config import load_retrieval_settings

    errs: list[str] = []

    def same(what: str, got: Path | None, want: Path) -> None:
        g = _posix(got) if got is not None else "None"
        if g != _posix(want):
            errs.append(f"{coll.name}: {what} is {g}, expected {_posix(want)}")

    ing = load_ingest_config(resolve_path(coll.ingest_config))
    same("ingest paths.raw_dir", ing.paths.raw_dir, coll.raw_dir)
    same("ingest paths.processed_dir", ing.paths.processed_dir, coll.processed_dir)
    ch = load_chunking_config(resolve_path(coll.chunking_config))
    same("chunking paths.processed_dir", ch.paths.processed_dir, coll.processed_dir)
    same("chunking paths.cache_dir", ch.paths.cache_dir, coll.semantic_cache_dir)
    rs = load_retrieval_settings(resolve_path(coll.retrieval_config))
    same("retrieval paths.processed_dir", rs.paths.processed_dir, coll.processed_dir)
    same("retrieval paths.emb_dir", rs.paths.emb_dir, coll.emb_dir)
    same("retrieval paths.chroma_dir", rs.paths.chroma_dir, coll.chroma_dir)
    same("retrieval paths.results_dir", rs.paths.results_dir, coll.results_dir)
    same("retrieval paths.questions", rs.paths.questions, coll.questions_file)
    gen = load_generation_settings(resolve_path(coll.generation_config))
    same("generation retrieval_config", gen.retrieval_config, coll.retrieval_config)
    same("generation llm_config", gen.llm_config, coll.llm_config)
    same("generation results_dir", gen.results_dir, coll.results_dir)
    same("generation nli.cache_dir", gen.nli.cache_dir, coll.nli_cache_dir)
    same("generation abstention.threshold_file", gen.abstention.threshold_file, coll.threshold_file)
    same("generation ui.feedback_file", gen.ui.feedback_file, coll.feedback_file)
    same("generation ui.sources_csv", gen.ui.sources_csv, coll.sources_csv)
    same("generation LLM cache_dir", gen.llm().cache_dir, coll.llm_cache_dir)
    if gen.profile != coll.profile:
        errs.append(f"{coll.name}: generation profile {gen.profile!r} != {coll.profile!r}")
    return errs
