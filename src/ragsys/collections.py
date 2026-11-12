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

Evaluated collections freeze the documents of their evaluations in ``eval_docs``; ``app_docs``
is the superset served by ask.py and the web UI. When the two differ, the collection has a
second index, the *app index* (``app_index``: its own ingest/chunking/retrieval configs and its
own processed directory), so the evaluation corpus is never rebuilt or touched. Collections that
are not evaluated (``evaluated: false``) have one index, which is an app index. Every evaluation
script refuses an app index (:func:`eval_index_refusal`).
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
    # example questions shown as chips; when empty, they come from the dev split
    examples: list[str] = Field(default_factory=list)
    # domain safety note, shown in the UI and added to the chat / general-knowledge prompts
    safety_note: str = ""
    # what the collection is about, for the chat prompt ("Indian law", "human biology", ...)
    domain: str = ""
    # chat uses the coding assistant prompt by default (cs_ai)
    code_assistant: bool = False


class AppIndexSpec(StrictModel):
    """The app index of an evaluated collection whose app_docs extend its eval_docs."""

    ingest: Path
    chunking: Path
    retrieval: Path
    processed_dir: Path
    index_models: list[str] = Field(default_factory=lambda: ["bge-small"])


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
    # generator-scale ablation on the TEST split (run by scripts/run_app_collections.py)
    ablation_systems: list[str] = Field(default_factory=list)


class CollectionSpec(StrictModel):
    title: str
    profile: str = "consumer_law"
    configs: CollectionConfigFiles
    roots: CollectionRoots
    ui: CollectionUI
    auto_eval: AutoEvalDefaults = Field(default_factory=AutoEvalDefaults)
    pipeline: PipelineDefaults = Field(default_factory=PipelineDefaults)
    notes: dict[str, str] = Field(default_factory=dict)
    evaluated: bool = True  # False: served in the app only (no evaluation set, no results)
    eval_docs: list[str] = Field(default_factory=list)  # frozen documents of the evaluations
    app_docs: list[str] = Field(default_factory=list)  # documents served in the app
    app_index: AppIndexSpec | None = None  # when app_docs extend eval_docs

    @model_validator(mode="after")
    def _docs(self) -> CollectionSpec:
        if not self.evaluated and (self.eval_docs or self.app_index is not None):
            raise ValueError("a collection that is not evaluated has no eval_docs or app_index")
        if not self.app_docs:  # not declared: the app serves the evaluation documents
            self.app_docs = list(self.eval_docs)
        missing = [d for d in self.eval_docs if d not in self.app_docs]
        if missing:
            raise ValueError(f"app_docs must include every eval doc (missing {missing})")
        differ = self.evaluated and set(self.app_docs) != set(self.eval_docs)
        if differ and self.app_index is None:
            raise ValueError("app_docs differ from eval_docs: an app_index is required")
        if not differ and self.app_index is not None:
            raise ValueError("app_index is only used when app_docs extend eval_docs")
        return self


class CollectionsFile(StrictModel):
    default: str = DEFAULT_COLLECTION
    ui_default: str | None = None  # collection selected first in the web UI
    ui_order: list[str] = Field(default_factory=list)  # selector order (default: file order)
    collections: dict[str, CollectionSpec]

    @model_validator(mode="after")
    def _known_defaults(self) -> CollectionsFile:
        for name in (self.default, self.ui_default, *self.ui_order):
            if name is not None and name not in self.collections:
                raise ValueError(f"unknown collection {name!r} (known: {sorted(self.collections)})")
        return self

    def ordered(self) -> list[str]:
        """Collection names in the web UI's order (ui_order first, then the rest)."""
        return [*self.ui_order, *[n for n in self.collections if n not in self.ui_order]]


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

    # ---------------------------------------------------------------- index variants

    @property
    def evaluated(self) -> bool:
        return self.spec.evaluated

    @property
    def has_app_index(self) -> bool:
        """True when the app serves a separate index (app_docs extend eval_docs)."""
        return self.spec.app_index is not None

    @property
    def app_ingest_config(self) -> Path:
        a = self.spec.app_index
        return a.ingest if a is not None else self.ingest_config

    @property
    def app_chunking_config(self) -> Path:
        a = self.spec.app_index
        return a.chunking if a is not None else self.chunking_config

    @property
    def app_retrieval_config(self) -> Path:
        a = self.spec.app_index
        return a.retrieval if a is not None else self.retrieval_config

    @property
    def app_processed_dir(self) -> Path:
        a = self.spec.app_index
        return a.processed_dir if a is not None else self.processed_dir

    @property
    def app_index_models(self) -> list[str]:
        a = self.spec.app_index
        return list(a.index_models) if a is not None else list(self.spec.pipeline.index_models)

    APP_PATH_NAMES = (
        "app_ingest_config",
        "app_chunking_config",
        "app_retrieval_config",
        "app_processed_dir",
    )

    def paths(self) -> dict[str, str]:
        """Every location of this collection as a repository-relative POSIX string (with the
        app index's configs and processed directory when the collection has one)."""
        out = {name: _posix(getattr(self, name)) for name in self.PATH_NAMES}
        if self.has_app_index:
            out.update({name: _posix(getattr(self, name)) for name in self.APP_PATH_NAMES})
        return out

    @staticmethod
    def abs(path: Path) -> Path:
        return resolve_path(path)

    # ---------------------------------------------------------------- config loading

    def llm(self, backend: str | None = None) -> Any:
        """A backend of configs/llm.yaml (default: its default backend) with this collection's
        cache."""
        from ragsys.llm.client import load_llm_config

        cfg = load_llm_config(resolve_path(self.llm_config), backend)
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

    # ---- frozen evaluation corpus vs app documents
    enabled = [d.doc_id for d in ing.enabled_documents()]
    want_index = "eval" if coll.evaluated else "app"
    if rs.index != want_index:
        errs.append(f"{coll.name}: retrieval index is {rs.index!r}, expected {want_index!r}")
    if coll.evaluated and coll.spec.eval_docs and enabled != list(coll.spec.eval_docs):
        errs.append(f"{coll.name}: eval_docs {coll.spec.eval_docs} != ingest documents {enabled}")
    if not coll.has_app_index:
        if coll.spec.app_docs and enabled != list(coll.spec.app_docs):
            errs.append(f"{coll.name}: app_docs {coll.spec.app_docs} != ingest documents {enabled}")
        return errs
    app_ing = load_ingest_config(resolve_path(coll.app_ingest_config))
    same("app ingest paths.raw_dir", app_ing.paths.raw_dir, coll.raw_dir)
    same("app ingest paths.processed_dir", app_ing.paths.processed_dir, coll.app_processed_dir)
    app_ch = load_chunking_config(resolve_path(coll.app_chunking_config))
    same("app chunking paths.processed_dir", app_ch.paths.processed_dir, coll.app_processed_dir)
    same("app chunking paths.cache_dir", app_ch.paths.cache_dir, coll.app_processed_dir / "cache")
    app_rs = load_retrieval_settings(resolve_path(coll.app_retrieval_config))
    same("app retrieval paths.processed_dir", app_rs.paths.processed_dir, coll.app_processed_dir)
    same("app retrieval paths.emb_dir", app_rs.paths.emb_dir, coll.app_processed_dir / "emb")
    if app_rs.index != "app":
        errs.append(f"{coll.name}: the app retrieval config must say index: app")
    app_enabled = [d.doc_id for d in app_ing.enabled_documents()]
    if app_enabled != list(coll.spec.app_docs):
        errs.append(f"{coll.name}: app_docs {coll.spec.app_docs} != app ingest {app_enabled}")
    eval_entries = {d.doc_id: d.model_dump() for d in ing.enabled_documents()}
    for d in app_ing.enabled_documents():
        if d.doc_id in eval_entries and d.model_dump() != eval_entries[d.doc_id]:
            errs.append(f"{coll.name}: app ingest entry of {d.doc_id} differs from the eval one")
    return errs


class AppIndexRefused(PermissionError):
    """An evaluation was asked to run on an app index (or a collection that is not evaluated)."""


def eval_index_refusal(coll: Collection | None, retrieval_settings: Any) -> str | None:
    """Why an evaluation script must not run with these settings, or None. Evaluations use the
    frozen evaluation index only: never an app index, never a collection that is not evaluated."""
    if coll is not None and not coll.evaluated:
        return (
            f"collection {coll.name!r} is served in the app only and is not evaluated; "
            "evaluation scripts do not run on it"
        )
    if getattr(retrieval_settings, "index", "eval") == "app":
        return (
            "these retrieval settings describe an app index (index: app), which is served by "
            "ask.py and the web UI only; evaluations use the frozen evaluation index"
        )
    return None
