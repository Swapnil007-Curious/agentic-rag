"""Generation configuration (``configs/generation.yaml``): context budgets, loop bounds, the
system definitions G1-G6, abstention, NLI-based answer evaluation and the local web UI.

A system either runs its own retrieval (``retrieval`` = a retrieval config id) or is *derived*
from another system's outputs without new LLM calls (``derived_from``). Two derivations exist:

* adding the deterministic guards to a synthesizer-only system (G3 = G2 + guards): a guard
  failure turns the answer into an abstention;
* adding the retrieval-score threshold (G5 = G4 + threshold): questions whose abstention score is
  below the dev-tuned threshold are abstained before any LLM call.

``retrieval`` may also be ``selected:<role>`` (e.g. ``selected:best_overall``): the retrieval
config chosen on dev by scripts/select_configs.py, read from
``<results_dir>/tables/selected_configs.json``. Evaluations require that file; interactive use
(ask.py, the web UI) may fall back to the first ``selection_fallback`` config whose embeddings
are built (see :meth:`ragsys.agents.engine.AnswerEngine.retrieval_config`).

Generator backends (configs/llm.yaml, :mod:`ragsys.llm.backends`): a system runs on the default
backend (``local``, qwen2.5:3b) unless it names another one (``backend: hosted``; G2H and G4H, the
generator-scale ablation, are G2 and G4 on the hosted backend and say so in ``ablation_of``). The
LLM settings of a backend are its client config with the collection's ``llm_overrides`` and,
for non-default backends, the backend's own ``generation_overrides``; its context budgets and
context window replace the collection's when the backend defines them. Nothing changes for the
default backend, so the fingerprints of the existing systems stay the same.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from ragsys.config import StrictModel, load_yaml_config, resolve_path
from ragsys.llm.backends import LLMFile, load_backends
from ragsys.llm.client import LLMConfig
from ragsys.retrieval.config import RetrievalConfig, parse_config_id


class ContextSettings(StrictModel):
    budget_tokens: int = Field(default=1024, gt=0)  # first synthesis round
    revision_budget_tokens: int = Field(default=2048, gt=0)  # revision round (extended context)
    min_passages: int = Field(default=1, ge=0)  # packed even if larger than the budget

    @model_validator(mode="after")
    def _revision_not_smaller(self) -> ContextSettings:
        if self.revision_budget_tokens < self.budget_tokens:
            raise ValueError("revision_budget_tokens must be >= budget_tokens")
        return self


class LoopSettings(StrictModel):
    max_synth_calls: int = Field(default=2, ge=1, le=2)  # synthesis rounds per question
    max_critic_calls: int = Field(default=2, ge=0, le=2)  # critic rounds per question
    # Round 1 with failed guards: revise from the guard reasons alone (saves one LLM call).
    critic_on_guard_fail: bool = False
    # Final round, guards pass but the critic still says "revise": abstain or release.
    on_final_revise: Literal["abstain", "answer"] = "abstain"


SELECTED_PREFIX = "selected:"


class SelectionMissing(FileNotFoundError):
    """A system uses ``selected:<role>`` but select_configs.py has not written that role yet."""


class SystemSpec(StrictModel):
    description: str
    retrieval: str | None = None  # retrieval config id (runnable system)
    derived_from: str | None = None  # computed from that system's outputs (no LLM calls)
    guards: bool = False  # a guard failure leads to revision (with critic) or abstention
    critic: bool = False  # critic + bounded revise loop
    threshold: bool = False  # retrieval-score abstention before any LLM call
    backend: str | None = None  # generator backend (None: the default backend of llm.yaml)
    ablation_of: str | None = None  # the default-backend system this one mirrors (G2H -> G2)

    @model_validator(mode="after")
    def _one_source(self) -> SystemSpec:
        if (self.retrieval is None) == (self.derived_from is None):
            raise ValueError("a system needs exactly one of 'retrieval' or 'derived_from'")
        if self.derived_from is not None and self.backend is not None:
            raise ValueError("a derived system uses its base system's backend")
        if self.critic and not self.guards:
            raise ValueError("the critic loop always runs the deterministic guards")
        if self.retrieval is not None and not self.retrieval.startswith(SELECTED_PREFIX):
            parse_config_id(self.retrieval)  # fail early on a bad id
        return self


class AbstentionSettings(StrictModel):
    threshold_file: Path = Path("results/tables/abstention_threshold.json")
    tuned_system: str | None = "G5"  # the system whose threshold is tuned on dev (None: none)


class NLISettings(StrictModel):
    model: str = "cross-encoder/nli-deberta-v3-base"
    fallback_model: str = "cross-encoder/nli-distilroberta-base"
    entailment_threshold: float = Field(default=0.5, gt=0, lt=1)
    window_tokens: int = Field(default=400, gt=16)
    window_stride: int = Field(default=300, gt=0)
    batch_size: int = Field(default=8, gt=0)
    cache_dir: Path = Path("data/processed/nli_cache")


class AnswerEvalSettings(StrictModel):
    bootstrap_resamples: int = Field(default=1000, gt=0)
    seed: int = 42
    reference_system: str = "G1"


class UISettings(StrictModel):
    port: int = Field(default=8000, gt=0, lt=65536)
    systems: list[str] = Field(default_factory=lambda: ["G2", "G4", "G5", "G6"])
    default_system: str = "G5"
    feedback_file: Path = Path("data/feedback.jsonl")
    sources_csv: Path = Path("data/raw/SOURCES.csv")


class GenerationSettings(StrictModel):
    profile: str = "consumer_law"  # prompt wording (ragsys.profiles)
    llm_config: Path = Path("configs/llm.yaml")
    llm_overrides: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.0})
    # Context window of the served model (Ollama: OLLAMA_CONTEXT_LENGTH); a warning is recorded
    # when a prompt plus max_tokens would come close to it. null disables the check.
    llm_context_window: int | None = Field(default=4096, gt=0)
    retrieval_config: Path = Path("configs/retrieval.yaml")
    results_dir: Path = Path("results")
    seed: int = 42
    context: ContextSettings = Field(default_factory=ContextSettings)
    loop: LoopSettings = Field(default_factory=LoopSettings)
    systems: dict[str, SystemSpec]
    abstention: AbstentionSettings = Field(default_factory=AbstentionSettings)
    # interactive fallback for "selected:" systems before select_configs.py has run
    selection_fallback: list[str] = Field(default_factory=list)
    # systems of the standard set that this collection does not run, with the reason
    skipped_systems: dict[str, str] = Field(default_factory=dict)
    nli: NLISettings = Field(default_factory=NLISettings)
    eval: AnswerEvalSettings = Field(default_factory=AnswerEvalSettings)
    ui: UISettings = Field(default_factory=UISettings)

    @model_validator(mode="after")
    def _check_systems(self) -> GenerationSettings:
        for name, spec in self.systems.items():
            if spec.derived_from is None:
                continue
            base = self.systems.get(spec.derived_from)
            if base is None:
                raise ValueError(f"{name}: derived_from {spec.derived_from!r} is not defined")
            if base.derived_from is not None:
                raise ValueError(f"{name}: derive from a runnable system, not {base!r}")
            adds_guards = (
                spec.guards and not base.guards and not spec.critic and not base.critic
            ) and spec.threshold == base.threshold
            adds_threshold = (
                spec.threshold
                and not base.threshold
                and (spec.guards, spec.critic) == (base.guards, base.critic)
            )
            if not (adds_guards or adds_threshold):
                raise ValueError(
                    f"{name}: a derived system may only add the guards (to a synthesizer-only "
                    "system) or the retrieval-score threshold to its base system"
                )
        for name in [*self.ui.systems, self.ui.default_system, self.eval.reference_system]:
            if name not in self.systems:
                raise ValueError(f"unknown system {name!r}")
        if self.ui.default_system not in self.ui.systems:
            raise ValueError("ui.default_system must be one of ui.systems")
        tuned = self.abstention.tuned_system
        if tuned is not None and tuned not in self.systems:
            raise ValueError(f"unknown abstention.tuned_system {tuned!r}")
        for name, spec in self.systems.items():
            base = self.systems.get(spec.ablation_of) if spec.ablation_of else None
            if spec.ablation_of is not None and (base is None or base.derived_from is not None):
                raise ValueError(f"{name}: ablation_of must name a runnable system")
        for cid in self.selection_fallback:
            parse_config_id(cid)
        from ragsys.profiles import get_profile

        get_profile(self.profile)  # fail early on an unknown profile
        return self

    # ------------------------------------------------------------------ helpers

    def system(self, name: str) -> SystemSpec:
        if name not in self.systems:
            raise KeyError(f"unknown system {name!r}; known: {', '.join(self.systems)}")
        return self.systems[name]

    def runnable_base(self, name: str) -> str:
        """The system whose retrieval and LLM calls ``name`` uses (itself when runnable)."""
        spec = self.system(name)
        return spec.derived_from or name

    def retrieval_id(self, name: str) -> str:
        rid = self.system(self.runnable_base(name)).retrieval
        assert rid is not None
        return rid

    def retrieval_cfg(self, name: str) -> RetrievalConfig:
        """The retrieval config of a system (``selected:<role>`` is read from the selection
        file; :class:`SelectionMissing` when it is not there yet)."""
        rid = self.retrieval_id(name)
        if rid.startswith(SELECTED_PREFIX):
            return self.selected_retrieval(rid[len(SELECTED_PREFIX) :])
        return parse_config_id(rid)

    def selection_path(self) -> Path:
        return resolve_path(self.results_dir) / "tables" / "selected_configs.json"

    def selected_retrieval(self, role: str) -> RetrievalConfig:
        path = self.selection_path()
        if not path.exists():
            raise SelectionMissing(
                f"{path} not found: run the dev retrieval grid and scripts/select_configs.py "
                f"before using a 'selected:{role}' system"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("selected", []):
            if role in entry.get("roles", []):
                return RetrievalConfig.model_validate(entry["config"])
        raise SelectionMissing(f"no config with role {role!r} in {path}")

    def is_derived(self, name: str) -> bool:
        return self.system(name).derived_from is not None

    # ------------------------------------------------------------------ backends

    def backends(self) -> LLMFile:
        return load_backends(resolve_path(self.llm_config))

    @property
    def default_backend(self) -> str:
        return self.backends().default_backend

    def backend_of(self, name: str) -> str:
        """The backend a system runs on (its base system's for derived systems)."""
        return self.system(self.runnable_base(name)).backend or self.default_backend

    def llm(self, backend: str | None = None) -> LLMConfig:
        """The client config of ``backend`` (default: llm.yaml's default backend) with
        ``llm_overrides`` applied (temperature 0 for generation) and, for other backends, the
        backend's ``generation_overrides`` (e.g. a larger completion limit for a model that
        also spends tokens on hidden reasoning)."""
        reg = self.backends()
        name = backend or reg.default_backend
        data = {**reg.client_config(name).model_dump(), **self.llm_overrides}
        if name != reg.default_backend:
            data.update(reg.spec(name).generation_overrides)
        return LLMConfig.model_validate(data)

    def context_for(self, backend: str | None = None) -> ContextSettings:
        """Context budgets of ``backend``: its own when it defines them, else the collection's."""
        reg = self.backends()
        spec = reg.spec(backend)
        if spec.context_budget_tokens is None or spec.revision_budget_tokens is None:
            return self.context
        return ContextSettings(
            budget_tokens=spec.context_budget_tokens,
            revision_budget_tokens=spec.revision_budget_tokens,
            min_passages=self.context.min_passages,
        )

    def context_window_for(self, backend: str | None = None) -> int | None:
        spec = self.backends().spec(backend)
        return spec.llm_context_window or self.llm_context_window

    def threshold_path(self) -> Path:
        return resolve_path(self.abstention.threshold_file)


def load_generation_settings(path: str | Path) -> GenerationSettings:
    return load_yaml_config(path, GenerationSettings)


def default_generation_settings() -> GenerationSettings:
    """The configuration of configs/generation.yaml (used when no file is given)."""
    return load_generation_settings(resolve_path(Path("configs/generation.yaml")))
