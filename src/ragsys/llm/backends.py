"""Named generator backends (configs/llm.yaml).

::

    default_backend: local
    backends:
      local:   {base_url: http://localhost:11434/v1, model: qwen2.5:3b, ...}   # Ollama
      hosted:  {base_url: https://api.groq.com/openai/v1, model: ..., requires_key: true, ...}

A backend holds the client settings of :class:`ragsys.llm.client.LLMConfig` plus what depends on
the generator: its context budgets for the grounded pipeline (the local model keeps 1024 /
2048 tokens), its context window, chat settings, per-backend generation overrides (applied over
a collection's ``llm_overrides``), client-side rate limits and where the model choice is
recorded (``model_selected_on``, ``model_selection``).

A file without ``backends:`` is the original single-endpoint layout and becomes the backend
``local``. The API key is read from the environment variable ``LLM_API_KEY`` only; it is never
stored on these objects, logged or written anywhere.

Evaluations never fall back from one backend to another: :func:`require_backend` raises
:class:`BackendUnavailable` with the reason, and the scripts stop. Only the interactive UI picks
a default (hosted when the key is set and the endpoint answers, else local) and shows which
model produced every answer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from ragsys.config import StrictModel, resolve_path
from ragsys.llm.client import API_KEY_ENV, LEGACY_BACKEND, LLMClient, LLMConfig

LLM_CONFIG_FIELDS = frozenset(LLMConfig.model_fields)


class BackendSpec(LLMConfig):
    """One generator backend: an LLM endpoint and what depends on its model."""

    label: str = ""  # shown in the UI, e.g. "Hosted: openai/gpt-oss-120b (Groq)"
    provider: str = ""  # e.g. "Ollama (this computer)", "Groq"
    requires_key: bool = False  # needs LLM_API_KEY in the environment
    privacy: str = ""  # one line for the UI ("questions stay on this computer", ...)
    # grounded pipeline (None: the collection's generation config)
    context_budget_tokens: int | None = Field(default=None, gt=0)
    revision_budget_tokens: int | None = Field(default=None, gt=0)
    llm_context_window: int | None = Field(default=None, gt=0)
    generation_overrides: dict[str, Any] = Field(default_factory=dict)
    # chat and general-knowledge answers
    chat_max_tokens: int = Field(default=1024, gt=0)
    chat_temperature: float = Field(default=0.3, ge=0, le=2)
    chat_history_tokens: int = Field(default=3000, gt=0)
    # the provider's daily quota (not enforced by the client: a request over it gets a 429
    # whose long Retry-After stops the run, see max_retry_wait_s); used for time estimates
    tokens_per_day: int | None = Field(default=None, gt=0)
    requests_per_day: int | None = Field(default=None, gt=0)
    # provenance of the model choice
    model_selected_on: str | None = None
    model_selection: str = ""

    @model_validator(mode="after")
    def _budgets(self) -> BackendSpec:
        b, r = self.context_budget_tokens, self.revision_budget_tokens
        if (b is None) != (r is None):
            raise ValueError("set both context_budget_tokens and revision_budget_tokens, or none")
        if b is not None and r is not None and r < b:
            raise ValueError("revision_budget_tokens must be >= context_budget_tokens")
        return self

    def client_config(self, name: str) -> LLMConfig:
        data = {k: v for k, v in self.model_dump().items() if k in LLM_CONFIG_FIELDS}
        data["backend"] = name
        return LLMConfig.model_validate(data)

    def describe(self, name: str) -> dict[str, Any]:
        """What the UI and the result files record about the backend (no secrets)."""
        return {
            "name": name,
            "label": self.label or f"{name}: {self.model}",
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "model_selected_on": self.model_selected_on,
            "privacy": self.privacy,
        }


class LLMFile(StrictModel):
    default_backend: str = LEGACY_BACKEND
    backends: dict[str, BackendSpec]

    @model_validator(mode="after")
    def _known_default(self) -> LLMFile:
        if self.default_backend not in self.backends:
            raise ValueError(
                f"default_backend {self.default_backend!r} is not one of {sorted(self.backends)}"
            )
        return self

    @property
    def names(self) -> list[str]:
        return list(self.backends)

    def spec(self, name: str | None = None) -> BackendSpec:
        key = name or self.default_backend
        if key not in self.backends:
            raise KeyError(f"unknown backend {key!r}; configured: {', '.join(self.backends)}")
        return self.backends[key]

    def client_config(self, name: str | None = None) -> LLMConfig:
        key = name or self.default_backend
        return self.spec(key).client_config(key)


def load_backends(path: str | Path) -> LLMFile:
    """Read configs/llm.yaml (either layout)."""
    p = resolve_path(Path(path))
    with p.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "backends" not in data:
        return LLMFile(
            default_backend=LEGACY_BACKEND, backends={LEGACY_BACKEND: BackendSpec(**data)}
        )
    return LLMFile.model_validate(data)


class BackendUnavailable(RuntimeError):
    """A requested backend cannot be used (missing key or unreachable endpoint)."""


def key_present() -> bool:
    return bool(os.environ.get(API_KEY_ENV))


def unavailable_reason(spec: BackendSpec, name: str, client: Any | None = None) -> str | None:
    """Why ``name`` cannot answer now, or None. Checks the key first (no network call when it
    is missing), then the endpoint's model listing. Fakes (no ``is_reachable``) are available."""
    if spec.requires_key and not key_present():
        return (
            f"backend {name!r} ({spec.model} at {spec.base_url}) needs the environment variable "
            f'{API_KEY_ENV}; set it in this shell (PowerShell: $env:{API_KEY_ENV} = "...")'
        )
    probe = client if client is not None else LLMClient(spec.client_config(name), use_cache=False)
    check = getattr(probe, "is_reachable", None)
    if check is not None and not check():
        hint = (
            f"start Ollama and run `ollama pull {spec.model}`"
            if not spec.requires_key
            else "check the network connection and the key"
        )
        return f"backend {name!r}: the endpoint {spec.base_url} is not reachable; {hint}"
    return None


def require_backend(spec: BackendSpec, name: str, client: Any | None = None) -> None:
    """Raise :class:`BackendUnavailable` when the backend cannot answer (evaluations call this;
    they never switch to another backend)."""
    reason = unavailable_reason(spec, name, client)
    if reason:
        raise BackendUnavailable(reason)
