"""Settings of the chat assistant (configs/assistant.yaml) and the gate-threshold rule."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from pydantic import Field

from ragsys.config import StrictModel, load_yaml_config, resolve_path
from ragsys.retrieval.config import DEFAULT_QUERY_INSTRUCTION

DEFAULT_CONFIG = Path("configs/assistant.yaml")
DEFAULT_GATE = 0.62  # bge-small, for collections without DEV score distributions


class WindowLimit(StrictModel):
    """At most ``max`` events per ``window_s`` seconds."""

    max: int = Field(gt=0)
    window_s: float = Field(gt=0)


class GateSettings(StrictModel):
    default_threshold: float = Field(default=DEFAULT_GATE, gt=0, lt=1)
    upload_threshold: float = Field(default=DEFAULT_GATE, gt=0, lt=1)
    percentile: float = Field(default=5.0, gt=0, lt=100)
    thresholds: dict[str, float] = Field(default_factory=dict)
    derived_from: dict[str, Path] = Field(default_factory=dict)
    follow_up_max_words: int = Field(default=12, ge=0)
    search_k: int = Field(default=20, gt=0)
    max_passages: int = Field(default=8, gt=0)

    def threshold(self, collection: str | None) -> float:
        if collection is None:
            return self.default_threshold
        return float(self.thresholds.get(collection, self.default_threshold))


class HistorySettings(StrictModel):
    max_turns: int = Field(default=10, gt=0)
    # per backend; None: the backend's chat_history_tokens in configs/llm.yaml
    budget_tokens: dict[str, int] = Field(default_factory=dict)
    summary_max_tokens: int = Field(default=400, gt=0)
    summary_target_share: float = Field(default=0.5, gt=0, le=1)
    summary_input_chars: int = Field(default=24000, gt=0)


class LimitSettings(StrictModel):
    max_message_chars: int = Field(default=4000, gt=0)
    max_output_tokens: int = Field(default=1500, gt=0)
    max_messages: int = Field(default=100, gt=0)
    max_concurrent_streams: int = Field(default=8, gt=0)
    queue_wait_s: float = Field(default=5.0, ge=0)


class RateLimitSettings(StrictModel):
    per_ip: list[WindowLimit] = Field(
        default_factory=lambda: [
            WindowLimit(max=20, window_s=600),
            WindowLimit(max=200, window_s=86400),
        ]
    )
    uploads_per_ip: list[WindowLimit] = Field(
        default_factory=lambda: [WindowLimit(max=10, window_s=3600)]
    )
    global_daily_requests: int = Field(default=800, gt=0)


class UploadSettings(StrictModel):
    max_files: int = Field(default=3, gt=0)
    max_mb: float = Field(default=20.0, gt=0)
    max_pages: int = Field(default=300, gt=0)
    ttl_s: float = Field(default=7200.0, gt=0)
    max_sessions: int = Field(default=100, gt=0)
    max_chunks_per_file: int = Field(default=4000, gt=0)

    @property
    def max_bytes(self) -> int:
        return int(self.max_mb * 1024 * 1024)


class AssistantSettings(StrictModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    chunker: str = "structural"
    context_tokens: dict[str, int] = Field(default_factory=lambda: {"hosted": 2048, "local": 1024})
    default_context_tokens: int = Field(default=1024, gt=0)
    gate: GateSettings = Field(default_factory=GateSettings)
    history: HistorySettings = Field(default_factory=HistorySettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    rate_limits: RateLimitSettings = Field(default_factory=RateLimitSettings)
    uploads: UploadSettings = Field(default_factory=UploadSettings)
    notices: dict[str, str] = Field(default_factory=dict)
    licence_fallbacks: dict[str, str] = Field(default_factory=dict)
    general_examples: list[str] = Field(default_factory=list)

    def context_budget(self, backend: str) -> int:
        return int(self.context_tokens.get(backend, self.default_context_tokens))

    def notice(self, collection: str | None) -> str:
        return self.notices.get(collection or "", "")


def load_assistant_settings(path: str | Path = DEFAULT_CONFIG) -> AssistantSettings:
    """Read the settings (relative paths are resolved against the repository root)."""
    return load_yaml_config(resolve_path(Path(path)), AssistantSettings)


def answerable_top1_scores(per_query: Path) -> list[float]:
    """Top-1 first-stage scores of the answerable questions of a retrieval per_query.jsonl."""
    out: list[float] = []
    with per_query.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            score = row.get("top1_first_stage_score")
            if row.get("answerable") and score is not None:
                out.append(float(score))
    return out


def derive_gate_threshold(per_query: Path, percentile: float = 5.0) -> float:
    """The gate rule: the ``percentile`` of the answerable questions' top-1 dense cosine,
    rounded down to two decimals."""
    scores = answerable_top1_scores(per_query)
    if not scores:
        raise ValueError(f"{per_query}: no answerable question with a top-1 score")
    value = float(np.percentile(np.asarray(scores, dtype=np.float64), percentile))
    return math.floor(value * 100 + 1e-9) / 100
