"""Helpers for the agent / generation tests: hand-made chunks and rankings, a stub retriever,
scripted fake LLMs and generation settings that only use fake models."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from ragsys.agents import critic as critic_mod
from ragsys.agents import synthesizer as synth_mod
from ragsys.agents.config import GenerationSettings
from ragsys.chunking.base import Chunk
from ragsys.llm.client import LLMResponse, Message
from ragsys.retrieval.config import parse_config_id
from ragsys.retrieval.fusion import ComponentHit
from ragsys.retrieval.pipeline import RetrievalResult, RetrievedChunk


def make_chunk(i: int, text: str, n_tokens: int = 100, doc: str = "testact") -> Chunk:
    return Chunk(
        chunk_id=f"structural:{doc}:{i:05d}",
        chunker="structural",
        doc_id=doc,
        text=f"Test Act > Section {i}\n{text}",
        display_text=text,
        node_ids=[f"{doc}:s{i}", f"{doc}:s{i}(1)"],
        primary_node_id=f"{doc}:s{i}",
        section_ids=[f"{doc}:s{i}"],
        heading_path=f"Test Act > Section {i} - Heading {i}",
        page_start=i,
        page_end=i + 1,
        char_start=i * 1000,
        char_end=i * 1000 + len(text),
        n_tokens=n_tokens,
        xref_targets=[],
    )


TEXTS = [
    "Every complaint shall be decided within a period of three months where no testing of goods is required.",
    "Where testing of goods is required, the complaint shall be decided within five months.",
    "An appeal may be filed within forty-five days before the State Commission.",
    "The District Commission has jurisdiction where the value does not exceed fifty lakh rupees.",
    "A fee of Rs. 5,00,000 is payable for complaints above one crore rupees.",
    "The Central Government may by notification prescribe other limits.",
]


def ranking(
    texts: Sequence[str] = TEXTS, tokens: Sequence[int] | None = None, top_score: float = 0.8
) -> list[RetrievedChunk]:
    out = []
    for i, t in enumerate(texts, start=1):
        n = tokens[i - 1] if tokens else 100
        score = round(top_score - 0.05 * (i - 1), 4)
        out.append(
            RetrievedChunk(
                rank=i,
                chunk=make_chunk(i, t, n),
                provenance="first_stage",
                first_stage_rank=i,
                first_stage_score=score,
                components={"dense": ComponentHit(i, score)},
            )
        )
    return out


class StubRetriever:
    """Returns a fixed ranking; ``scores`` maps question -> top score (default 0.8)."""

    def __init__(
        self,
        texts: Sequence[str] = TEXTS,
        tokens: Sequence[int] | None = None,
        scores: dict[str, float] | None = None,
        config_id: str = "structural|dense|bge-base",
    ) -> None:
        self.texts = list(texts)
        self.tokens = tokens
        self.scores = scores or {}
        self.cfg = parse_config_id(config_id)
        self.config_id = config_id
        self.calls = 0

    def search(self, query: str) -> RetrievalResult:
        self.calls += 1
        ranked = ranking(self.texts, self.tokens, self.scores.get(query, 0.8))
        return RetrievalResult(
            query, self.config_id, ranked, 10, {"dense_encode": 5.0, "total": 7.5}
        )


# ---------------------------------------------------------------- scripted LLM replies


def synth_reply(answer: str = "", citations: Sequence[str] = (), abstain: bool = False) -> str:
    return json.dumps({"abstain": abstain, "answer": answer, "citations": list(citations)})


def critic_reply(
    verdict: str = "accept", unsupported: Sequence[str] = (), feedback: str = ""
) -> str:
    return json.dumps(
        {"verdict": verdict, "unsupported_sentences": list(unsupported), "feedback": feedback}
    )


def role_of(messages: Sequence[Message]) -> str:
    system = messages[0]["content"]
    if system.startswith(critic_mod.SYSTEM_PROMPT[:40]):
        return "critic"
    if system.startswith(synth_mod.SYSTEM_PROMPT[:40]):
        return "synthesizer"
    return "other"


class ScriptedLLM:
    """Replies from per-role queues (the last reply repeats); records every call."""

    model = "scripted"

    def __init__(
        self,
        synth: Sequence[str] = (),
        critic: Sequence[str] = (),
        cached_latency_ms: float | None = None,
    ) -> None:
        self.queues: dict[str, list[str]] = {"synthesizer": list(synth), "critic": list(critic)}
        self.used: dict[str, int] = {"synthesizer": 0, "critic": 0}
        self.calls: list[tuple[str, list[Message]]] = []
        self.cached_latency_ms = cached_latency_ms

    def chat(self, messages: Sequence[Message], *, json_mode: bool = False) -> LLMResponse:
        role = role_of(messages)
        self.calls.append((role, list(messages)))
        q = self.queues.get(role) or ['{"unexpected": true}']
        i = min(self.used.get(role, 0), len(q) - 1)
        self.used[role] = self.used.get(role, 0) + 1
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        if self.cached_latency_ms is not None:
            return LLMResponse(
                q[i],
                self.model,
                usage,
                latency_ms=0.0,
                cached=True,
                stored_latency_ms=self.cached_latency_ms,
            )
        return LLMResponse(q[i], self.model, usage, latency_ms=0.0)

    def chat_json(self, messages: Sequence[Message]) -> tuple[dict[str, Any], LLMResponse]:
        resp = self.chat(messages, json_mode=True)
        return json.loads(resp.content), resp

    def n(self, role: str) -> int:
        return sum(1 for r, _m in self.calls if r == role)


# ---------------------------------------------------------------- settings for script tests


def generation_settings_dict(tmp: Path, retrieval_yaml: Path) -> dict[str, Any]:
    llm_yaml = tmp / "llm.yaml"
    llm_yaml.write_text(
        yaml.safe_dump(
            {
                "base_url": "http://llm.invalid/v1",
                "model": "fake",
                "cache_dir": str(tmp / "llm_cache"),
            }
        ),
        encoding="utf-8",
    )
    return {
        "llm_config": str(llm_yaml),
        "llm_overrides": {"temperature": 0.0},
        "retrieval_config": str(retrieval_yaml),
        "results_dir": str(tmp / "results"),
        "context": {"budget_tokens": 300, "revision_budget_tokens": 600},
        "systems": {
            "G1": {"description": "naive", "retrieval": "fixed_64|dense|fake-a"},
            "G2": {"description": "structural", "retrieval": "structural|dense|fake-a"},
            "G3": {"description": "G2 + guards", "derived_from": "G2", "guards": True},
            "G4": {
                "description": "guards + critic",
                "retrieval": "structural|dense|fake-a",
                "guards": True,
                "critic": True,
            },
            "G5": {
                "description": "G4 + threshold",
                "derived_from": "G4",
                "guards": True,
                "critic": True,
                "threshold": True,
            },
            "G6": {
                "description": "hybrid + rerank",
                "retrieval": "structural|hybrid_linear|fake-a|a0.5|rerank-fake",
                "guards": True,
                "critic": True,
            },
        },
        "abstention": {
            "threshold_file": str(tmp / "results" / "tables" / "abstention_threshold.json")
        },
        "nli": {
            "model": "fake:overlap-nli",
            "cache_dir": str(tmp / "nli_cache"),
            "window_tokens": 50,
            "window_stride": 40,
        },
        "eval": {"bootstrap_resamples": 200, "seed": 7, "reference_system": "G1"},
        "ui": {
            "feedback_file": str(tmp / "feedback.jsonl"),
            "sources_csv": str(tmp / "SOURCES.csv"),
        },
    }


def write_generation_settings(tmp: Path, retrieval_yaml: Path) -> tuple[Path, GenerationSettings]:
    data = generation_settings_dict(tmp, retrieval_yaml)
    path = tmp / "generation.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path, GenerationSettings.model_validate(data)


Responder = Callable[[Sequence[Message]], str]


def write_two_backend_llm(tmp: Path, hosted_budgets: tuple[int, int] = (400, 800)) -> Path:
    """An llm.yaml with a "local" and a "hosted" backend (fake endpoints, no network)."""
    path = tmp / "llm_backends.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "default_backend": "local",
                "backends": {
                    "local": {
                        "base_url": "http://local.invalid/v1",
                        "model": "fake",
                        "cache_dir": str(tmp / "llm_cache"),
                    },
                    "hosted": {
                        "base_url": "http://hosted.invalid/v1",
                        "model": "big-model",
                        "requires_key": True,
                        "label": "Hosted: big-model",
                        "provider": "Test provider",
                        "model_selected_on": "2026-11-12",
                        "context_budget_tokens": hosted_budgets[0],
                        "revision_budget_tokens": hosted_budgets[1],
                        "generation_overrides": {"max_tokens": 1024},
                        "cache_dir": str(tmp / "llm_cache"),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path
