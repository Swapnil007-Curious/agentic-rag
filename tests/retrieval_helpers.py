"""Helpers for retrieval / evaluation tests: fake-model settings and a small question set."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ragsys.eval.split import split_of
from ragsys.retrieval.config import RetrievalSettings


def fake_settings(processed: Path, results: Path, questions: Path) -> RetrievalSettings:
    """Settings that only use the dependency-free fake models (no downloads)."""
    return RetrievalSettings.model_validate(
        {
            "paths": {
                "processed_dir": str(processed),
                "emb_dir": str(processed / "emb"),
                "chroma_dir": str(processed / "chroma"),
                "results_dir": str(results),
                "questions": str(questions),
            },
            "dense_models": {"fake-a": "fake:hash-64", "fake-b": "fake:hash-32"},
            "rerankers": {"fake": "fake:overlap"},
            "defaults": {"chunker": "structural", "dense_model": "fake-a", "rerank_model": "fake"},
            "eval": {"bootstrap_resamples": 200, "seed": 7},
            "grid": [
                {
                    "chunkers": ["fixed_64", "structural"],
                    "retrievers": [
                        {"retriever": "bm25"},
                        {"retriever": "dense", "dense_model": "fake-a"},
                        {"retriever": "hybrid_rrf", "dense_model": "fake-a"},
                    ],
                    "xref_expand": [False, True],
                    "rerank": [False, True],
                    "rerank_model": "fake",
                }
            ],
            "selection": {
                "baselines": {"naive_rag": "fixed_64|dense|fake-a", "bm25": "fixed_64|bm25"},
                "min_answerable_for_test": 1,
            },
        }
    )


def write_settings(path: Path, settings: RetrievalSettings) -> Path:
    path.write_text(yaml.safe_dump(settings.model_dump(mode="json")), encoding="utf-8")
    return path


def ids_for(split: str, n: int, start: int = 1) -> list[str]:
    out = []
    i = start
    while len(out) < n:
        qid = f"q{i:03d}"
        if split_of(qid) == split:
            out.append(qid)
        i += 1
    return out


ANSWERABLE = [
    ("Who may file a complaint and how can it be filed?", "procedural", ["testact:s3(1)"]),
    ("What does the word goods mean?", "definition", ["testact:s2(2)"]),
    ("Within how many days can an appeal be made?", "numeric", ["testact:s4(1)"]),
    ("What are the rules called?", "factual", ["testrules2024:r1"]),
    (
        "Which provision of the Act governs a written complaint under the rules?",
        "cross_reference",
        ["testrules2024:r3(2)", "testact:s3(1)"],
    ),
]
UNANSWERABLE = ["What is the rate of goods and services tax on vegetables?"]


def write_questions(path: Path) -> list[dict[str, object]]:
    """Answerable + unanswerable questions spread over dev and test ids."""
    dev, test = ids_for("dev", 3), ids_for("test", 3)
    ids = [dev[0], test[0], dev[1], test[1], test[2], dev[2]]
    rows: list[dict[str, object]] = []
    for qid, (q, t, gold) in zip(ids, ANSWERABLE, strict=False):
        rows.append(
            {
                "id": qid,
                "question": q,
                "type": t,
                "answerable": True,
                "gold_node_ids": gold,
                "gold_answer": "reference answer",
                "difficulty": "easy",
                "author": "Swapnil",
            }
        )
    rows.append(
        {
            "id": ids[5],
            "question": UNANSWERABLE[0],
            "type": "unanswerable",
            "answerable": False,
            "gold_node_ids": [],
            "gold_answer": "not covered",
            "difficulty": "medium",
            "author": "Swapnil",
        }
    )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows
