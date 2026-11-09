"""Config loading (YAML, extra fields forbidden), seeding and run info."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pydantic
import pytest

from ragsys.config import REPO_ROOT, load_chunking_config, load_ingest_config
from ragsys.utils.runinfo import collect_run_info
from ragsys.utils.seed import set_seed


def test_repo_configs_load() -> None:
    ing = load_ingest_config(REPO_ROOT / "configs" / "ingest.yaml")
    assert [d.doc_id for d in ing.enabled_documents()] == [
        "cpa2019",
        "ecom2020",
        "commissions2020",
        "jurisdiction2021",
        "mediation2020",
    ]
    ch = load_chunking_config(REPO_ROOT / "configs" / "chunking.yaml")
    assert [f.name for f in ch.fixed] == ["fixed_256", "fixed_512"]
    assert (ch.fixed[0].size, ch.fixed[0].overlap, ch.fixed[1].size, ch.fixed[1].overlap) == (
        256,
        32,
        512,
        64,
    )
    assert (ch.semantic.breakpoint_percentile, ch.semantic.min_tokens, ch.semantic.max_tokens) == (
        90.0,
        64,
        512,
    )
    assert (ch.structural.max_tokens, ch.structural.min_tokens) == (384, 48)


def test_extra_fields_forbidden(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("tokenizer_model: x\nsurprise: 1\n", encoding="utf-8")
    with pytest.raises(pydantic.ValidationError):
        load_chunking_config(p)
    p.write_text("fixed:\n  - {name: f, size: 10, overlap: 10}\n", encoding="utf-8")
    with pytest.raises(pydantic.ValidationError):
        load_chunking_config(p)


def test_seed_is_reproducible() -> None:
    set_seed(7)
    a = (random.random(), float(np.random.rand()))
    set_seed(7)
    assert (random.random(), float(np.random.rand())) == a


def test_run_info_fields() -> None:
    info = collect_run_info()
    for key in ("utc_time", "git", "python", "os", "cpu_model", "ram_total_gb", "packages"):
        assert key in info
    assert info["python"].startswith("3.11")
    assert info["packages"]["pydantic"]
    assert set(info["git"]) == {"commit", "dirty"}
