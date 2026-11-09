"""Shared fixtures: synthetic PDFs, an ingest config for them, and one ingest run."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from pdf_fixtures import build_act_pdf, build_report_pdf, build_rules_pdf
from ragsys.config import DocumentConfig, IngestConfig
from ragsys.ingest.pipeline import IngestResult, run_ingest

REPO = Path(__file__).resolve().parents[1]


def fixture_config() -> IngestConfig:
    return IngestConfig(
        documents=[
            DocumentConfig(
                file="testact.pdf",
                doc_id="testact",
                title="The Test Protection Act, 2024",
                doc_type="act",
                parser="statute",
            ),
            DocumentConfig(
                file="testrules.pdf",
                doc_id="testrules2024",
                title="The Test Rules, 2024",
                doc_type="rules",
                parser="statute",
                parent_act="testact",
                start_marker=r"^MINISTRY OF TESTING",
                end_marker=r"^\[\s*F\.?\s*No\.?",
            ),
            DocumentConfig(
                file="report.pdf",
                doc_id="report",
                title="Annual Garden Report",
                doc_type="generic",
                parser="generic",
            ),
        ],
        structure={"act_aliases": {"Test Protection Act, 2024": "testact"}},
    )


@pytest.fixture(scope="session")
def pdf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("raw")
    build_act_pdf(d / "testact.pdf")
    build_rules_pdf(d / "testrules.pdf")
    build_report_pdf(d / "report.pdf")
    return d


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("processed")


@pytest.fixture(scope="session")
def ingested(pdf_dir: Path, processed_dir: Path) -> IngestResult:
    return run_ingest(fixture_config(), pdf_dir, processed_dir, write=True)


@pytest.fixture(scope="session")
def nodes_by_id(ingested: IngestResult) -> dict[str, object]:
    return {n.node_id: n for n in ingested.nodes}


@pytest.fixture(scope="session")
def tokenizer():  # type: ignore[no-untyped-def]
    from ragsys.chunking.tokenizer import get_tokenizer

    return get_tokenizer()


def load_script(name: str) -> ModuleType:
    """Import scripts/<name>.py as a module (so tests can call main(argv) in-process)."""
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"script_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def script() -> Callable[[str], ModuleType]:
    return load_script
