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


@pytest.fixture(scope="session")
def retrieval_dir(
    ingested: IngestResult,
    processed_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
    tokenizer: object,
) -> Path:
    """Copy of the synthetic processed corpus plus chunks/fixed_64.jsonl and structural.jsonl."""
    import shutil

    from ragsys.chunking.fixed import FixedChunker
    from ragsys.chunking.pipeline import load_corpus
    from ragsys.chunking.structural import StructuralChunker

    d = tmp_path_factory.mktemp("retrieval") / "processed"
    shutil.copytree(processed_dir / "structure", d / "structure")
    shutil.copytree(processed_dir / "clean", d / "clean")
    docs = load_corpus(d)
    (d / "chunks").mkdir()
    chunkers = [
        FixedChunker("fixed_64", 64, 8, tokenizer),  # type: ignore[arg-type]
        StructuralChunker(tokenizer, 384, 48, True),  # type: ignore[arg-type]
    ]
    for chunker in chunkers:
        with (d / "chunks" / f"{chunker.name}.jsonl").open("w", encoding="utf-8") as fh:
            for doc in docs:
                for c in chunker.chunk_document(doc):
                    fh.write(c.model_dump_json() + "\n")
    return d
