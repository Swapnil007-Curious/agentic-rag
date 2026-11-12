"""The Hugging Face Space build: self-contained, with the app indexes and attributions, and
never a PDF, a .env file or a secret."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragsys.config import REPO_ROOT


def _module(script):  # type: ignore[no-untyped-def]
    return script("build_space")


def test_assemble_refuses_pdfs_env_files_and_raw_documents(tmp_path: Path, script) -> None:  # type: ignore[no-untyped-def]
    bs = _module(script)
    src = tmp_path / "src.bin"
    src.write_bytes(b"data")
    for dst in (
        "collections/x/raw/book.pdf",
        ".env",
        "configs/.env.local",
        "x/raw/notes.txt",
        "k.pem",
    ):
        out = tmp_path / f"out{abs(hash(dst))}"
        out.mkdir()
        with pytest.raises(ValueError):
            bs.assemble([bs.Entry(src, dst)], out, {})
    out = tmp_path / "ok"
    out.mkdir()
    bs.assemble([bs.Entry(src, "collections/x/raw/SOURCES.csv")], out, {"configs/a.json": b"{}"})
    assert (out / "collections/x/raw/SOURCES.csv").exists() and (out / bs.MARKER).exists()


def test_scan_finds_pdfs_env_files_and_secrets(tmp_path: Path, script) -> None:  # type: ignore[no-untyped-def]
    bs = _module(script)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "renamed.bin").write_bytes(b"%PDF-1.7\n...")
    (tmp_path / "notes.pdf").write_bytes(b"x")
    (tmp_path / ".env").write_text("X=1", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("key: gsk_live_secret_value_123\n", encoding="utf-8")
    (tmp_path / "clean.txt").write_text("nothing here", encoding="utf-8")
    secrets = bs.secret_values({"LLM_API_KEY": "gsk_live_secret_value_123", "HF_TOKEN": "short"})
    assert secrets == ["gsk_live_secret_value_123"]  # values shorter than 8 characters are ignored
    problems = bs.scan(tmp_path, secrets)
    assert any("renamed.bin: PDF content" in p for p in problems)
    assert any("notes.pdf" in p for p in problems) and any(".env" in p for p in problems)
    assert any("config.yaml: contains the value of LLM_API_KEY" in p for p in problems)
    assert not any("clean.txt" in p for p in problems)
    assert "gsk_live_secret_value_123" not in "\n".join(problems)  # never echoed


def test_prepare_out_only_deletes_a_previous_build(tmp_path: Path, script) -> None:  # type: ignore[no-untyped-def]
    bs = _module(script)
    other = tmp_path / "important"
    other.mkdir()
    (other / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        bs.prepare_out(other)
    assert (other / "keep.txt").exists()
    build = tmp_path / "build"
    build.mkdir()
    (build / bs.MARKER).write_text("", encoding="utf-8")
    (build / "old.txt").write_text("x", encoding="utf-8")
    bs.prepare_out(build)
    assert list(build.iterdir()) == []


def test_lfs_attributes(tmp_path: Path, script) -> None:  # type: ignore[no-untyped-def]
    bs = _module(script)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "big file.jsonl").write_bytes(b"x" * 2048)
    (tmp_path / "small.txt").write_bytes(b"x")
    assert bs.lfs_attributes(tmp_path, threshold=1024) == ["d/big file.jsonl"]
    text = (tmp_path / ".gitattributes").read_text(encoding="utf-8")
    assert "d/big[[:space:]]file.jsonl filter=lfs diff=lfs merge=lfs -text" in text


def test_full_build_has_indexes_and_no_pdfs_or_secrets(
    tmp_path: Path,
    script,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    bs = _module(script)
    try:
        entries = bs.plan(REPO_ROOT)
    except FileNotFoundError as exc:
        pytest.skip(f"app indexes are not built here: {exc}")
    dsts = {e.dst for e in entries}
    assert not any(d.lower().endswith(".pdf") for d in dsts)
    assert "Dockerfile" in dsts and "README.md" in dsts and "scripts/serve.py" in dsts
    assert "src/ragsys/assistant/service.py" in dsts and "configs/assistant.yaml" in dsts
    assert "src/ragsys/app/static/vendor/katex/LICENSE" in dsts
    for d in (
        "data/processed/chunks/structural.jsonl",
        "data/processed/emb/BAAI_bge-small-en-v1.5/structural.npy",
        "collections/cs_ai/processed_app/emb/BAAI_bge-small-en-v1.5/structural.json",
        "collections/physics_chemistry/processed/chunks/structural.jsonl",
        "collections/cs_ai/raw/SOURCES_app.csv",
        "data/raw/SOURCES.csv",
    ):
        assert d in dsts, d
    data = [d for d in dsts if not d.startswith("src/")]
    assert not any("/structure/" in d or "llm_cache" in d or "/results/" in d for d in data)
    # build into a temporary folder with fake secrets in the environment: the scan is clean
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-0123456789")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_0123456789")
    out = tmp_path / "build"
    assert bs.main(["--out", str(out)]) == 0
    files = [p for p in out.rglob("*") if p.is_file()]
    assert files and not any(p.suffix.lower() == ".pdf" or p.name.startswith(".env") for p in files)
    assert bs.scan(out, ["test-llm-key-0123456789", "hf_test_token_0123456789"]) == []
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert (
        readme.startswith("---\ntitle: Agentic-RAG Study Assistant\n") and "sdk: docker" in readme
    )
    assert "app_port: 7860" in readme and "license: mit" in readme
    docker = (out / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in docker and "download.pytorch.org/whl/cpu" in docker
    assert "USER user" in docker and "HOST=0.0.0.0" in docker and "EXPOSE 7860" in docker
    assert "snapshot_download('BAAI/bge-small-en-v1.5'" in docker and "HF_HUB_OFFLINE=1" in docker
    assert (out / "configs" / "assistant_examples.json").exists()
