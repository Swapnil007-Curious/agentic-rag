"""Assemble a self-contained Hugging Face Docker Space in deploy/hf_space/build/.

Usage:
    python scripts/build_space.py [--out deploy/hf_space/build] [--collections-file ...]

The folder gets the application code (src/ragsys, scripts/serve.py), every config, the app
index of each collection (structural chunks JSONL plus their bge-small embeddings and sidecar;
nothing else is read at run time), the SOURCES.csv attributions, the vendored static assets,
example questions (configs/assistant_examples.json) and the Space files from deploy/hf_space/
(Dockerfile, README.md with the Space front matter, requirements.txt). Files larger than 10 MB
are listed for Git LFS in .gitattributes. Source PDFs, .env files and secrets are never
copied: the finished folder is scanned for PDFs (by name and content), .env/key files and the
values of LLM_API_KEY and HF_TOKEN from the environment, and the build fails if any is found.
Prints a tree summary and the total size. Nothing is uploaded (see scripts/deploy_space.py).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from ragsys.agents.config import load_generation_settings
from ragsys.agents.runtime import serving_retrieval
from ragsys.assistant.config import load_assistant_settings
from ragsys.assistant.setup import collection_examples, sources_csv
from ragsys.collections import DEFAULT_COLLECTIONS_FILE, Collection, load_collections
from ragsys.config import REPO_ROOT, resolve_path
from ragsys.retrieval.embed import model_slug

DEFAULT_OUT = Path("deploy/hf_space/build")
TEMPLATE_DIR = Path("deploy/hf_space")
TEMPLATE_FILES = ("Dockerfile", "README.md", "requirements.txt")
MARKER = ".space-build"
LFS_THRESHOLD = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = (".pdf", ".key", ".pem")
SECRET_ENV = ("LLM_API_KEY", "HF_TOKEN")
_SKIP_PARTS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


class Entry(NamedTuple):
    src: Path  # absolute source file
    dst: str  # POSIX path inside the build folder


def forbidden(name: str) -> bool:
    low = name.lower()
    return low.endswith(FORBIDDEN_SUFFIXES) or low == ".env" or low.startswith(".env.")


def _tree(root: Path, base: Path, dst_prefix: str) -> list[Entry]:
    out: list[Entry] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or _SKIP_PARTS & set(p.parts) or p.suffix in (".pyc", ".pyo"):
            continue
        if any(part.endswith(".egg-info") for part in p.parts):
            continue
        out.append(Entry(p, f"{dst_prefix}{p.relative_to(base).as_posix()}"))
    return out


def _rel(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def plan(
    repo: Path = REPO_ROOT, collections_file: str | Path = DEFAULT_COLLECTIONS_FILE
) -> list[Entry]:
    """Every file of the build (sources resolved against ``repo``)."""
    entries: list[Entry] = []
    entries += _tree(repo / "src" / "ragsys", repo / "src", "src/")
    entries.append(Entry(repo / "scripts" / "serve.py", "scripts/serve.py"))
    entries.append(Entry(repo / "LICENSE", "LICENSE"))
    entries += [e for e in _tree(repo / "configs", repo, "") if e.src.suffix in (".yaml", ".yml")]
    settings = load_assistant_settings(repo / "configs" / "assistant.yaml")
    slug = model_slug(settings.embedding_model)
    reg = load_collections(collections_file)
    for name in reg.ordered():
        coll = Collection(name, reg.collections[name])
        gen = load_generation_settings(resolve_path(coll.generation_config, repo))
        rs, note = serving_retrieval(coll)
        processed = resolve_path(rs.paths.processed_dir, repo)
        emb = resolve_path(rs.paths.emb_dir, repo) / slug
        chunker = settings.chunker
        files = [
            processed / "chunks" / f"{chunker}.jsonl",
            emb / f"{chunker}.npy",
            emb / f"{chunker}.json",
        ]
        csvs = {sources_csv(coll, gen, served_app=not note), resolve_path(gen.ui.sources_csv, repo)}
        for f in [*files, *sorted(csvs)]:
            if not f.exists():
                raise FileNotFoundError(f"{name}: {f} is missing (build the app index first)")
            entries.append(Entry(f, _rel(f, repo)))
    for name in TEMPLATE_FILES:
        entries.append(Entry(repo / TEMPLATE_DIR / name, name))
    seen: dict[str, Entry] = {}
    for e in entries:
        seen.setdefault(e.dst, e)
    return list(seen.values())


def examples_json(collections_file: str | Path) -> bytes:
    reg = load_collections(collections_file)
    data = {
        name: collection_examples(Collection(name, reg.collections[name]), {})
        for name in reg.ordered()
    }
    return (json.dumps(data, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def prepare_out(out: Path) -> None:
    """Empty ``out`` (only a previous build folder, recognised by its marker, is deleted)."""
    if out.exists():
        if any(out.iterdir()) and not (out / MARKER).exists():
            raise SystemExit(f"{out} exists and is not a Space build folder; refusing to delete it")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def assemble(entries: list[Entry], out: Path, generated: dict[str, bytes]) -> None:
    for e in entries:
        raw = "/raw/" in f"/{e.dst}" and not Path(e.dst).name.startswith("SOURCES")
        if forbidden(Path(e.dst).name) or raw:
            raise ValueError(f"refusing to copy {e.dst}")
        target = out / e.dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(e.src, target)
    for dst, data in generated.items():
        target = out / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (out / MARKER).write_text("built by scripts/build_space.py\n", encoding="utf-8")


def lfs_attributes(out: Path, threshold: int = LFS_THRESHOLD) -> list[str]:
    """Write .gitattributes for files larger than ``threshold``; returns their paths."""
    big = sorted(
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file() and p.stat().st_size > threshold
    )
    lines = ["# files larger than 10 MB are stored with Git LFS"]
    lines += [
        f"{path.replace(' ', '[[:space:]]')} filter=lfs diff=lfs merge=lfs -text" for path in big
    ]
    (out / ".gitattributes").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return big


def secret_values(env: dict[str, str] | None = None) -> list[str]:
    src = os.environ if env is None else env
    return [v for k in SECRET_ENV if len(v := (src.get(k) or "").strip()) >= 8]


def scan(out: Path, secrets: list[str]) -> list[str]:
    """Problems in a build folder: PDFs (by name or content), .env / key files, secrets."""
    problems: list[str] = []
    needles = [s.encode("utf-8") for s in secrets]
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(out).as_posix()
        if forbidden(p.name):
            problems.append(f"{rel}: forbidden file type")
        data = p.read_bytes()
        if data[:1024].lstrip().startswith(b"%PDF-"):
            problems.append(f"{rel}: PDF content")
        for i, needle in enumerate(needles):
            if needle in data:
                problems.append(
                    f"{rel}: contains the value of {SECRET_ENV[i] if i < len(SECRET_ENV) else 'a secret'}"
                )
    return problems


def summary(out: Path) -> tuple[list[tuple[str, int, int]], int, int]:
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total = n = 0
    for p in out.rglob("*"):
        if not p.is_file():
            continue
        parts = p.relative_to(out).parts
        depth = 3 if parts[0] in ("data", "collections") else 2
        key = "/".join(parts[: min(depth, len(parts) - 1)]) or parts[0]
        size = p.stat().st_size
        groups[key][0] += 1
        groups[key][1] += size
        total += size
        n += 1
    rows = sorted((k, v[0], v[1]) for k, v in groups.items())
    return rows, n, total


def mb(n: int) -> str:
    return f"{n / 1024 / 1024:7.1f} MB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--collections-file", default=str(DEFAULT_COLLECTIONS_FILE))
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    out = resolve_path(Path(args.out))
    entries = plan(REPO_ROOT, args.collections_file)
    prepare_out(out)
    assemble(
        entries, out, {"configs/assistant_examples.json": examples_json(args.collections_file)}
    )
    big = lfs_attributes(out)
    problems = scan(out, secret_values())
    if problems:
        print("BUILD REJECTED:")
        for p in problems:
            print(f"  {p}")
        shutil.rmtree(out)
        return 1
    rows, n, total = summary(out)
    print(f"Space build: {out}")
    for key, count, size in rows:
        print(f"  {key:<52} {count:5d} files {mb(size)}")
    print(f"  {'(total)':<52} {n:5d} files {mb(total)}")
    print(f"Git LFS (> 10 MB): {', '.join(big) if big else 'none'}")
    print("Checked: no PDFs, no .env or key files, no LLM_API_KEY / HF_TOKEN values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
