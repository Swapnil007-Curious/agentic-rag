"""The frozen results of the evaluated collections (consumer_law, cs_ai) and how they are checked.

tests/data/frozen_results.json records, for every result table, every per-question / per-query
file and the test-access log of both evaluated collections, how the file may evolve and its
digest (text files with line endings normalised to LF):

* ``exact``: the whole file must be byte-identical;
* ``prefix``: the generation_test tables, to which the generator-scale ablation appends rows;
  the recorded bytes must be an unchanged prefix (every existing row untouched);
* ``json_prefix``: the test-access log, a JSON list to which every test evaluation appends an
  entry; the recorded entries must be unchanged;
* ``summary``: SUMMARY.md, regenerated with an ablation section; every existing section must be
  unchanged (the "written <time>" line and the ablation section are left out of the digest).

Per-question files are git-ignored (regenerable); when absent (a fresh clone) they are skipped,
when present they must match. The tables directories may only gain the files in
``allowed_new``.

Regenerate deliberately (only after results were intentionally recomputed):
    python tests/frozen_results.py --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).parent / "data" / "frozen_results.json"
RESULT_ROOTS = {"consumer_law": "results", "cs_ai": "collections/cs_ai/results"}
PREFIX_TABLES = (
    "generation_test.csv",
    "generation_test_by_type.csv",
    "generation_test_bootstrap.csv",
)
ALLOWED_NEW = (
    "SUMMARY.md",
    "generation_test_ablation.meta.json",
)
PER_QUESTION = ("per_question.jsonl", "eval.jsonl", "per_query.jsonl", "summary.json")
ABLATION_HEADING = "## Generator-scale ablation (test)"


def normalized(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summary_core(text: str) -> str:
    """SUMMARY.md without its "written <time>" line and without the ablation section."""
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    skipping = False
    for i, ln in enumerate(lines):
        if i == 2 and ln.startswith("Collection `"):
            continue  # "Collection `x` (profile `x`), written <UTC time> by ..."
        if ln.startswith("## "):
            skipping = ln.strip() == ABLATION_HEADING
        if not skipping:
            out.append(ln)
    return "\n".join(out)


def tracked_files() -> set[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", *RESULT_ROOTS.values()],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def entry_for(rel: str, tracked: set[str]) -> dict[str, Any]:
    path = REPO / rel
    name = path.name
    data = normalized(path)
    e: dict[str, Any] = {"tracked": rel in tracked}
    if name in PREFIX_TABLES and "/tables/" in rel:
        e.update(mode="prefix", sha256=sha(data), bytes=len(data))
    elif name == "test_access_log.json":
        entries = json.loads(data.decode("utf-8"))
        e.update(mode="json_prefix", entries=len(entries), sha256=_entries_digest(entries))
    elif name == "SUMMARY.md":
        e.update(mode="summary", sha256=sha(summary_core(data.decode("utf-8")).encode("utf-8")))
    else:
        e.update(mode="exact", sha256=sha(data))
    return e


def _entries_digest(entries: list[Any]) -> str:
    return sha(json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def collect() -> dict[str, Any]:
    tracked = tracked_files()
    files: dict[str, Any] = {}
    for root in RESULT_ROOTS.values():
        base = REPO / root
        for p in sorted((base / "tables").iterdir()):
            if p.is_file():
                rel = p.relative_to(REPO).as_posix()
                files[rel] = entry_for(rel, tracked)
        log = base / "test_access_log.json"
        if log.exists():
            rel = log.relative_to(REPO).as_posix()
            files[rel] = entry_for(rel, tracked)
        for sub in ("generation", "retrieval"):
            for p in sorted((base / sub).rglob("*")):
                if p.is_file() and p.name in PER_QUESTION:
                    rel = p.relative_to(REPO).as_posix()
                    files[rel] = entry_for(rel, tracked)
    return {
        "note": "frozen results of the evaluated collections; see tests/frozen_results.py",
        "allowed_new": [
            f"{root}/tables/{n}" for root in RESULT_ROOTS.values() for n in ALLOWED_NEW
        ],
        "files": files,
    }


def check(manifest: dict[str, Any]) -> list[str]:
    """Problems found (empty when every frozen file is intact)."""
    problems: list[str] = []
    for rel, e in manifest["files"].items():
        path = REPO / rel
        if not path.exists():
            if e.get("tracked"):
                problems.append(f"{rel}: missing")
            continue
        data = normalized(path)
        mode = e["mode"]
        if mode == "exact":
            ok = sha(data) == e["sha256"]
        elif mode == "prefix":
            ok = len(data) >= e["bytes"] and sha(data[: e["bytes"]]) == e["sha256"]
        elif mode == "json_prefix":
            entries = json.loads(data.decode("utf-8"))
            ok = (
                len(entries) >= e["entries"]
                and _entries_digest(entries[: e["entries"]]) == e["sha256"]
            )
        elif mode == "summary":
            ok = sha(summary_core(data.decode("utf-8")).encode("utf-8")) == e["sha256"]
        else:
            ok = False
        if not ok:
            problems.append(f"{rel}: changed ({mode})")
    known = set(manifest["files"]) | set(manifest["allowed_new"])
    for root in RESULT_ROOTS.values():
        for p in (REPO / root / "tables").iterdir():
            rel = p.relative_to(REPO).as_posix()
            if p.is_file() and rel not in known:
                problems.append(f"{rel}: new file in a frozen tables directory")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--write", action="store_true", help="(re)write tests/data/frozen_results.json")
    args = ap.parse_args()
    if args.write:
        data = collect()
        MANIFEST.write_text(
            json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"wrote {MANIFEST} ({len(data['files'])} files)")
        return 0
    problems = check(json.loads(MANIFEST.read_text(encoding="utf-8")))
    print("\n".join(problems) or "all frozen results intact")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
