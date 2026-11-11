"""Select the configs to evaluate on the test split, from DEV results only.

Usage:
    python scripts/select_configs.py [--collection NAME] [--config FILE] [--smoke]
Reads results/tables/retrieval_dev.csv and picks, by DEV token-budget coverage@1024 (ties:
higher Recall@5, then lower median total latency, then config id), the best config overall and
the best config per chunker. The baselines of configs/retrieval.yaml (naive RAG
fixed_512|dense|bge-base and fixed_512|bm25) are always included. Writes
results/tables/selected_configs.json with the rule used and each config's full settings.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from typing import Any

from ragsys.collections import add_collection_arguments, collection_from_args, config_path
from ragsys.config import resolve_path
from ragsys.eval.harness import SMOKE_LABEL
from ragsys.retrieval.config import config_dir_name, load_retrieval_settings, parse_config_id
from ragsys.utils.console import emit
from ragsys.utils.runinfo import collect_run_info

RULE = (
    "maximise DEV coverage@1024 (token-budget coverage); ties: higher DEV recall@5, then lower "
    "median total latency, then config_id; best overall + best per chunker; baselines always "
    "included"
)


def _f(v: str | None) -> float | None:
    return float(v) if v not in (None, "") else None


def sort_key(row: dict[str, str]) -> tuple[float, float, float, str]:
    cov = _f(row.get("coverage@1024"))
    rec = _f(row.get("recall@5"))
    lat = _f(row.get("latency_total_p50_ms"))
    return (
        -(cov if cov is not None else -1.0),
        -(rec if rec is not None else -1.0),
        lat if lat is not None else float("inf"),
        row["config_id"],
    )


def select(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Map config_id -> roles."""
    usable = [r for r in rows if _f(r.get("coverage@1024")) is not None]
    roles: dict[str, list[str]] = {}
    if usable:
        best = sorted(usable, key=sort_key)[0]
        roles.setdefault(best["config_id"], []).append("best_overall")
        for chunker in sorted({r["chunker"] for r in usable}):
            b = sorted((r for r in usable if r["chunker"] == chunker), key=sort_key)[0]
            roles.setdefault(b["config_id"], []).append(f"best_{chunker}")
    return roles


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="default: the collection's retrieval config")
    ap.add_argument("--smoke", action="store_true", help="read/write under results/smoke")
    add_collection_arguments(ap)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    coll = collection_from_args(args)
    settings = load_retrieval_settings(
        resolve_path(config_path(args.config, coll.retrieval_config))
    )
    root = resolve_path(settings.paths.results_dir) / ("smoke" if args.smoke else "")
    table = root / "tables" / "retrieval_dev.csv"
    if not table.exists():
        emit(f"{table} not found; run scripts/run_retrieval_grid.py --split dev first")
        return 1
    with table.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_id = {r["config_id"]: r for r in rows}
    roles = select(rows)
    if not roles:
        emit(
            "WARNING: no config has DEV metrics (no answerable dev questions); only baselines kept"
        )
    for name, cid in settings.selection.baselines.items():
        roles.setdefault(cid, []).append(f"baseline_{name}")

    selected: list[dict[str, Any]] = []
    for cid, rl in roles.items():
        summary_path = root / "retrieval" / "dev" / config_dir_name(cid) / "summary.json"
        if summary_path.exists():
            cfg = json.loads(summary_path.read_text(encoding="utf-8"))["config"]
        else:
            cfg = parse_config_id(cid).model_dump(mode="json")
        row = by_id.get(cid, {})
        selected.append(
            {
                "config_id": cid,
                "roles": rl,
                "config": cfg,
                "dev": {
                    k: _f(row.get(k)) for k in ("coverage@1024", "recall@5", "latency_total_p50_ms")
                }
                | {"n_answerable": int(row["n_answerable"]) if row.get("n_answerable") else None},
            }
        )
    ref = settings.selection.baselines[settings.selection.reference_baseline]
    out = {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "label": SMOKE_LABEL if args.smoke else "",
        "rule": RULE,
        "source_table": str(table),
        "reference_baseline_id": ref,
        "selected": selected,
        "settings": settings.model_dump(mode="json", exclude={"grid"}),
        "run_info": collect_run_info(),
    }
    out_path = root / "tables" / "selected_configs.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    emit(f"rule: {RULE}")
    for s in selected:
        d = s["dev"]
        cov = "n/a" if d["coverage@1024"] is None else f"{d['coverage@1024']:.3f}"
        rec = "n/a" if d["recall@5"] is None else f"{d['recall@5']:.3f}"
        emit(f"  {s['config_id']:55s} {','.join(s['roles']):40s} cov@1024={cov} R@5={rec}")
    emit(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
