"""Generator-scale ablation: systems that mirror an existing system on another generator backend
(G2H = G2 and G4H = G4 with the hosted model), evaluated once on the TEST split.

The existing results stay untouched:

* the reference systems' outputs (``generation/test/<G1|G2|G4>/per_question.jsonl`` and
  ``eval.jsonl``) are only read; they must cover exactly the current test questions;
* the new systems' rows are *appended* to ``generation_test.csv``, ``generation_test_by_type.csv``
  and ``generation_test_bootstrap.csv``: every existing line is kept byte for byte (a repeated
  run replaces only the lines of the ablation systems), the columns are the existing header's
  and the ``label`` column names the backend, the model and the date it was chosen;
* the paired bootstrap compares every ablation system with the reference system (G1), with G2
  (same retrieval, local generator) and with the system it mirrors (G4 for G4H);
* ``generation_test_ablation.meta.json`` records the backend (model id, selection date and
  rationale), the NLI model, the questions digest, token totals and API latency;
* ``generation/test/_complete_ablation.json`` marks a finished run (the standard run's
  ``_complete.json`` is separate and untouched).
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragsys.agents.config import GenerationSettings
from ragsys.eval.answers import bootstrap_rows, by_type_rows, summary_rows
from ragsys.eval.generation import load_system_records, questions_digest, read_jsonl
from ragsys.eval.harness import _cell
from ragsys.eval.schema import EvalQuestion
from ragsys.utils.runinfo import collect_run_info

ABLATION_MARKER = "_complete_ablation.json"
LABEL_PREFIX = "ablation:"


class AblationError(RuntimeError):
    """The ablation cannot run or cannot append its rows without touching existing results."""


def ablation_systems(settings: GenerationSettings, names: Sequence[str]) -> list[str]:
    """The names among ``names`` that run on a non-default backend."""
    default = settings.default_backend
    return [n for n in names if settings.backend_of(n) != default]


def ablation_label(backend: dict[str, Any]) -> str:
    date = backend.get("model_selected_on") or "date not recorded"
    return f"{LABEL_PREFIX} {backend['name']} {backend['model']} (chosen {date})"


def comparisons(settings: GenerationSettings, systems: Sequence[str]) -> dict[str, list[str]]:
    """Reference systems of each ablation system: the reference system (G1), G2 and the system
    it mirrors (``ablation_of``), in that order, without duplicates."""
    out: dict[str, list[str]] = {}
    for s in systems:
        refs = [settings.eval.reference_system, "G2", settings.system(s).ablation_of or ""]
        out[s] = [r for r in dict.fromkeys(refs) if r and r in settings.systems and r != s]
    return out


def reference_rows(
    out_dir: Path, system: str, questions: Sequence[EvalQuestion]
) -> list[dict[str, Any]]:
    """The frozen evaluation rows of ``system`` for exactly ``questions`` (read only)."""
    recs = load_system_records(out_dir, system, questions)
    if recs is None:
        raise AblationError(
            f"{out_dir / system / 'per_question.jsonl'} does not cover the current test "
            f"questions; the standard test evaluation of {system} must be complete first"
        )
    rows = {r["qid"]: r for r in read_jsonl(out_dir / system / "eval.jsonl")}
    missing = [q.id for q in questions if q.id not in rows]
    if missing:
        raise AblationError(
            f"{out_dir / system / 'eval.jsonl'} lacks {len(missing)} test question(s) "
            f"(e.g. {missing[:3]}); the standard test evaluation must be complete first"
        )
    return [rows[q.id] for q in questions]


# --------------------------------------------------------------------------- append-only CSV


def _line_terminator(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def append_rows(path: Path, rows: Sequence[dict[str, Any]], replace_systems: Sequence[str]) -> None:
    """Append ``rows`` to an existing table. Every existing line is kept byte for byte except
    lines of ``replace_systems`` (an earlier run of the same ablation), which are dropped. The
    rows must have exactly the table's columns."""
    if not path.exists():
        raise AblationError(f"{path} does not exist; run the standard test evaluation first")
    raw = path.read_bytes().decode("utf-8")
    term = _line_terminator(raw)
    lines = raw.split(term)
    if lines and lines[-1] == "":
        lines = lines[:-1]
    header = next(csv.reader([lines[0]]))
    if "system" not in header:
        raise AblationError(f"{path}: no 'system' column")
    sys_col = header.index("system")
    for r in rows:
        if set(r) != set(header):
            extra = sorted(set(r) - set(header))
            lack = sorted(set(header) - set(r))
            raise AblationError(f"{path}: columns differ (extra {extra}, missing {lack})")
    kept = [lines[0]]
    for ln in lines[1:]:
        cells = next(csv.reader([ln]))
        if cells and len(cells) > sys_col and cells[sys_col] in replace_systems:
            continue
        kept.append(ln)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, lineterminator=term)
    for r in rows:
        w.writerow({k: _cell(r.get(k)) for k in header})
    text = term.join(kept) + term + buf.getvalue()
    path.write_bytes(text.encode("utf-8"))


# --------------------------------------------------------------------------- cost


def cost_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Token totals and API latency of the LLM calls in ``records`` (reported latency: the
    measured one, or for a cached call the one measured when it was first made)."""
    calls = [c for r in records for s in r.get("steps", []) for c in s.get("llm_calls", [])]
    lat = [float(c["ms"]) for c in calls]
    fresh = [float(c["measured_ms"]) for c in calls if not c.get("cached")]

    def pct(vals: list[float], p: float) -> float | None:
        if not vals:
            return None
        vals = sorted(vals)
        k = min(len(vals) - 1, max(0, round(p / 100 * (len(vals) - 1))))
        return round(vals[k], 1)

    return {
        "questions": len(records),
        "llm_calls": len(calls),
        "cached_calls": sum(1 for c in calls if c.get("cached")),
        "prompt_tokens": sum(int(c.get("prompt_tokens", 0)) for c in calls),
        "completion_tokens": sum(int(c.get("completion_tokens", 0)) for c in calls),
        "api_latency_ms_p50": pct(lat, 50),
        "api_latency_ms_p95": pct(lat, 95),
        "api_latency_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
        "uncached_wall_ms_mean": round(statistics.fmean(fresh), 1) if fresh else None,
    }


# --------------------------------------------------------------------------- tables


def write_ablation_tables(
    settings: GenerationSettings,
    tables_dir: Path,
    split: str,
    new_rows: dict[str, list[dict[str, Any]]],
    ref_rows: dict[str, list[dict[str, Any]]],
    records: dict[str, list[dict[str, Any]]],
    questions: Sequence[EvalQuestion],
    backend: dict[str, Any],
    nli: dict[str, Any],
    extra_meta: dict[str, Any] | None = None,
) -> list[Path]:
    """Append the ablation rows to ``generation_<split>{,_by_type,_bootstrap}.csv`` and write
    ``generation_<split>_ablation.meta.json``."""
    label = ablation_label(backend)
    systems = list(new_rows)
    prefix = f"generation_{split}"
    paths: list[Path] = []
    p = tables_dir / f"{prefix}.csv"
    append_rows(p, summary_rows(new_rows, label), systems)
    paths.append(p)
    p = tables_dir / f"{prefix}_by_type.csv"
    append_rows(p, by_type_rows(new_rows, label), systems)
    paths.append(p)
    comp = comparisons(settings, systems)
    boot: list[dict[str, Any]] = []
    for s in systems:
        for ref in comp[s]:
            boot += bootstrap_rows(
                {ref: ref_rows[ref], s: new_rows[s]},
                ref,
                settings.eval.bootstrap_resamples,
                settings.eval.seed,
                label,
            )
    p = tables_dir / f"{prefix}_bootstrap.csv"
    append_rows(p, boot, systems)
    paths.append(p)
    meta = {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "split": split,
        "label": label,
        "systems": {s: settings.system(s).model_dump(mode="json") for s in systems},
        "backend": backend,
        "llm": {
            s: settings.llm(settings.backend_of(s)).model_dump(mode="json", exclude={"cache_dir"})
            for s in systems
        },
        "context": {
            s: settings.context_for(settings.backend_of(s)).model_dump(mode="json") for s in systems
        },
        "comparisons": comp,
        "nli": nli,
        "questions_digest": questions_digest(questions),
        "n_questions": len(questions),
        "cost": {s: cost_summary(records[s]) for s in systems},
        "tables_appended": [q.name for q in paths],
        "run_info": collect_run_info(),
        **(extra_meta or {}),
    }
    p = tables_dir / f"{prefix}_ablation.meta.json"
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    paths.append(p)
    return paths
