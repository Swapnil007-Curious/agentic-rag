"""Resumable step runner for scripts/run_collection_pipeline.py.

A pipeline is an ordered list of :class:`Step` objects. A step is skipped when it is complete:
its ``done()`` check passes and, if it has a fingerprint, the fingerprint recorded in the state
file when it last succeeded equals the current one (inputs unchanged). Otherwise it is run as a
subprocess; its output is copied to the console and to the log file, and its peak resident
memory is sampled while it runs. Steps that are themselves resumable (embeddings, the LLM cache,
the retrieval grid, per-question generation records) continue where they stopped, so rerunning
the pipeline after a crash or a restart repeats no finished work.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import psutil

from ragsys.utils.console import ascii_safe
from ragsys.utils.memory import fmt_duration

Executor = Callable[["Step", Callable[[str], None]], tuple[int, int]]


@dataclass
class Step:
    name: str
    argv: list[str]
    done: Callable[[], bool] = lambda: False
    fingerprint: Callable[[], str] | None = None
    estimate_s: float = 0.0
    accept_codes: tuple[int, ...] = (0,)
    fatal: bool = True
    note: str = ""
    always: bool = False  # run on every pipeline run (cheap checks); never "complete"
    # called before running when the step is not complete; may return replacement argv
    prepare: Callable[[list[str]], list[str]] | None = None
    # exit codes after which the step is started again after retry_wait_s (an exhausted API
    # quota: the step resumes from its caches), at most max_retries times
    retry_codes: tuple[int, ...] = ()
    retry_wait_s: float = 0.0
    max_retries: int = 0

    def command(self) -> str:
        return " ".join(_quote(a) for a in self.argv)


def _quote(arg: str) -> str:
    return f'"{arg}"' if (" " in arg or "|" in arg) else arg


@dataclass
class StepOutcome:
    name: str
    status: str  # skipped | ok | failed | warned
    seconds: float = 0.0
    peak_rss_mb: int = 0
    returncode: int | None = None


@dataclass
class RunReport:
    outcomes: list[StepOutcome] = field(default_factory=list)
    failed: str | None = None
    total_s: float = 0.0

    @property
    def complete(self) -> bool:
        return self.failed is None


# --------------------------------------------------------------------------- fingerprints


def file_digest(path: Path) -> str:
    """sha256 of a file's content ("missing" when absent)."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stat_digest(path: Path) -> str:
    """Cheap fingerprint for large files: size and modification time."""
    if not path.exists():
        return "missing"
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def combine(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:20]


# --------------------------------------------------------------------------- state + log


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"steps": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"steps": {}}
    return data if isinstance(data, dict) and "steps" in data else {"steps": {}}


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


class Log:
    """Timestamped lines to the console and to an append-only log file."""

    def __init__(self, path: Path, echo: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.echo = echo
        self._fh: TextIO = path.open("a", encoding="utf-8", newline="\n")

    def line(self, text: str, stamp: bool = True) -> None:
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out = f"[{now}] {text}" if stamp else text
        self._fh.write(out + "\n")
        self._fh.flush()
        if self.echo:
            print(ascii_safe(out), flush=True)

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- execution


def _tree_rss(proc: psutil.Process) -> int:
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += int(p.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def subprocess_executor(step: Step, out: Callable[[str], None]) -> tuple[int, int]:
    """Run the step's command; returns (exit code, peak RSS in bytes of the process tree)."""
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        step.argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    peak = 0
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        try:
            ps = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            return
        while not stop.is_set():
            try:
                peak = max(peak, _tree_rss(ps))
                peak = max(peak, int(getattr(ps.memory_info(), "peak_wset", 0) or 0))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return
            stop.wait(0.5)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    assert proc.stdout is not None
    for raw in proc.stdout:
        out(raw.rstrip("\n"))
    rc = proc.wait()
    stop.set()
    sampler.join(timeout=2)
    return rc, peak


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while the pipeline runs (no-op elsewhere)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        flags = es_continuous | (es_system_required if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:  # pragma: no cover - best effort only
        pass


def run_steps(
    steps: Sequence[Step],
    state_path: Path,
    log: Log,
    *,
    executor: Executor = subprocess_executor,
    resume_hint: str = "",
    sleep: Callable[[float], None] = time.sleep,
) -> RunReport:
    report = RunReport()
    state = read_state(state_path)
    t_all = time.perf_counter()
    for step in steps:
        entry = state["steps"].get(step.name, {})
        fp = step.fingerprint() if step.fingerprint else None
        complete = (
            not step.always and step.done() and (fp is None or entry.get("fingerprint") == fp)
        )
        if complete:
            log.line(f"SKIP  {step.name} (already complete)")
            report.outcomes.append(StepOutcome(step.name, "skipped"))
            continue
        argv = step.prepare(list(step.argv)) if step.prepare else list(step.argv)
        run = Step(**{**step.__dict__, "argv": argv})
        est = f" (estimate {fmt_duration(step.estimate_s)})" if step.estimate_s else ""
        log.line(f"START {step.name}{est}: {run.command()}")
        t0 = time.perf_counter()
        retries = 0
        while True:
            try:
                rc, peak = executor(run, lambda s: log.line("    " + s, stamp=False))
            except OSError as exc:
                log.line(f"    could not start: {exc}")
                rc, peak = 127, 0
            if rc not in step.retry_codes or retries >= step.max_retries:
                break
            retries += 1
            log.line(
                f"WAIT  {step.name} exited with {rc} (API quota); starting it again in "
                f"{fmt_duration(step.retry_wait_s)} (retry {retries} of {step.max_retries})"
            )
            sleep(step.retry_wait_s)
            log.line(f"RETRY {step.name}: {run.command()}")
        secs = time.perf_counter() - t0
        mb = round(peak / 1024**2)
        ok = rc in step.accept_codes
        if ok and not step.always and not step.done():
            log.line(f"    exit code {rc} but the completion check of {step.name} failed")
            ok = False
        if ok:
            log.line(f"DONE  {step.name} in {fmt_duration(secs)} (exit {rc}, peak RSS {mb:,} MB)")
            state["steps"][step.name] = {
                "fingerprint": step.fingerprint() if step.fingerprint else None,
                "completed_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                "seconds": round(secs, 1),
                "peak_rss_mb": mb,
                "returncode": rc,
            }
            write_state(state_path, state)
            report.outcomes.append(StepOutcome(step.name, "ok", secs, mb, rc))
            continue
        if not step.fatal:
            log.line(f"WARN  {step.name} failed (exit {rc}); continuing (not required)")
            report.outcomes.append(StepOutcome(step.name, "warned", secs, mb, rc))
            continue
        log.line(f"FAIL  {step.name} (exit {rc}) after {fmt_duration(secs)}, peak RSS {mb:,} MB")
        report.outcomes.append(StepOutcome(step.name, "failed", secs, mb, rc))
        report.failed = step.name
        break
    report.total_s = time.perf_counter() - t_all
    if report.complete:
        log.line(f"PIPELINE COMPLETE in {fmt_duration(report.total_s)}")
    else:
        log.line(f"PIPELINE STOPPED at step '{report.failed}' after {fmt_duration(report.total_s)}")
        if resume_hint:
            log.line(f"Fix the problem above, then resume with: {resume_hint}")
    return report
