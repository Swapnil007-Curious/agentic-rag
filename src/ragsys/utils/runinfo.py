"""Collect environment / provenance information embedded in every report JSON."""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil

from ragsys.config import REPO_ROOT

TRACKED_PACKAGES = (
    "ragsys",
    "pymupdf",
    "pydantic",
    "PyYAML",
    "numpy",
    "psutil",
    "torch",
    "transformers",
    "tokenizers",
    "sentence-transformers",
)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info(cwd: Path = REPO_ROOT) -> dict[str, Any]:
    commit = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain"], cwd)
    return {
        "commit": commit,  # None when the repository has no commits yet
        "dirty": None if status is None else bool(status),
    }


def _cpu_model_windows() -> str | None:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            return str(value).strip()
        except OSError:
            return None
    return None


def _cpu_model_linux() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def _cpu_model_macos() -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def cpu_model() -> str:
    model: str | None
    if sys.platform == "win32":
        model = _cpu_model_windows()
    elif sys.platform == "darwin":
        model = _cpu_model_macos()
    else:
        model = _cpu_model_linux()
    return model or platform.processor() or "unknown"


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_run_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "utc_time": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "git": git_info(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / 1024**3, 2),
        "packages": package_versions(),
    }
