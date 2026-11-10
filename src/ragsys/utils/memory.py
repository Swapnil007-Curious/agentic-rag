"""Process memory reporting and CPU thread settings for long CPU-only runs."""

from __future__ import annotations

import sys

import psutil


def rss_bytes() -> int:
    """Current resident set size (working set on Windows) of this process."""
    return int(psutil.Process().memory_info().rss)


def peak_rss_bytes() -> int:
    """Peak resident set size of this process so far (peak working set on Windows)."""
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is not None:
        return int(peak)
    if sys.platform != "win32":
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(ru if sys.platform == "darwin" else ru * 1024)  # bytes on macOS, KiB on Linux
    return int(info.rss)


def fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**2:,.0f} MB"


def set_torch_threads(n: int) -> int:
    """Set torch's intra-op CPU threads (no-op if torch is unavailable); returns the setting."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return 0
    torch.set_num_threads(n)
    return int(torch.get_num_threads())


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(round(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"
