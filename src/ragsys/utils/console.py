"""Console-safe output for Windows terminals (cp1252 / OEM code pages, PowerShell pipes)."""

from __future__ import annotations

import unicodedata

_ASCII = {
    "—": "--",
    "–": "-",
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen (common in model replies)
    "−": "-",  # minus sign
    " ": " ",  # narrow no-break space
    "→": "->",
    "≤": "<=",
    "≥": ">=",
    "×": "x",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "₹": "Rs.",
    "•": "*",
    " ": " ",
}


def ascii_safe(text: str) -> str:
    """Fold to plain ASCII so output reads correctly in any Windows console or pipe."""
    text = "".join(_ASCII.get(c, c) for c in text)
    return unicodedata.normalize("NFKD", text).encode("ascii", "replace").decode("ascii")


def emit(*parts: object) -> None:
    print(*(ascii_safe(str(p)) for p in parts))
