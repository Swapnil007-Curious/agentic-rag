"""Citation labels in a chat reply: parse [C1]-style labels in the prose (not in code), strip the
labels that do not name a passage given to the model, and the source line shown under the reply.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass

from ragsys.agents.modes import normalize_citations

LABEL_RE = re.compile(r"\[(C\d{1,3})\]")
_STRIP_RE = re.compile(r"[ \t]*\[(C\d{1,3})\]")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"(`+)(?:.+?)\1")

GROUNDED_LINE = "Answered with documents ({n} {noun})"
GENERAL_LINE = "General knowledge — no matching documents"


def _map_prose(text: str, fn: Callable[[str], str]) -> str:
    """Apply ``fn`` to the prose of a Markdown text: fenced code blocks and inline code spans
    are left untouched."""
    out: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        m = _FENCE_RE.match(line)
        if fence is not None:
            out.append(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            continue
        if m:
            fence = m.group(1)
            out.append(line)
            continue
        pos = 0
        parts: list[str] = []
        for cm in _INLINE_CODE_RE.finditer(line):
            parts.append(fn(line[pos : cm.start()]))
            parts.append(cm.group(0))
            pos = cm.end()
        parts.append(fn(line[pos:]))
        out.append("".join(parts))
    return "\n".join(out)


def find_labels(text: str) -> list[str]:
    """[Cn] labels of the prose, in order of first appearance."""
    seen: list[str] = []

    def collect(segment: str) -> str:
        for m in LABEL_RE.finditer(normalize_citations(segment)):
            if m.group(1) not in seen:
                seen.append(m.group(1))
        return segment

    _map_prose(text, collect)
    return seen


@dataclass
class CitationResult:
    text: str  # the reply with normalised labels and invalid labels removed
    used: list[str]  # valid labels cited, in order of first appearance
    removed: list[str]  # labels that named no passage

    @property
    def grounded(self) -> bool:
        return bool(self.used)

    def source_line(self) -> str:
        return source_line(len(self.used))


def source_line(n_sources: int) -> str:
    if n_sources <= 0:
        return GENERAL_LINE
    return GROUNDED_LINE.format(n=n_sources, noun="source" if n_sources == 1 else "sources")


def clean_citations(text: str, valid: Collection[str]) -> CitationResult:
    """Normalise citation variants ("【C3】", "[C1, C3]") in the prose and remove every label
    that is not in ``valid`` (all of them when no passages were given)."""
    used: list[str] = []
    removed: list[str] = []

    def fix(segment: str) -> str:
        segment = normalize_citations(segment)

        def repl(m: re.Match[str]) -> str:
            label = m.group(1)
            if label in valid:
                if label not in used:
                    used.append(label)
                return m.group(0)
            if label not in removed:
                removed.append(label)
            return ""

        return _STRIP_RE.sub(repl, segment)

    cleaned = _map_prose(text, fix)
    return CitationResult(cleaned, used, removed)
