"""Answer text handling shared by the guards, the evaluation and the UI: citation tags like
``[C2]`` / ``[C1][C3]`` / ``[C1, C3]``, sentence splitting that respects legal abbreviations
("Rs. 50", "s. 35", "No. 4") and citation groups placed after the full stop, and tag removal."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# One citation group: [C2], [c2], [C1, C3], [C1; C3], [C1,3]
CITATION_GROUP_RE = re.compile(r"\[\s*[Cc]\s*\d+(?:\s*[,;]\s*(?:[Cc]\s*)?\d+)*\s*\]")
_LABEL_NUM_RE = re.compile(r"\d+")
# the same tags written with fullwidth brackets, as some models do: 【C3】, 【C3†L1-L4】, 【C1, C3】
_FULLWIDTH_GROUP_RE = re.compile(
    r"【\s*([Cc]\s*\d+(?:\s*[,;]\s*(?:[Cc]\s*)?\d+)*)(?:\s*†[^】]*)?\s*】"
)
_LOOSE_LABEL_RE = re.compile(r"^\s*\[?\s*[Cc]?\s*(\d+)\s*\]?\s*$")

# Words after which a full stop does not end a sentence.
_ABBREVIATIONS = frozenset(
    {
        "rs",
        "no",
        "nos",
        "s",
        "ss",
        "r",
        "rr",
        "sec",
        "secs",
        "cl",
        "art",
        "arts",
        "sub",
        "vs",
        "v",
        "viz",
        "e.g",
        "i.e",
        "eg",
        "ie",
        "mr",
        "mrs",
        "ms",
        "dr",
        "govt",
        "ltd",
        "co",
        "inc",
        "para",
        "paras",
        "ch",
        "sch",
        "u/s",
        "approx",
        "fig",
        "p",
        "pp",
    }
)
_END_RE = re.compile(
    r"[.!?]+[\"'”’)\]]*"  # terminator plus closing quotes/brackets
    r"(?P<cites>(?:\s*" + CITATION_GROUP_RE.pattern + r")*)"  # citations after the full stop
    r"(?=\s+[\"'“‘(A-Z0-9]|\s*$)"
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")
_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z./]*)$")


def normalize_label(value: Any) -> str | None:
    """'C2', '[C2]', 'c2', 2 or '2' -> 'C2'; anything else -> None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"C{value}" if value > 0 else None
    if not isinstance(value, str):
        return None
    m = _LOOSE_LABEL_RE.match(value)
    if not m or int(m.group(1)) <= 0:
        return None
    return f"C{int(m.group(1))}"


def normalize_citation_brackets(text: str) -> str:
    """Rewrite fullwidth citation tags ("【C3】", "【C3†L1-L4】") as "[C3]". Answers that use
    ASCII brackets (every answer of the local model) are returned unchanged."""
    if "【" not in text:
        return text
    return _FULLWIDTH_GROUP_RE.sub(lambda m: f"[{m.group(1)}]", text)


def labels_in(text: str) -> list[str]:
    """Citation labels in ``text`` in order of first appearance, e.g. ['C2', 'C1']."""
    out: list[str] = []
    for group in CITATION_GROUP_RE.findall(text):
        for num in _LABEL_NUM_RE.findall(group):
            label = f"C{int(num)}"
            if label not in out:
                out.append(label)
    return out


def strip_citations(text: str) -> str:
    """Remove citation tags and tidy the whitespace they leave behind."""
    out = CITATION_GROUP_RE.sub("", text)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return re.sub(r"\s+", " ", out).strip()


@dataclass
class Sentence:
    raw: str  # with citation tags
    text: str  # citation tags removed
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "citations": list(self.labels)}


def _is_abbreviation(before: str) -> bool:
    m = _WORD_BEFORE_RE.search(before)
    if not m:
        return False
    word = m.group(1).lower().rstrip(".")
    return word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())


def _split_line(line: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for m in _END_RE.finditer(line):
        if _is_abbreviation(line[start : m.start()]):
            continue
        piece = line[start : m.end()].strip()
        if piece:
            parts.append(piece)
        start = m.end()
    rest = line[start:].strip()
    if rest:
        parts.append(rest)
    return parts


def split_sentences(answer: str) -> list[Sentence]:
    """Split an answer into sentences; each keeps the citation tags that end it.

    Lines are split separately (bullet markers removed); a line ending with ':' is a lead-in and
    is joined with the next line; a piece made only of citation tags is attached to the
    preceding sentence."""
    lines = [_BULLET_RE.sub("", ln).strip() for ln in answer.replace("\r", "").split("\n")]
    lines = [ln for ln in lines if ln]
    merged: list[str] = []
    for ln in lines:
        if merged and merged[-1].endswith(":"):
            merged[-1] = f"{merged[-1]} {ln}"
        else:
            merged.append(ln)
    pieces: list[str] = []
    for ln in merged:
        for piece in _split_line(ln):
            if pieces and not strip_citations(piece) and labels_in(piece):
                pieces[-1] = f"{pieces[-1]} {piece}"
            else:
                pieces.append(piece)
    out = []
    for p in pieces:
        text = strip_citations(p)
        if text or labels_in(p):
            out.append(Sentence(raw=p, text=text, labels=labels_in(p)))
    return out
