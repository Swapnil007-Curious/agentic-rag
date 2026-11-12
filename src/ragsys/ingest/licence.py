"""Licence of a textbook PDF, read from its licence/copyright page.

The first ``max_pages`` pages are searched for a licence statement (Creative Commons with its
elements and version, GNU FDL, Apache, MIT). A statement with a version is used as found, with
the page and the quoted sentence. When the PDF names no licence, or names one without a version,
the document's configured ``licence`` is used and the provenance says so (and quotes any
version-less mention), so SOURCES.csv never presents a configured value as read from the PDF.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf

from ragsys.config import DocumentConfig

CC_RE = re.compile(
    r"Creative\s+Commons\s+Attribution"
    r"(?P<elements>(?:[\s-]+(?:Non-?Commercial|Share-?Alike|No-?Deriv(?:ative)?s))*)"
    r"(?:\s+(?P<version>\d\.\d))?"
    r"(?:\s+(?P<port>International|Unported|Generic|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))?"
    r"\s+[Ll]icen[cs]e",
)
OTHER_RES = [
    (
        re.compile(r"GNU\s+Free\s+Documentation\s+License(?:,?\s+Version\s+(?P<version>\d\.\d))?"),
        "GFDL",
    ),
    (re.compile(r"Apache\s+License,?\s+Version\s+(?P<version>\d\.\d)"), "Apache"),
    (re.compile(r"\bMIT\s+License\b(?P<version>)"), "MIT"),
]


@dataclass
class LicenceInfo:
    licence: str  # e.g. "CC BY-NC-SA 3.0 Unported"
    licence_url: str
    source: str  # where it was read: "PDF p.7: ..." or "not stated in the PDF; ..."
    from_pdf: bool
    page: int | None = None
    quote: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sentence(text: str, start: int, end: int) -> str:
    """The sentence around a match, cut to at most ~90 characters on either side of it."""
    idx = text.rfind(". ", 0, start)
    a = max(idx + 2 if idx >= 0 else 0, start - 90)
    b = text.find(". ", end)
    b = min(len(text) if b < 0 else b + 1, end + 90)
    quote = " ".join(text[a:b].split())
    return ("..." if a > 0 and text[a - 1] not in ". " else "") + quote


def _cc_name(m: re.Match[str]) -> tuple[str, bool]:
    raw = re.sub(r"[\s-]+", "", (m.group("elements") or "").lower())
    els = []
    for word, tag in (("noncommercial", "NC"), ("noderiv", "ND"), ("sharealike", "SA")):
        if word in raw and tag not in els:
            els.append(tag)
    name = "CC BY" + ("-" + "-".join(els) if els else "")
    version = m.group("version")
    if version:
        name += f" {version}"
        port = m.group("port")
        if port and port.lower() not in ("license", "licence"):
            name += f" {port}"
    return name, bool(version)


def detect_licence(path: str | Path, max_pages: int = 40) -> tuple[LicenceInfo | None, str | None]:
    """(licence with a version read from the PDF, or None; quote of a version-less mention)."""
    vague: str | None = None
    with pymupdf.open(Path(path)) as doc:  # type: ignore[no-untyped-call]
        for index in range(min(max_pages, doc.page_count)):
            # words hyphenated across a line break ("Cre-\native Commons") are joined first
            raw = re.sub(r"(?<=\w)-\n(?=\w)", "", doc[index].get_text())
            text = " ".join(raw.split())
            for m in CC_RE.finditer(text):
                name, versioned = _cc_name(m)
                quote = _sentence(text, m.start(), m.end())
                if versioned:
                    return (
                        LicenceInfo(
                            licence=name,
                            licence_url="",
                            source=f'PDF p.{index + 1}: "{quote}"',
                            from_pdf=True,
                            page=index + 1,
                            quote=quote,
                        ),
                        vague,
                    )
                if vague is None:
                    vague = f'p.{index + 1}: "{quote}"'
            for rx, label in OTHER_RES:
                m2 = rx.search(text)
                if m2:
                    version = m2.group("version") or ""
                    quote = _sentence(text, m2.start(), m2.end())
                    if version or label == "MIT":
                        name = f"{label} {version}".strip()
                        return (
                            LicenceInfo(
                                name, "", f'PDF p.{index + 1}: "{quote}"', True, index + 1, quote
                            ),
                            vague,
                        )
                    if vague is None:
                        vague = f'p.{index + 1}: "{quote}"'
    return None, vague


def resolve_licence(path: str | Path, doc: DocumentConfig) -> LicenceInfo:
    found, vague = detect_licence(path)
    if found is not None:
        url = (
            doc.licence_url
            if doc.licence and doc.licence.split()[:2] == found.licence.split()[:2]
            else ""
        )
        found.licence_url = url
        return found
    parts = ["not stated with a version in the PDF"]
    if vague:
        parts.append(f"the PDF only says {vague}")
    if doc.licence_note:
        parts.append(doc.licence_note)
    return LicenceInfo(
        licence=doc.licence or "unknown",
        licence_url=doc.licence_url,
        source="; ".join(parts),
        from_pdf=False,
    )
