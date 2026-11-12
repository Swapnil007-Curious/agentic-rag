"""data/raw/SOURCES.csv: provenance of every raw PDF (hash, size, pages, URL)."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from ragsys.config import IngestConfig
from ragsys.ingest.extract import pdf_page_count

FIELDS = [
    "file_name",
    "doc_id",
    "official_title",
    "source_url",
    "sha256",
    "bytes",
    "pages",
    "downloaded_on",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_sources_rows(
    cfg: IngestConfig, raw_dir: Path, existing: Path | None = None
) -> list[dict[str, str]]:
    """One row per enabled document; keeps any downloaded_on dates already filled in."""
    previous: dict[tuple[str, str], str] = {}
    if existing is not None and existing.exists():
        with existing.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                previous[(row.get("file_name", ""), row.get("doc_id", ""))] = row.get(
                    "downloaded_on", ""
                )
    rows = []
    for d in cfg.enabled_documents():
        path = raw_dir / d.file
        present = path.exists()
        rows.append(
            {
                "file_name": d.file,
                "doc_id": d.doc_id,
                "official_title": d.title,
                "source_url": d.source_url,
                "sha256": sha256_file(path) if present else "",
                "bytes": str(path.stat().st_size) if present else "",
                "pages": str(pdf_page_count(path)) if present else "",
                "downloaded_on": previous.get((d.file, d.doc_id), ""),
            }
        )
    return rows


def write_sources_csv(cfg: IngestConfig, raw_dir: Path, out: Path) -> list[dict[str, str]]:
    rows = build_sources_rows(cfg, raw_dir, existing=out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


# --------------------------------------------------------------------------- textbooks

TEXTBOOK_FIELDS = [
    "file_name",
    "doc_id",
    "title",
    "authors",
    "edition",
    "licence",
    "licence_url",
    "licence_source",
    "source_url",
    "sha256",
    "bytes",
    "pages",
    "pages_kept",
    "chapters",
    "downloaded_on",
]


def _chapters_cell(ranges: list[dict[str, Any]]) -> str:
    cells = []
    for r in ranges:
        if r.get("found"):
            cells.append(f"{r['title']} (pp. {r['start']}-{r['end']})")
        else:
            cells.append(f"{r['title']} (not found)")
    return "; ".join(cells)


def build_textbook_rows(
    cfg: IngestConfig, raw_dir: Path, report: dict[str, Any]
) -> list[dict[str, str]]:
    """One row per enabled document, with the licence and chapter selection from ingestion."""
    docs = report.get("documents", {})
    rows = []
    for d in cfg.enabled_documents():
        path = raw_dir / d.file
        present = path.exists()
        rep = docs.get(d.doc_id, {})
        lic = rep.get("licence") or {}
        chapters = _chapters_cell(rep.get("chapter_ranges", []))
        if d.doc_type != "textbook":
            # statutes in a textbook-format SOURCES.csv: licence from the config (the PDFs
            # state none) and a structure summary instead of chapter ranges
            lic = {
                "licence": d.licence or "unknown",
                "licence_url": d.licence_url,
                "source": d.licence_note or "not stated in the PDF",
            }
            num = rep.get("numbering") or {}
            if rep:
                chapters = (
                    f"{num.get('count', 0)} {num.get('unit', 'section')}s (highest number "
                    f"{num.get('max_number')}); {rep.get('chapters', 0)} parts/chapters; "
                    f"{rep.get('schedules', 0)} schedules"
                )
        rows.append(
            {
                "file_name": d.file,
                "doc_id": d.doc_id,
                "title": d.title,
                "authors": d.authors,
                "edition": d.edition,
                "licence": str(lic.get("licence", d.licence)),
                "licence_url": str(lic.get("licence_url", d.licence_url)),
                "licence_source": str(lic.get("source", "")),
                "source_url": d.source_url,
                "sha256": sha256_file(path) if present else "",
                "bytes": str(path.stat().st_size) if present else "",
                "pages": str(pdf_page_count(path)) if present else "",
                "pages_kept": str(rep.get("pages_kept", "")),
                "chapters": chapters,
                "downloaded_on": d.downloaded_on,
            }
        )
    return rows


def write_textbook_sources_csv(
    cfg: IngestConfig, raw_dir: Path, out: Path, report: dict[str, Any]
) -> list[dict[str, str]]:
    rows = build_textbook_rows(cfg, raw_dir, report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TEXTBOOK_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows
