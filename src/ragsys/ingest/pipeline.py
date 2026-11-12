"""End-to-end ingestion: PDF -> cleaned text -> structure nodes -> cross-references."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragsys.config import DocumentConfig, IngestConfig
from ragsys.ingest.chapters import outline_chapter_titles, resolve_chapters
from ragsys.ingest.clean import CleanDocument, Footnote, clean_document
from ragsys.ingest.extract import RawDocument, extract_pdf, pdf_page_count
from ragsys.ingest.licence import resolve_licence
from ragsys.ingest.textbook import clean_textbook
from ragsys.structure.base import Node, ParseResult, get_parser
from ragsys.structure.statute import numbering_report
from ragsys.structure.textbook_xrefs import extract_textbook_xrefs
from ragsys.structure.xrefs import XRef, XRefResolver, extract_xrefs, xref_summary
from ragsys.utils.runinfo import collect_run_info


@dataclass
class IngestResult:
    nodes: list[Node] = field(default_factory=list)
    xrefs: list[XRef] = field(default_factory=list)
    footnotes: list[Footnote] = field(default_factory=list)
    clean_docs: dict[str, CleanDocument] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _link_footnotes(footnotes: list[Footnote], nodes: list[Node]) -> None:
    for fn in footnotes:
        fn.section_ids = [
            n.node_id
            for n in nodes
            if n.doc_id == fn.doc_id and n.is_unit and n.page_start <= fn.page <= n.page_end
        ]


def run_ingest(
    cfg: IngestConfig, raw_dir: Path, processed_dir: Path, write: bool = True
) -> IngestResult:
    clean_dir = processed_dir / "clean"
    structure_dir = processed_dir / "structure"
    result = IngestResult()
    per_doc: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    raw_cache: dict[str, RawDocument] = {}
    doc_types: dict[str, str] = {}
    parent_acts: dict[str, str | None] = {}

    for d in cfg.enabled_documents():
        path = raw_dir / d.file
        if not path.exists():
            skipped.append({"doc_id": d.doc_id, "file": d.file, "reason": "file not found"})
            continue
        if d.doc_type == "textbook":
            tb = _ingest_textbook(cfg, d, path)
            if tb is None:
                skipped.append(
                    {
                        "doc_id": d.doc_id,
                        "file": d.file,
                        "reason": "no text layer (OCR out of scope)",
                    }
                )
                continue
            cd, parsed, rep = tb
            _link_footnotes(cd.footnotes, parsed.nodes)
            result.clean_docs[d.doc_id] = cd
            result.nodes.extend(parsed.nodes)
            result.footnotes.extend(cd.footnotes)
            doc_types[d.doc_id] = d.doc_type
            parent_acts[d.doc_id] = None
            per_doc[d.doc_id] = rep
            if write:
                _write_clean(clean_dir, d.doc_id, cd)
            continue
        raw = raw_cache.get(d.file)
        if raw is None:
            raw = extract_pdf(path, cfg.cleaning.marker_max_size_ratio)
            raw_cache[d.file] = raw
        if not raw.has_text_layer:
            skipped.append(
                {"doc_id": d.doc_id, "file": d.file, "reason": "no text layer (OCR out of scope)"}
            )
            continue
        cd = clean_document(raw, d, cfg.cleaning)
        parsed = get_parser(d.parser, cfg.structure).parse(cd, d)
        _link_footnotes(cd.footnotes, parsed.nodes)
        result.clean_docs[d.doc_id] = cd
        result.nodes.extend(parsed.nodes)
        result.footnotes.extend(cd.footnotes)
        doc_types[d.doc_id] = d.doc_type
        parent_acts[d.doc_id] = d.parent_act

        levels = Counter(n.level for n in parsed.nodes)
        units = [n for n in parsed.nodes if n.level in ("section", "rule")]
        per_doc[d.doc_id] = {
            "doc_id": d.doc_id,
            "file": d.file,
            "title": d.title,
            "doc_type": d.doc_type,
            "parser": d.parser,
            "pages": raw.page_count,
            "pages_dropped_devanagari": cd.stats.get("pages_dropped_devanagari", []),
            "chapters": levels.get("chapter", 0),
            "sections_or_rules": len(units),
            "definitions": sum(1 for n in parsed.nodes if n.defined_term),
            "schedules": levels.get("schedule", 0),
            "nodes_total": len(parsed.nodes),
            "nodes_per_level": dict(sorted(levels.items())),
            "numbering": numbering_report(parsed.nodes) if units else None,
            "footnotes": len(cd.footnotes),
            "clean_chars": len(cd.text),
            "cleaning": cd.stats,
            "parser_stats": parsed.stats,
            "warnings": [*cd.warnings, *parsed.warnings],
        }
        if cfg.sources_format == "textbook":  # statutes listed in a textbook-format SOURCES.csv
            per_doc[d.doc_id]["pages_kept"] = len({p for _a, _b, p in cd.page_spans})
        if write:
            clean_dir.mkdir(parents=True, exist_ok=True)
            (clean_dir / f"{d.doc_id}.txt").write_text(cd.text, encoding="utf-8", newline="\n")
            (clean_dir / f"{d.doc_id}.pages.json").write_text(
                json.dumps({"doc_id": d.doc_id, "page_spans": cd.page_spans}), encoding="utf-8"
            )

    resolver = XRefResolver(result.nodes, doc_types, parent_acts, cfg.structure.act_aliases)
    for doc_id, cd in result.clean_docs.items():
        if doc_types[doc_id] == "textbook":
            doc_xrefs = extract_textbook_xrefs(doc_id, cd.text, result.nodes)
        else:
            doc_xrefs = extract_xrefs(doc_id, cd.text, result.nodes, resolver)
        result.xrefs.extend(doc_xrefs)
        summary = xref_summary(doc_xrefs)
        per_doc[doc_id]["xrefs"] = summary
        unresolved = [x for x in doc_xrefs if x.resolution == "unresolved"]
        if unresolved:
            sample = sorted({x.raw_text for x in unresolved})[:8]
            per_doc[doc_id]["xrefs_unresolved_sample"] = sample

    for rep in per_doc.values():
        num = rep.get("numbering")
        if num and num["missing"]:
            rep["warnings"] = [w for w in rep["warnings"] if not w.startswith("gaps in")]
            rep["warnings"].append(f"gaps in {num['unit']} numbering: missing {num['missing']}")

    report: dict[str, Any] = {
        "documents": per_doc,
        "skipped": skipped,
        "totals": {
            "documents": len(per_doc),
            "nodes": len(result.nodes),
            "xrefs": xref_summary(result.xrefs),
            "footnotes": len(result.footnotes),
        },
        "config": cfg.model_dump(mode="json"),
        "run_info": collect_run_info(),
    }
    result.report = report
    if write:
        _write_jsonl(structure_dir / "nodes.jsonl", [n.model_dump() for n in result.nodes])
        _write_jsonl(structure_dir / "xrefs.jsonl", [x.model_dump() for x in result.xrefs])
        _write_jsonl(
            structure_dir / "footnotes.jsonl",
            [
                {
                    "footnote_id": f.footnote_id,
                    "doc_id": f.doc_id,
                    "page": f.page,
                    "marker": f.marker,
                    "text": f.text,
                    "section_ids": f.section_ids,
                }
                for f in result.footnotes
            ],
        )
        (structure_dir / "structure_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return result


def _write_clean(clean_dir: Path, doc_id: str, cd: CleanDocument) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    (clean_dir / f"{doc_id}.txt").write_text(cd.text, encoding="utf-8", newline="\n")
    (clean_dir / f"{doc_id}.pages.json").write_text(
        json.dumps({"doc_id": doc_id, "page_spans": cd.page_spans}), encoding="utf-8"
    )


def _ingest_textbook(
    cfg: IngestConfig, d: DocumentConfig, path: Path
) -> tuple[CleanDocument, ParseResult, dict[str, Any]] | None:
    """Chapter selection -> extraction of those pages -> textbook cleaning -> structure."""
    titles = list(d.chapters)
    warnings: list[str] = []
    if not titles and d.outline_chapters:
        titles = outline_chapter_titles(path, d.outline_chapters)
        if not titles:
            warnings.append(f"no level-1 outline entry matches {d.outline_chapters!r}")
    ranges = resolve_chapters(path, titles) if titles else []
    warnings += [
        f"chapter {r.title!r} not found in the outline or headings" for r in ranges if not r.found
    ]
    pages = sorted({p for r in ranges for p in r.pages}) if titles else None
    raw = extract_pdf(path, cfg.cleaning.marker_max_size_ratio, pages=pages, scripts="caret")
    if not raw.has_text_layer:
        return None
    cd = clean_textbook(raw, d, cfg.cleaning, cfg.textbook, ranges)
    parsed = get_parser(d.parser, cfg.structure).parse(cd, d)
    licence = resolve_licence(path, d)
    levels = Counter(n.level for n in parsed.nodes)
    rep: dict[str, Any] = {
        "doc_id": d.doc_id,
        "file": d.file,
        "title": d.title,
        "authors": d.authors,
        "edition": d.edition,
        "doc_type": d.doc_type,
        "parser": d.parser,
        "pages": pdf_page_count(path),
        "pages_kept": raw.page_count,
        "chapter_ranges": [r.to_dict() for r in ranges],
        "chapters_not_found": [r.title for r in ranges if not r.found],
        "printed_page_offset": cd.stats.get("printed_page_offset"),
        "pages_dropped_devanagari": [],
        "chapters": levels.get("chapter", 0),
        "sections_or_rules": levels.get("section", 0),
        "definitions": 0,
        "schedules": 0,
        "code_blocks": levels.get("code", 0),
        "headings": cd.stats.get("headings", 0),
        "sections_dropped": cd.stats.get("sections_dropped", []),
        "nodes_total": len(parsed.nodes),
        "nodes_per_level": dict(sorted(levels.items())),
        "numbering": None,
        "footnotes": len(cd.footnotes),
        "clean_chars": len(cd.text),
        "licence": licence.to_dict(),
        "cleaning": cd.stats,
        "parser_stats": parsed.stats,
        "warnings": [*warnings, *cd.warnings, *parsed.warnings],
    }
    return cd, parsed, rep


# --------------------------------------------------------------------------- loading helpers


def load_nodes(path: Path) -> list[Node]:
    with path.open(encoding="utf-8") as fh:
        return [Node.model_validate_json(line) for line in fh if line.strip()]


def load_xrefs(path: Path) -> list[XRef]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [XRef.model_validate_json(line) for line in fh if line.strip()]


def load_footnotes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_page_spans(path: Path) -> list[tuple[int, int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(int(a), int(b), int(p)) for a, b, p in data["page_spans"]]
