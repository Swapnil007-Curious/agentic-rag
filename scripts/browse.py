"""Browse the parsed corpus (for writing evaluation questions).

Usage (PowerShell or any shell):
    python scripts/browse.py list --doc cpa2019        section/rule numbers + headings
    python scripts/browse.py show cpa2019:s35          full text, sub-node IDs inline, pages
    python scripts/browse.py show "cpa2019:s2(7)"      quote IDs that contain parentheses
    python scripts/browse.py search "unfair trade practice"   keyword search, top 20
    python scripts/browse.py defs [--doc ecom2020]     all defined terms with node IDs
    python scripts/browse.py --collection cs_ai list --doc d2l      chapters/sections of a book
    python scripts/browse.py --collection cs_ai show d2l:11.5       a textbook section
Output is plain text (no box-drawing characters) and safe for cp1252 consoles.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

from ragsys.chunking.base import node_label, textbook_label
from ragsys.collections import add_collection_arguments, collection_from_args, config_path
from ragsys.config import resolve_path
from ragsys.eval.validate import suggest_ids
from ragsys.ingest.pipeline import load_footnotes, load_nodes, load_xrefs
from ragsys.structure.base import Node
from ragsys.utils.console import emit

WIDTH = 100


def own_segments(node: Node, by_id: dict[str, Node], text: str) -> list[str]:
    """Text of ``node`` that is not inside any child, as ordered segments."""
    cursor = node.char_start
    segs: list[str] = []
    for cid in node.children:
        ch = by_id[cid]
        segs.append(text[cursor : ch.char_start])
        cursor = ch.char_end
    segs.append(text[cursor : node.char_end])
    return segs


def node_title(n: Node) -> str:
    if n.doc_type == "textbook":
        return textbook_label(n)
    if n.level in ("section", "rule"):
        return f"{'Section' if n.level == 'section' else 'Rule'} {n.number} - {n.heading}"
    if n.level == "chapter":
        return f"CHAPTER {n.number} - {n.heading or ''}".rstrip(" -")
    if n.level == "schedule":
        return f"Schedule {n.number}: {n.heading}"
    return node_label(n, n.doc_type)


def wrap(text: str, indent: str) -> str:
    return textwrap.fill(
        " ".join(text.split()),
        width=WIDTH,
        initial_indent=indent,
        subsequent_indent=indent + "    ",
    )


def cmd_list(nodes: list[Node], doc: str | None) -> None:
    docs = (
        [doc]
        if doc
        else sorted(
            {n.doc_id for n in nodes}, key=lambda d: min(n.order for n in nodes if n.doc_id == d)
        )
    )
    for d in docs:
        dn = [n for n in nodes if n.doc_id == d]
        if not dn:
            emit(f"no nodes for doc {d!r}; known: {sorted({n.doc_id for n in nodes})}")
            continue
        emit(f"== {d}: {dn[0].doc_title}")
        if dn[0].doc_type == "textbook":
            depth = {"chapter": 0, "section": 1, "subsection": 2, "heading": 3}
            for n in sorted(dn, key=lambda n: n.order):
                if n.level == "code":
                    continue
                pages = (
                    f"p.{n.page_start}"
                    if n.page_start == n.page_end
                    else f"pp.{n.page_start}-{n.page_end}"
                )
                indent = "  " * (depth.get(n.level, 3) + 1)
                emit(f"{indent}{n.node_id:24s} {node_title(n)}  [{pages}]")
            continue
        for n in sorted(dn, key=lambda n: n.order):
            pages = (
                f"p.{n.page_start}"
                if n.page_start == n.page_end
                else f"pp.{n.page_start}-{n.page_end}"
            )
            if n.level == "chapter":
                emit(f"  {n.node_id:22s} {node_title(n)}")
            elif n.is_unit:
                indent = "    " if n.parent_id else "  "
                emit(f"{indent}{n.node_id:22s} {node_title(n)}  [{pages}]")


def cmd_show(nodes: list[Node], node_id: str, processed: Path) -> int:
    by_id = {n.node_id: n for n in nodes}
    node = by_id.get(node_id)
    if node is None:
        sug = suggest_ids(node_id, list(by_id))
        emit(f"node {node_id!r} not found." + (f" Did you mean: {', '.join(sug)}" if sug else ""))
        return 1
    text = (processed / "clean" / f"{node.doc_id}.txt").read_text(encoding="utf-8")
    chain = []
    cur: Node | None = node
    while cur is not None:
        chain.append(cur)
        cur = by_id.get(cur.parent_id) if cur.parent_id else None
    path = " > ".join(node_title(n) for n in reversed(chain[1:]))
    emit(f"{node.node_id}   {node_title(node)}")
    emit(f"{node.doc_title}" + (f" > {path}" if path else ""))
    emit(f"pages {node.page_start}-{node.page_end}   chars {node.char_start}-{node.char_end}")
    emit("-" * WIDTH)

    last_pages = [(node.page_start, node.page_end)]

    def walk(n: Node, depth: int) -> None:
        segs = own_segments(n, by_id, text)
        indent = "  " * depth
        first = segs[0].strip()
        tag = n.node_id
        if (n.page_start, n.page_end) != last_pages[-1]:  # show pages where they change
            pages = (
                f"p.{n.page_start}"
                if n.page_start == n.page_end
                else f"pp.{n.page_start}-{n.page_end}"
            )
            tag = f"{n.node_id} | {pages}"
            last_pages.append((n.page_start, n.page_end))
        emit(wrap(f"[{tag}] {first}" if first else f"[{tag}]", indent))
        for i, cid in enumerate(n.children):
            walk(by_id[cid], depth + 1)
            tail = segs[i + 1].strip()
            if tail:
                emit(wrap(f"[{n.node_id}, continued] {tail}", indent))

    walk(node, 0)
    subtree = {node.node_id}
    stack = list(node.children)
    while stack:
        cid = stack.pop()
        subtree.add(cid)
        stack.extend(by_id[cid].children)
    xrefs = load_xrefs(processed / "structure" / "xrefs.jsonl")
    out_refs = sorted({x.target_node for x in xrefs if x.source_node in subtree and x.target_node})
    in_refs = sorted(
        {x.source_node for x in xrefs if x.target_node in subtree and x.source_node not in subtree}
    )
    emit("-" * WIDTH)
    if out_refs:
        emit(wrap("refers to: " + ", ".join(out_refs), ""))
    if in_refs:
        emit(wrap("referred to by: " + ", ".join(in_refs), ""))
    fns = [
        f
        for f in load_footnotes(processed / "structure" / "footnotes.jsonl")
        if f["doc_id"] == node.doc_id and node.page_start <= f["page"] <= node.page_end
    ]
    for f in fns:
        emit(wrap(f"footnote (p.{f['page']}): {f['text']}", ""))
    return 0


def cmd_search(nodes: list[Node], query: str, processed: Path, limit: int = 20) -> None:
    by_id = {n.node_id: n for n in nodes}
    texts: dict[str, str] = {}
    words = [w for w in re.findall(r"\w+", query.lower()) if w]
    phrase = re.compile(re.escape(" ".join(query.split())), re.IGNORECASE)
    hits: list[tuple[float, int, Node, str]] = []
    for n in nodes:
        if n.level == "chapter":
            continue
        if n.doc_id not in texts:
            texts[n.doc_id] = (processed / "clean" / f"{n.doc_id}.txt").read_text(encoding="utf-8")
        own = " ".join(own_segments(n, by_id, texts[n.doc_id]))
        found = phrase.findall(own)
        score = float(len(found))
        if not found and words and all(w in own.lower() for w in words):
            score = 0.5
        if n.defined_term and n.defined_term.lower() == " ".join(query.split()).lower():
            score += 100.0  # the definition of the searched term comes first
        if score > 0:
            m = phrase.search(own) or re.search(re.escape(words[0]), own, re.IGNORECASE)
            pos = m.start() if m else 0
            snippet = " ".join(own[max(0, pos - 70) : pos + 110].split())
            hits.append((score, len(own), n, snippet))
    hits.sort(key=lambda h: (-h[0], h[1], h[2].order))
    emit(f"{len(hits)} node(s) match {query!r}; showing top {min(limit, len(hits))}")
    for score, _ln, n, snip in hits[:limit]:
        tag = "definition" if score >= 100 else ("phrase" if score >= 1 else "all words")
        emit(f"{n.node_id:28s} p.{n.page_start:<3d} [{tag}]")
        emit(wrap(f"... {snip} ...", "    "))


def cmd_defs(nodes: list[Node], doc: str | None) -> None:
    defs = [
        n for n in sorted(nodes, key=lambda n: (n.doc_id != "cpa2019", n.order)) if n.defined_term
    ]
    if doc:
        defs = [n for n in defs if n.doc_id == doc]
    emit(f"{len(defs)} defined term(s)")
    for n in defs:
        emit(f"{n.node_id:26s} {n.defined_term}  [p.{n.page_start}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--processed-dir", default=None, help="default: the collection's")
    add_collection_arguments(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list", help="sections/rules with headings")
    p_list.add_argument("--doc", default=None)
    p_show = sub.add_parser("show", help="full text of a node with sub-node IDs")
    p_show.add_argument("node_id", nargs="+", help='e.g. cpa2019:s35 or "cpa2019:s35(1)"')
    p_search = sub.add_parser("search", help="case-insensitive keyword search")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--limit", type=int, default=20)
    p_defs = sub.add_parser("defs", help="all defined terms")
    p_defs.add_argument("--doc", default=None)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    coll = collection_from_args(args)
    processed = resolve_path(config_path(args.processed_dir, coll.processed_dir))
    nodes_path = processed / "structure" / "nodes.jsonl"
    if not nodes_path.exists():
        emit(f"{nodes_path} not found; run scripts/ingest.py first")
        return 1
    nodes = load_nodes(nodes_path)
    try:
        if args.cmd == "list":
            cmd_list(nodes, args.doc)
        elif args.cmd == "show":
            parts = args.node_id
            # PowerShell splits an unquoted cpa2019:s35(1) into "cpa2019:s35" and "1"
            node_id = parts[0] + "".join(f"({p.strip('()')})" for p in parts[1:])
            return cmd_show(nodes, node_id, processed)
        elif args.cmd == "search":
            cmd_search(nodes, " ".join(args.query), processed, args.limit)
        elif args.cmd == "defs":
            cmd_defs(nodes, args.doc)
    except BrokenPipeError:  # e.g. piped into Select-Object -First 15
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
