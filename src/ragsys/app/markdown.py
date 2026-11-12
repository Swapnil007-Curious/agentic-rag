"""A small, safe Markdown subset for model answers and SUMMARY.md pages.

Model text is never turned into HTML strings for the browser. :func:`parse_markdown` returns a
JSON-serialisable block tree that the page builds with DOM text nodes; nothing in it can carry
markup, attributes or URLs that the browser would follow:

* blocks: ``heading`` (levels 1-6), ``paragraph``, ``list`` (ordered or not, one nesting level),
  ``code`` (fenced; the language tag is reduced to ``[A-Za-z0-9_+#.-]``, at most 20 characters),
  ``table`` (only with ``tables=True``, for SUMMARY.md) and ``rule``;
* inline: ``text``, ``code``, ``strong``, ``em`` and ``cite`` (``[C3]`` passage labels);
  links ``[label](url)`` become plain text ``label (url)``, so ``javascript:`` URLs are inert;
* raw HTML is not interpreted: ``<script>`` stays literal text.

:func:`render_html` renders the tree with every text escaped (used by tests and as a fallback);
it emits only bare tags (no attributes except a sanitised ``data-lang`` on code blocks).
"""

from __future__ import annotations

import html
import re
from typing import Any

Block = dict[str, Any]
Inline = dict[str, Any]

_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})\s*([^\s`]*)?.*$")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d{1,3}[.)])\s+(.*)$")
_RULE_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_LANG_RE = re.compile(r"[^A-Za-z0-9_+#.-]")
_INLINE_RE = re.compile(
    r"(?P<code>`+)(?P<codetext>.+?)(?P=code)"
    r"|\[(?P<cite>C\d{1,3})\]"
    r"|\[(?P<ltext>[^\]\n]{1,300})\]\((?P<url>[^)\s]{0,500})\)"
    r"|(?P<sb>\*\*|__)(?P<stext>.+?)(?P=sb)"
    r"|(?<![\w*])(?P<eb>[*_])(?P<etext>[^*_\s](?:.*?[^*_\s])?)(?P=eb)(?![\w*])",
    re.DOTALL,
)
MAX_CHARS = 60_000


def lang_tag(raw: str | None) -> str:
    return _LANG_RE.sub("", raw or "")[:20]


def parse_inline(text: str, depth: int = 0) -> list[Inline]:
    """Inline tokens of one line or paragraph (see the module docstring)."""
    out: list[Inline] = []
    pos = 0

    def add_text(t: str) -> None:
        if not t:
            return
        if out and out[-1]["t"] == "text":
            out[-1]["v"] += t
        else:
            out.append({"t": "text", "v": t})

    for m in _INLINE_RE.finditer(text):
        add_text(text[pos : m.start()])
        pos = m.end()
        if m.group("code"):
            out.append({"t": "code", "v": m.group("codetext")})
        elif m.group("cite"):
            out.append({"t": "cite", "v": m.group("cite")})
        elif m.group("ltext") is not None:
            label, url = m.group("ltext"), m.group("url")
            add_text(f"{label} ({url})" if url and url != label else label)
        elif m.group("sb"):
            inner = parse_inline(m.group("stext"), depth + 1) if depth < 3 else []
            out.append({"t": "strong", "c": inner or [{"t": "text", "v": m.group("stext")}]})
        else:
            inner = parse_inline(m.group("etext"), depth + 1) if depth < 3 else []
            out.append({"t": "em", "c": inner or [{"t": "text", "v": m.group("etext")}]})
    add_text(text[pos:])
    return out


def _cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def parse_markdown(text: str, *, tables: bool = False) -> list[Block]:
    """Block tree of ``text`` (see the module docstring)."""
    lines = (text or "")[:MAX_CHARS].replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Block] = []
    para: list[str] = []
    i = 0

    def flush() -> None:
        if para:
            blocks.append({"type": "paragraph", "inlines": parse_inline(" ".join(para))})
            para.clear()

    while i < len(lines):
        ln = lines[i]
        fm = _FENCE_RE.match(ln)
        if fm:
            flush()
            fence = fm.group(1)
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence[0] * len(fence)):
                body.append(lines[i])
                i += 1
            blocks.append({"type": "code", "lang": lang_tag(fm.group(2)), "text": "\n".join(body)})
            i += 1
            continue
        if not ln.strip():
            flush()
            i += 1
            continue
        hm = _HEADING_RE.match(ln)
        if hm:
            flush()
            blocks.append(
                {"type": "heading", "level": len(hm.group(1)), "inlines": parse_inline(hm.group(2))}
            )
            i += 1
            continue
        if _RULE_RE.match(ln):
            flush()
            blocks.append({"type": "rule"})
            i += 1
            continue
        if tables and "|" in ln and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1]):
            flush()
            header = [parse_inline(c) for c in _cells(ln)]
            rows: list[list[list[Inline]]] = []
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append([parse_inline(c) for c in _cells(lines[i])])
                i += 1
            blocks.append({"type": "table", "header": header, "rows": rows})
            continue
        im = _ITEM_RE.match(ln)
        if im:
            flush()
            ordered = im.group(2)[0].isdigit()
            base_indent = len(im.group(1))
            items: list[dict[str, Any]] = []
            while i < len(lines):
                m2 = _ITEM_RE.match(lines[i])
                if m2 and len(m2.group(1)) <= base_indent + 1:
                    if m2.group(2)[0].isdigit() != ordered:
                        break  # another kind of list starts
                    items.append({"inlines": parse_inline(m2.group(3)), "sub": None})
                    i += 1
                    continue
                if m2 and items:  # nested item (one level)
                    sub = items[-1]["sub"] or {
                        "type": "list",
                        "ordered": m2.group(2)[0].isdigit(),
                        "items": [],
                    }
                    sub["items"].append({"inlines": parse_inline(m2.group(3)), "sub": None})
                    items[-1]["sub"] = sub
                    i += 1
                    continue
                cont = lines[i]
                if cont.strip() and items and cont.startswith((" ", "\t")):
                    items[-1]["inlines"] += [{"t": "text", "v": " "}, *parse_inline(cont.strip())]
                    i += 1
                    continue
                break
            blocks.append({"type": "list", "ordered": ordered, "items": items})
            continue
        para.append(ln.strip())
        i += 1
    flush()
    return blocks


# --------------------------------------------------------------------------- HTML (escaped)


def _inline_html(tokens: list[Inline]) -> str:
    out: list[str] = []
    for t in tokens:
        kind = t["t"]
        if kind == "text":
            out.append(html.escape(t["v"]))
        elif kind == "code":
            out.append(f"<code>{html.escape(t['v'])}</code>")
        elif kind == "cite":
            out.append(f"<sup>[{html.escape(t['v'])}]</sup>")
        elif kind == "strong":
            out.append(f"<strong>{_inline_html(t['c'])}</strong>")
        elif kind == "em":
            out.append(f"<em>{_inline_html(t['c'])}</em>")
    return "".join(out)


def _list_html(block: Block) -> str:
    tag = "ol" if block["ordered"] else "ul"
    items = []
    for it in block["items"]:
        sub = _list_html(it["sub"]) if it.get("sub") else ""
        items.append(f"<li>{_inline_html(it['inlines'])}{sub}</li>")
    return f"<{tag}>{''.join(items)}</{tag}>"


def render_html(blocks: list[Block]) -> str:
    """Escaped HTML of a block tree (no attributes, no links)."""
    out: list[str] = []
    for b in blocks:
        t = b["type"]
        if t == "heading":
            n = min(6, max(1, int(b["level"])))
            out.append(f"<h{n}>{_inline_html(b['inlines'])}</h{n}>")
        elif t == "paragraph":
            out.append(f"<p>{_inline_html(b['inlines'])}</p>")
        elif t == "list":
            out.append(_list_html(b))
        elif t == "code":
            lang = lang_tag(b.get("lang"))
            attr = f' data-lang="{lang}"' if lang else ""
            out.append(f"<pre{attr}><code>{html.escape(b['text'])}</code></pre>")
        elif t == "table":
            head = "".join(f"<th>{_inline_html(c)}</th>" for c in b["header"])
            rows = "".join(
                "<tr>" + "".join(f"<td>{_inline_html(c)}</td>" for c in r) + "</tr>"
                for r in b["rows"]
            )
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
        elif t == "rule":
            out.append("<hr>")
    return "\n".join(out)


def citation_labels(blocks: list[Block]) -> list[str]:
    """Passage labels cited anywhere in the tree, in order of first appearance."""
    seen: list[str] = []

    def walk(tokens: list[Inline]) -> None:
        for t in tokens:
            if t["t"] == "cite" and t["v"] not in seen:
                seen.append(t["v"])
            elif t["t"] in ("strong", "em"):
                walk(t["c"])

    for b in blocks:
        if b["type"] in ("heading", "paragraph"):
            walk(b["inlines"])
        elif b["type"] == "list":
            for it in b["items"]:
                walk(it["inlines"])
                for sub in it["sub"]["items"] if it.get("sub") else []:
                    walk(sub["inlines"])
        elif b["type"] == "table":
            for c in [*b["header"], *[c for r in b["rows"] for c in r]]:
                walk(c)
    return seen
