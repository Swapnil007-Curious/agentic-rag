"""The Markdown subset used for model answers: structure is kept, nothing executable survives
(script tags stay text, no attributes can be injected, javascript: links are inert text)."""

from __future__ import annotations

import json

from ragsys.app.markdown import citation_labels, lang_tag, parse_inline, parse_markdown, render_html


def test_blocks_and_inline_formatting() -> None:
    md = (
        "# Title\n\nSome **bold** and *italic* text with `code()`.\n\n"
        "- one\n- two\n  - nested\n1. first\n2. second\n\n"
        "```python\ndef f(x):\n    return x * 2\n```\n\nCited [C1] and [C12]."
    )
    blocks = parse_markdown(md)
    kinds = [b["type"] for b in blocks]
    assert kinds == ["heading", "paragraph", "list", "list", "code", "paragraph"]
    assert blocks[0]["level"] == 1
    para = blocks[1]["inlines"]
    assert {"t": "strong", "c": [{"t": "text", "v": "bold"}]} in para
    assert {"t": "em", "c": [{"t": "text", "v": "italic"}]} in para
    assert {"t": "code", "v": "code()"} in para
    assert blocks[2]["items"][1]["sub"]["items"][0]["inlines"] == [{"t": "text", "v": "nested"}]
    assert blocks[3]["ordered"] is True
    assert blocks[4] == {"type": "code", "lang": "python", "text": "def f(x):\n    return x * 2"}
    assert citation_labels(blocks) == ["C1", "C12"]


def test_script_tags_are_text() -> None:
    md = "<script>alert('x')</script>\n\n```html\n<script>evil()</script>\n```"
    html = render_html(parse_markdown(md))
    assert "<script" not in html and "&lt;script&gt;" in html
    assert "&lt;script&gt;evil()&lt;/script&gt;" in html


def test_attribute_injection_is_impossible() -> None:
    md = '**x" onmouseover="alert(1)** <img src=x onerror=alert(1)>\n\n```python" onload="x\nprint(1)\n```'
    html = render_html(parse_markdown(md))
    assert "<img" not in html and "&lt;img" in html
    assert ' onload="' not in html and ' onmouseover="' not in html  # quotes are escaped
    assert lang_tag('python" onload="x') == "pythononloadx"  # only [A-Za-z0-9_+#.-]
    for tag in ("<p>", "<strong>", "<pre"):
        assert tag in html
    # every tag is bare (data-lang on code blocks is the only attribute and is sanitised)
    import re

    attrs = re.findall(r"<([a-z0-9]+)(\s[^>]*)?>", html)
    assert all(not a or a.startswith(' data-lang="') for _t, a in attrs)


def test_javascript_links_are_inert_text() -> None:
    tokens = parse_inline(
        "see [click me](javascript:alert(document.cookie)) and [ok](https://e.org)"
    )
    assert all(t["t"] == "text" for t in tokens)
    text = "".join(t["v"] for t in tokens)
    assert "click me (javascript:alert(document.cookie" in text and "ok (https://e.org)" in text
    html = render_html(parse_markdown("[click](javascript:alert(1))"))
    assert "<a" not in html and "href" not in html


def test_tables_only_for_summaries_and_json_safe() -> None:
    md = "| a | b |\n|---|---|\n| 1 | **2** |\n"
    assert parse_markdown(md)[0]["type"] == "paragraph"
    t = parse_markdown(md, tables=True)[0]
    assert t["type"] == "table" and len(t["rows"]) == 1 and t["rows"][0][1][0]["t"] == "strong"
    json.dumps(parse_markdown("x" * 10, tables=True))
    assert len(parse_markdown("y" * 100_000)[0]["inlines"][0]["v"]) <= 60_000
