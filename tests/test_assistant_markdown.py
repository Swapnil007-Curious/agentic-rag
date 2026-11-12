"""The chat page's Markdown renderer (static/chat-md.js, run under Node): HTML in model output
stays text, event-handler attributes and javascript: links are inert, code and math are kept
verbatim; and the static assets use no HTML-string sinks, no inline script and no external
resources, with the vendored libraries' licence files present."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "src" / "ragsys" / "app" / "static"
NODE = shutil.which("node")


def render(texts: list[str]) -> list[dict[str, Any]]:
    if NODE is None:
        pytest.skip("node is not installed")
    proc = subprocess.run(
        [NODE, str(REPO / "tests" / "js" / "render_markdown.js")],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    out: list[dict[str, Any]] = json.loads(proc.stdout)
    return out


def links(tree: Any) -> list[str]:
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("t") == "link":
                found.append(node["href"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(tree)
    return found


def test_html_in_model_output_is_inert() -> None:
    cases = [
        "<script>alert(1)</script>",
        '<img src=x onerror="alert(1)">',
        '<a href="javascript:alert(1)">x</a> <svg onload=alert(1)>',
        "**bold <b onmouseover=alert(1)>x</b>**",
        "| a | <iframe src=//evil> |\n|---|---|\n| <script>x</script> | 2 |",
    ]
    for _case, out in zip(cases, render(cases), strict=True):
        html = out["html"]
        assert "<script" not in html and "<img" not in html and "<iframe" not in html, html
        assert "<svg" not in html and "<b " not in html
        assert not re.search(r"<[^>]*\son\w+\s*=", html), html  # no event-handler attribute
        assert "&lt;" in html
        assert links(out["tree"]) == []


def test_only_http_and_mailto_links_become_links() -> None:
    cases = [
        "[click](javascript:alert(1))",
        "[click](javascript:alert`1`)",
        "[x](data:text/html;base64,PHNjcmlwdD4=)",
        "[x](vbscript:msgbox)",
        "[x](JaVaScRiPt:alert(1))",
        "[docs](https://example.com/a?b=1) and <https://example.org> and mailto [m](mailto:a@b.c)",
        "see https://example.net/page.",
    ]
    outs = render(cases)
    for out in outs[:5]:
        assert links(out["tree"]) == [] and "href" not in out["html"], out["html"]
    assert links(outs[5]["tree"]) == [
        "https://example.com/a?b=1",
        "https://example.org",
        "mailto:a@b.c",
    ]
    assert 'rel="noopener noreferrer nofollow"' in outs[5]["html"]
    assert links(outs[6]["tree"]) == ["https://example.net/page"]


def test_code_math_citations_and_structure() -> None:
    cases = [
        "```python\nprint('<b>')  # [C1]\n```",
        "Inline $x^2 + y$ and \\(a+b\\), prices $5 and $10.",
        "$$\n\\int_0^1 x\\,dx\n$$",
        "Fact [C1][C2]. Not a label [D1].",
        "1. one\n2. two\n   - nested\n\n> quote\n\n# Title\n\n---",
        "```js\nunclosed",
    ]
    outs = render(cases)
    assert outs[0]["tree"][0] == {
        "t": "code",
        "lang": "python",
        "v": "print('<b>')  # [C1]",
        "closed": True,
    }
    maths = [n for n in outs[1]["tree"][0]["c"] if n["t"] == "math"]
    assert [m["v"] for m in maths] == ["x^2 + y", "a+b"]  # "$5 and $10" is not math
    assert outs[2]["tree"][0]["t"] == "math" and outs[2]["tree"][0]["display"] is True
    assert outs[2]["tree"][0]["v"] == "\\int_0^1 x\\,dx"
    cites = [n["v"] for n in outs[3]["tree"][0]["c"] if n["t"] == "cite"]
    assert cites == ["C1", "C2"]
    kinds = [b["t"] for b in outs[4]["tree"]]
    assert kinds == ["list", "quote", "h", "hr"] and outs[4]["tree"][0]["ordered"] is True
    assert outs[4]["tree"][0]["items"][1][1]["t"] == "list"  # nested list
    assert outs[5]["tree"][0]["closed"] is False and outs[5]["tree"][0]["v"] == "unclosed"


def test_static_assets_are_safe() -> None:
    for name in ("chat.js", "chat-md.js"):
        js = (STATIC / name).read_text(encoding="utf-8")
        for sink in (
            ".innerHTML",
            ".outerHTML",
            "insertAdjacentHTML",
            "document.write",
            "eval(",
            "new Function",
        ):
            assert sink not in js, (name, sink)
    md = (STATIC / "chat-md.js").read_text(encoding="utf-8")
    assert "trust: false" in md and "createTextNode" in md and "highlightElement" in md
    html = (STATIC / "chat.html").read_text(encoding="utf-8")
    assert "<script>" not in html and "style=" not in html
    assert not re.search(r"<[^>]*\son[a-z]+\s*=", html)  # no inline event handlers
    for ref in re.findall(r'(?:src|href)="([^"]+)"', html):
        assert ref.startswith(("/static/", "/research", "#")), ref
    css = (STATIC / "chat.css").read_text(encoding="utf-8")
    assert (
        "prefers-color-scheme: dark" in css
        and ":focus-visible" in css
        and "@media (max-width: 420px)" in css
    )
    assert "http" not in css
    vendor = STATIC / "vendor"
    assert (vendor / "katex" / "LICENSE").read_text(encoding="utf-8").startswith("The MIT License")
    assert "BSD 3-Clause" in (vendor / "highlight" / "LICENSE").read_text(encoding="utf-8")
    assert (vendor / "katex" / "katex.min.js").exists() and (
        vendor / "katex" / "katex.min.css"
    ).exists()
    assert len(list((vendor / "katex" / "fonts").glob("*.woff2"))) >= 15
    katex_css = (vendor / "katex" / "katex.min.css").read_text(encoding="utf-8")
    assert "http" not in katex_css.replace("http://www.w3.org", "")  # fonts are local
