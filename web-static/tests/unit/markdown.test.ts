import { describe, expect, it } from "vitest";
import { parse, toHtml, type Block, type Inline } from "../../src/markdown.ts";

function links(blocks: Block[]): string[] {
  const out: string[] = [];
  const walk = (nodes: Inline[]): void => {
    for (const n of nodes) {
      if (n.t === "link") out.push(n.href);
      if ("c" in n) walk(n.c);
    }
  };
  for (const b of blocks) {
    if (b.t === "p" || b.t === "h") walk(b.c);
    else if (b.t === "quote") out.push(...links(b.c));
    else if (b.t === "list") b.items.forEach((it) => out.push(...links(it)));
    else if (b.t === "table") [...b.head, ...b.rows.flat()].forEach(walk);
  }
  return out;
}

describe("markdown sanitiser", () => {
  it("shows HTML as text and never creates event handlers", () => {
    for (const text of ["<script>alert(1)</script>", '<img src=x onerror="alert(1)">', "**bold <b onmouseover=alert(1)>x</b>**", "| a | <iframe src=//evil> |\n|---|---|\n| <script>x</script> | 2 |"]) {
      const html = toHtml(parse(text));
      expect(html).not.toMatch(/<(script|img|iframe|b )/);
      expect(html).not.toMatch(/<[^>]*\son\w+\s*=/);
      expect(html).toContain("&lt;");
      expect(links(parse(text))).toEqual([]);
    }
  });
  it("keeps only http(s) and mailto links", () => {
    for (const text of ["[x](javascript:alert(1))", "[x](JaVaScRiPt:alert(1))", "[x](data:text/html;base64,PHNjcmlwdD4=)", "[x](vbscript:msgbox)"]) {
      expect(links(parse(text))).toEqual([]);
      expect(toHtml(parse(text))).not.toContain("href");
    }
    const ok = parse("[docs](https://example.com/a?b=1) and <https://example.org> and [m](mailto:a@b.c) see https://example.net/page.");
    expect(links(ok)).toEqual(["https://example.com/a?b=1", "https://example.org", "mailto:a@b.c", "https://example.net/page"]);
    expect(toHtml(ok)).toContain('rel="noopener noreferrer nofollow"');
  });
  it("parses code, math, citations and structure", () => {
    expect(parse("```python\nprint('<b>')  # [C1]\n```")[0]).toEqual({ t: "code", lang: "python", v: "print('<b>')  # [C1]", closed: true });
    const inline = parse("Inline $x^2 + y$ and \\(a+b\\), prices $5 and $10.")[0];
    expect(inline?.t === "p" ? inline.c.filter((n) => n.t === "math").map((n) => (n.t === "math" ? n.v : "")) : []).toEqual(["x^2 + y", "a+b"]);
    const display = parse("$$\n\\int_0^1 x\\,dx\n$$")[0];
    expect(display).toMatchObject({ t: "math", display: true, v: "\\int_0^1 x\\,dx" });
    const cites = parse("Fact [C1][C2]. Not a label [D1].")[0];
    expect(cites?.t === "p" ? cites.c.filter((n) => n.t === "cite").map((n) => (n.t === "cite" ? n.v : "")) : []).toEqual(["C1", "C2"]);
    const kinds = parse("1. one\n2. two\n   - nested\n\n> quote\n\n# Title\n\n---").map((b) => b.t);
    expect(kinds).toEqual(["list", "quote", "h", "hr"]);
    expect(parse("```js\nunclosed")[0]).toMatchObject({ t: "code", closed: false, v: "unclosed" });
  });
});
