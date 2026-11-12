// Safe Markdown for model replies: text -> plain node tree -> DOM (port of the Python app's
// src/ragsys/app/static/chat-md.js, same rules). Model output is never parsed as HTML: elements
// are built with createElement / textContent, so <script> or <img onerror=...> in a reply shows as
// text. Links keep only http(s) and mailto targets and open with rel="noopener noreferrer
// nofollow". Math is rendered by the vendored KaTeX (trust off); fenced code is highlighted by the
// vendored highlight.js on the final render. toHtml() serialises a tree with everything escaped.

export type Inline =
  | { t: "text"; v: string }
  | { t: "code"; v: string }
  | { t: "math"; v: string; display: boolean }
  | { t: "cite"; v: string }
  | { t: "link"; href: string; c: Inline[] }
  | { t: "strong" | "em" | "del"; c: Inline[] };

export type Block =
  | { t: "p"; c: Inline[] }
  | { t: "h"; level: number; c: Inline[] }
  | { t: "hr" }
  | { t: "code"; lang: string; v: string; closed: boolean }
  | { t: "math"; v: string; display: true; closed?: boolean }
  | { t: "quote"; c: Block[] }
  | { t: "list"; ordered: boolean; start: number; items: Block[][] }
  | { t: "table"; head: Inline[][]; rows: Inline[][][]; align: string[] };

interface KatexLike {
  render(tex: string, el: HTMLElement, options: Record<string, unknown>): void;
}
interface HljsLike {
  getLanguage(name: string): unknown;
  highlightElement(el: HTMLElement): void;
}
declare global {
  interface Window {
    katex?: KatexLike;
    hljs?: HljsLike;
  }
}

const MAX_DEPTH = 6;
export const SAFE_URL = /^(https?:\/\/|mailto:)[^\s<>"'`]+$/i;
const CITE_RE = /^C\d{1,3}$/;

const INLINE_SOURCE = [
  "(`+)([\\s\\S]*?[^`])\\1(?!`)", // 1,2 code span
  "\\\\\\(([\\s\\S]+?)\\\\\\)", // 3 \( math \)
  "\\$(?!\\s)([^$\\n]*?[^\\s$\\\\])\\$(?!\\d)", // 4 $math$
  "\\[(C\\d{1,3})\\]", // 5 citation
  "\\[([^\\]\\n]{1,300})\\]\\(([^()\\s]{1,2000})\\)", // 6,7 link
  "<(https?:\\/\\/[^\\s<>]{1,2000})>", // 8 autolink
  "(\\*\\*|__)(?=\\S)([\\s\\S]*?\\S)\\9", // 9,10 strong
  "~~(?=\\S)([\\s\\S]*?\\S)~~", // 11 strike
  "(?<![\\w*])(\\*|_)(?=[^\\s*_])([\\s\\S]*?[^\\s*_\\\\])\\12(?![\\w*])", // 12,13 em
  "(https?:\\/\\/[^\\s<>()\\[\\]]{2,2000}[^\\s<>()\\[\\].,;:!?'\"])", // 14 bare URL
].join("|");

export function parseInline(text: string, depth = 0): Inline[] {
  const out: Inline[] = [];
  let pos = 0;
  const pushText = (t: string): void => {
    if (!t) return;
    const last = out[out.length - 1];
    if (last && last.t === "text") last.v += t;
    else out.push({ t: "text", v: t });
  };
  const nested = (s: string): Inline[] => (depth < MAX_DEPTH ? parseInline(s, depth + 1) : [{ t: "text", v: s }]);
  const re = new RegExp(INLINE_SOURCE, "g");
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m[0] === "") {
      re.lastIndex++;
      continue;
    }
    pushText(text.slice(pos, m.index));
    pos = m.index + m[0].length;
    if (m[1] !== undefined) out.push({ t: "code", v: (m[2] ?? "").replace(/^ (.*) $/s, "$1") });
    else if (m[3] !== undefined) out.push({ t: "math", v: m[3], display: false });
    else if (m[4] !== undefined) out.push({ t: "math", v: m[4], display: false });
    else if (m[5] !== undefined) out.push({ t: "cite", v: m[5] });
    else if (m[6] !== undefined) {
      const url = m[7] ?? "";
      if (SAFE_URL.test(url)) out.push({ t: "link", href: url, c: nested(m[6]) });
      else pushText(`${m[6]} (${url})`);
    } else if (m[8] !== undefined) out.push({ t: "link", href: m[8], c: [{ t: "text", v: m[8] }] });
    else if (m[9] !== undefined) out.push({ t: "strong", c: nested(m[10] ?? "") });
    else if (m[11] !== undefined) out.push({ t: "del", c: nested(m[11]) });
    else if (m[12] !== undefined) out.push({ t: "em", c: nested(m[13] ?? "") });
    else if (m[14] !== undefined) out.push({ t: "link", href: m[14], c: [{ t: "text", v: m[14] }] });
  }
  pushText(text.slice(pos));
  return out;
}

const FENCE_RE = /^ {0,3}(`{3,}|~{3,})\s*([^\s`]*)/;
const HEADING_RE = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE_RE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
const ITEM_RE = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
const QUOTE_RE = /^ {0,3}> ?(.*)$/;
const TABLE_SEP_RE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
const MATH_OPEN_RE = /^\s*(\$\$|\\\[)(.*)$/;

function langTag(raw: string | undefined): string {
  return String(raw ?? "").replace(/[^A-Za-z0-9_+#.-]/g, "").slice(0, 20).toLowerCase();
}

function cells(line: string): string[] {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
  return s.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
}

function aligns(line: string): string[] {
  return cells(line).map((c) =>
    c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : c.startsWith(":") ? "left" : "",
  );
}

function isDigit(s: string | undefined): boolean {
  return /\d/.test(s ?? "");
}

function parseBlocks(lines: string[], depth: number): Block[] {
  const blocks: Block[] = [];
  const para: string[] = [];
  let i = 0;
  const flush = (): void => {
    if (para.length) {
      blocks.push({ t: "p", c: parseInline(para.join(" ").trim(), 0) });
      para.length = 0;
    }
  };
  while (i < lines.length) {
    const ln = lines[i] ?? "";
    const fm = FENCE_RE.exec(ln);
    if (fm) {
      flush();
      const fence = fm[1] ?? "```";
      const body: string[] = [];
      i++;
      let closed = false;
      while (i < lines.length) {
        const close = /^ {0,3}(`{3,}|~{3,})\s*$/.exec(lines[i] ?? "");
        const f = close?.[1];
        if (f && f[0] === fence[0] && f.length >= fence.length) {
          closed = true;
          i++;
          break;
        }
        body.push(lines[i] ?? "");
        i++;
      }
      blocks.push({ t: "code", lang: langTag(fm[2]), v: body.join("\n"), closed });
      continue;
    }
    const mm = MATH_OPEN_RE.exec(ln);
    if (mm) {
      const close = mm[1] === "$$" ? "$$" : "\\]";
      const rest = mm[2] ?? "";
      const end = rest.indexOf(close);
      if (end >= 0 && rest.slice(end + close.length).trim() === "") {
        flush();
        blocks.push({ t: "math", v: rest.slice(0, end).trim(), display: true });
        i++;
        continue;
      }
      if (end < 0) {
        const body = [rest];
        let j = i + 1;
        let found = false;
        while (j < lines.length) {
          const line = lines[j] ?? "";
          const k = line.indexOf(close);
          if (k >= 0) {
            body.push(line.slice(0, k));
            found = line.slice(k + close.length).trim() === "";
            break;
          }
          body.push(line);
          j++;
        }
        if (found || j >= lines.length) {
          flush();
          blocks.push({ t: "math", v: body.join("\n").trim(), display: true, closed: found });
          i = j + 1;
          continue;
        }
      }
    }
    if (!ln.trim()) {
      flush();
      i++;
      continue;
    }
    const hm = HEADING_RE.exec(ln);
    if (hm) {
      flush();
      blocks.push({ t: "h", level: (hm[1] ?? "#").length, c: parseInline(hm[2] ?? "", 0) });
      i++;
      continue;
    }
    if (RULE_RE.test(ln)) {
      flush();
      blocks.push({ t: "hr" });
      i++;
      continue;
    }
    if (QUOTE_RE.test(ln) && depth < MAX_DEPTH) {
      flush();
      const inner: string[] = [];
      while (i < lines.length && QUOTE_RE.test(lines[i] ?? "")) {
        inner.push(QUOTE_RE.exec(lines[i] ?? "")?.[1] ?? "");
        i++;
      }
      blocks.push({ t: "quote", c: parseBlocks(inner, depth + 1) });
      continue;
    }
    const next = lines[i + 1];
    if (ln.includes("|") && next !== undefined && TABLE_SEP_RE.test(next) && cells(ln).length === cells(next).length) {
      flush();
      const head = cells(ln).map((c) => parseInline(c, 0));
      const align = aligns(next);
      const rows: Inline[][][] = [];
      i += 2;
      while (i < lines.length && (lines[i] ?? "").includes("|") && (lines[i] ?? "").trim()) {
        rows.push(cells(lines[i] ?? "").map((c) => parseInline(c, 0)));
        i++;
      }
      blocks.push({ t: "table", head, rows, align });
      continue;
    }
    const im = ITEM_RE.exec(ln);
    if (im && depth < MAX_DEPTH) {
      flush();
      const baseIndent = (im[1] ?? "").length;
      const ordered = isDigit(im[2]?.[0]);
      const start = ordered ? Number.parseInt(im[2] ?? "1", 10) : 1;
      const items: string[][] = [];
      let cur: string[] | null = null;
      while (i < lines.length) {
        const l2 = lines[i] ?? "";
        const m2 = ITEM_RE.exec(l2);
        if (m2 && (m2[1] ?? "").length <= baseIndent + 1 && isDigit(m2[2]?.[0]) === ordered) {
          cur = [m2[3] ?? ""];
          items.push(cur);
          i++;
          continue;
        }
        if (m2 && (m2[1] ?? "").length <= baseIndent + 1) break;
        if (!l2.trim()) {
          const nxt = lines[i + 1];
          const nm = nxt !== undefined ? ITEM_RE.exec(nxt) : null;
          if (nxt !== undefined && cur && (/^\s{2,}\S/.test(nxt) || (nm && (nm[1] ?? "").length <= baseIndent + 1))) {
            cur.push("");
            i++;
            continue;
          }
          break;
        }
        if (cur && (/^\s+\S/.test(l2) || !FENCE_RE.test(l2))) {
          if (!/^\s/.test(l2) && (HEADING_RE.test(l2) || QUOTE_RE.test(l2) || RULE_RE.test(l2))) break;
          cur.push(l2.replace(new RegExp(`^\\s{0,${baseIndent + 4}}`), ""));
          i++;
          continue;
        }
        break;
      }
      blocks.push({ t: "list", ordered, start, items: items.map((it) => parseBlocks(it, depth + 1)) });
      continue;
    }
    para.push(ln.trim());
    i++;
  }
  flush();
  return blocks;
}

export function parse(text: string): Block[] {
  const src = String(text ?? "").slice(0, 100000).replace(/\r\n?/g, "\n");
  return parseBlocks(src.split("\n"), 0);
}

// ------------------------------------------------------------------ escaped HTML (tests)

const ESC: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (s: string): string => s.replace(/[&<>"']/g, (c) => ESC[c] ?? c);

function inlineHtml(nodes: Inline[]): string {
  return nodes
    .map((n) => {
      switch (n.t) {
        case "text":
          return esc(n.v);
        case "code":
          return `<code>${esc(n.v)}</code>`;
        case "math":
          return `<span class="math">${esc(n.v)}</span>`;
        case "cite":
          return `<button class="cite">${esc(n.v)}</button>`;
        case "link":
          return `<a href="${esc(n.href)}" rel="noopener noreferrer nofollow">${inlineHtml(n.c)}</a>`;
        default:
          return `<${n.t}>${inlineHtml(n.c)}</${n.t}>`;
      }
    })
    .join("");
}

export function toHtml(blocks: Block[]): string {
  return blocks
    .map((b) => {
      switch (b.t) {
        case "p":
          return `<p>${inlineHtml(b.c)}</p>`;
        case "h":
          return `<h${b.level}>${inlineHtml(b.c)}</h${b.level}>`;
        case "hr":
          return "<hr>";
        case "code":
          return `<pre><code class="language-${esc(b.lang)}">${esc(b.v)}</code></pre>`;
        case "math":
          return `<div class="math">${esc(b.v)}</div>`;
        case "quote":
          return `<blockquote>${toHtml(b.c)}</blockquote>`;
        case "list": {
          const tag = b.ordered ? "ol" : "ul";
          return `<${tag}>${b.items.map((it) => `<li>${toHtml(it)}</li>`).join("")}</${tag}>`;
        }
        case "table":
          return (
            `<table><thead><tr>${b.head.map((c) => `<th>${inlineHtml(c)}</th>`).join("")}</tr></thead><tbody>` +
            b.rows.map((r) => `<tr>${r.map((c) => `<td>${inlineHtml(c)}</td>`).join("")}</tr>`).join("") +
            "</tbody></table>"
          );
        default:
          return "";
      }
    })
    .join("");
}

// ------------------------------------------------------------------ DOM

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string | null, text?: string | null): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

export function copyText(text: string, button: HTMLElement, label = "Copy"): void {
  const done = (ok: boolean): void => {
    button.textContent = ok ? "Copied" : "Copy failed";
    setTimeout(() => {
      button.textContent = label;
    }, 1500);
  };
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text).then(
      () => done(true),
      () => done(false),
    );
    return;
  }
  done(false);
}

const KATEX_OPTIONS = { throwOnError: false, trust: false, strict: "ignore", maxSize: 20, maxExpand: 500, output: "htmlAndMathml" };

function mathNode(tex: string, display: boolean): HTMLElement {
  const node = el(display ? "div" : "span", display ? "math-block" : "math-inline");
  const k = window.katex;
  if (k && tex.length <= 5000) {
    try {
      k.render(tex, node, { ...KATEX_OPTIONS, displayMode: display });
      if (display) node.tabIndex = 0;
      return node;
    } catch {
      node.textContent = "";
    }
  }
  node.textContent = display ? tex : `$${tex}$`;
  node.classList.add("math-raw");
  return node;
}

export interface RenderOptions {
  cite?: (label: string) => HTMLElement | null;
  highlight?: boolean;
}

function inlineDom(nodes: Inline[], opts: RenderOptions, parent: HTMLElement): HTMLElement {
  for (const n of nodes) {
    if (n.t === "text") parent.appendChild(document.createTextNode(n.v));
    else if (n.t === "code") parent.appendChild(el("code", "md-code", n.v));
    else if (n.t === "math") parent.appendChild(mathNode(n.v, false));
    else if (n.t === "cite") {
      const made = opts.cite && CITE_RE.test(n.v) ? opts.cite(n.v) : null;
      parent.appendChild(made ?? document.createTextNode(`[${n.v}]`));
    } else if (n.t === "link") {
      if (SAFE_URL.test(n.href)) {
        const a = el("a");
        a.href = n.href;
        a.rel = "noopener noreferrer nofollow";
        a.target = "_blank";
        parent.appendChild(inlineDom(n.c, opts, a));
      } else inlineDom(n.c, opts, parent);
    } else parent.appendChild(inlineDom(n.c, opts, el(n.t)));
  }
  return parent;
}

function codeDom(b: Extract<Block, { t: "code" }>, opts: RenderOptions): HTMLElement {
  const wrap = el("div", "md-pre");
  const bar = el("div", "md-pre-bar");
  bar.appendChild(el("span", "md-lang", b.lang || "text"));
  const btn = el("button", "md-copy", "Copy");
  btn.type = "button";
  btn.setAttribute("aria-label", `Copy code${b.lang ? ` (${b.lang})` : ""}`);
  btn.addEventListener("click", () => copyText(b.v, btn));
  bar.appendChild(btn);
  wrap.appendChild(bar);
  const pre = el("pre");
  pre.tabIndex = 0;
  const code = el("code", b.lang ? `language-${b.lang}` : null, b.v);
  pre.appendChild(code);
  wrap.appendChild(pre);
  const hl = window.hljs;
  if (opts.highlight && hl && b.lang && hl.getLanguage(b.lang) && b.v.length < 60000) {
    try {
      hl.highlightElement(code);
    } catch {
      code.textContent = b.v;
    }
  }
  return wrap;
}

function tableDom(b: Extract<Block, { t: "table" }>, opts: RenderOptions): HTMLElement {
  const scroller = el("div", "md-table-wrap");
  scroller.tabIndex = 0;
  const t = el("table", "md-table");
  const thead = el("thead");
  const hr = el("tr");
  b.head.forEach((c, i) => {
    const th = inlineDom(c, opts, el("th")) as HTMLTableCellElement;
    th.scope = "col";
    if (b.align[i]) th.className = `align-${b.align[i]}`;
    hr.appendChild(th);
  });
  thead.appendChild(hr);
  t.appendChild(thead);
  const tbody = el("tbody");
  for (const r of b.rows) {
    const tr = el("tr");
    r.forEach((c, i) => {
      const td = inlineDom(c, opts, el("td"));
      if (b.align[i]) td.className = `align-${b.align[i]}`;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  t.appendChild(tbody);
  scroller.appendChild(t);
  return scroller;
}

function blocksDom(blocks: Block[], opts: RenderOptions, parent: HTMLElement | DocumentFragment): HTMLElement | DocumentFragment {
  for (const b of blocks) {
    if (b.t === "p") parent.appendChild(inlineDom(b.c, opts, el("p")));
    else if (b.t === "h") parent.appendChild(inlineDom(b.c, opts, el(`h${Math.min(6, b.level + 2)}` as "h3", "md-h")));
    else if (b.t === "hr") parent.appendChild(el("hr"));
    else if (b.t === "code") parent.appendChild(codeDom(b, opts));
    else if (b.t === "math") parent.appendChild(mathNode(b.v, true));
    else if (b.t === "quote") parent.appendChild(blocksDom(b.c, opts, el("blockquote")));
    else if (b.t === "table") parent.appendChild(tableDom(b, opts));
    else if (b.t === "list") {
      const list = el(b.ordered ? "ol" : "ul", "md-list");
      if (b.ordered && b.start !== 1) (list as HTMLOListElement).start = b.start;
      for (const item of b.items) {
        const li = el("li");
        const first = item[0];
        if (item.length === 1 && first?.t === "p") inlineDom(first.c, opts, li);
        else blocksDom(item, opts, li);
        list.appendChild(li);
      }
      parent.appendChild(list);
    }
  }
  return parent;
}

export function render(text: string, opts: RenderOptions = {}): DocumentFragment {
  return blocksDom(parse(text), opts, document.createDocumentFragment()) as DocumentFragment;
}
