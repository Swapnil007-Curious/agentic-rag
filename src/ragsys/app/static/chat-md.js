// Safe Markdown for the chat assistant: model text -> a plain node tree -> DOM.
//
// Model output is untrusted and is never parsed as HTML: the parser produces a tree of plain
// objects and the renderer builds elements with createElement / textContent, so raw HTML in a
// reply (<script>, <img onerror=...>) is shown as text. Links keep only http(s) and mailto
// targets (anything else, e.g. javascript:, stays plain text) and open with rel="noopener
// noreferrer nofollow". Math ($...$, $$...$$, \(...\), \[...\]) is rendered by KaTeX with
// trust off (KaTeX builds DOM nodes and sets styles through the CSSOM, which the page's
// Content-Security-Policy allows); fenced code is highlighted by highlight.js on the final
// render. toHtml() serialises a tree with everything escaped (used by the tests).
"use strict";

(function (root) {
  const MAX_DEPTH = 6;
  const SAFE_URL = /^(https?:\/\/|mailto:)[^\s<>"'`]+$/i;
  const CITE_RE = /^C\d{1,3}$/;

  // ------------------------------------------------------------------ inline parsing

  const INLINE_RE = new RegExp(
    [
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
    ].join("|"),
    "g"
  );

  function parseInline(text, depth) {
    const out = [];
    const d = depth || 0;
    let pos = 0;
    const pushText = (t) => {
      if (!t) return;
      const last = out[out.length - 1];
      if (last && last.t === "text") last.v += t;
      else out.push({ t: "text", v: t });
    };
    INLINE_RE.lastIndex = 0;
    let m;
    const re = new RegExp(INLINE_RE.source, "g");
    while ((m = re.exec(text)) !== null) {
      if (m[0] === "") {
        re.lastIndex++;
        continue;
      }
      pushText(text.slice(pos, m.index));
      pos = m.index + m[0].length;
      if (m[1] !== undefined) out.push({ t: "code", v: m[2].replace(/^ (.*) $/s, "$1") });
      else if (m[3] !== undefined) out.push({ t: "math", v: m[3], display: false });
      else if (m[4] !== undefined) out.push({ t: "math", v: m[4], display: false });
      else if (m[5] !== undefined) out.push({ t: "cite", v: m[5] });
      else if (m[6] !== undefined) {
        const url = m[7];
        if (SAFE_URL.test(url)) {
          out.push({ t: "link", href: url, c: d < MAX_DEPTH ? parseInline(m[6], d + 1) : [{ t: "text", v: m[6] }] });
        } else {
          pushText(m[6] + " (" + url + ")");
        }
      } else if (m[8] !== undefined) out.push({ t: "link", href: m[8], c: [{ t: "text", v: m[8] }] });
      else if (m[9] !== undefined) out.push({ t: "strong", c: d < MAX_DEPTH ? parseInline(m[10], d + 1) : [{ t: "text", v: m[10] }] });
      else if (m[11] !== undefined) out.push({ t: "del", c: d < MAX_DEPTH ? parseInline(m[11], d + 1) : [{ t: "text", v: m[11] }] });
      else if (m[12] !== undefined) out.push({ t: "em", c: d < MAX_DEPTH ? parseInline(m[13], d + 1) : [{ t: "text", v: m[13] }] });
      else if (m[14] !== undefined) out.push({ t: "link", href: m[14], c: [{ t: "text", v: m[14] }] });
    }
    pushText(text.slice(pos));
    return out;
  }

  // ------------------------------------------------------------------ block parsing

  const FENCE_RE = /^ {0,3}(`{3,}|~{3,})\s*([^\s`]*)/;
  const HEADING_RE = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
  const RULE_RE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
  const ITEM_RE = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
  const QUOTE_RE = /^ {0,3}> ?(.*)$/;
  const TABLE_SEP_RE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
  const MATH_OPEN_RE = /^\s*(\$\$|\\\[)(.*)$/;

  function langTag(raw) {
    return String(raw || "").replace(/[^A-Za-z0-9_+#.-]/g, "").slice(0, 20).toLowerCase();
  }

  function cells(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
    return s.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
  }

  function aligns(line) {
    return cells(line).map((c) => (c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : c.startsWith(":") ? "left" : ""));
  }

  function parseBlocks(lines, depth) {
    const blocks = [];
    const para = [];
    const d = depth || 0;
    let i = 0;
    const flush = () => {
      if (para.length) {
        blocks.push({ t: "p", c: parseInline(para.join(" ").trim(), 0) });
        para.length = 0;
      }
    };
    while (i < lines.length) {
      const ln = lines[i];
      const fm = FENCE_RE.exec(ln);
      if (fm) {
        flush();
        const fence = fm[1];
        const body = [];
        i++;
        let closed = false;
        while (i < lines.length) {
          const close = /^ {0,3}(`{3,}|~{3,})\s*$/.exec(lines[i]);
          if (close && close[1][0] === fence[0] && close[1].length >= fence.length) {
            closed = true;
            i++;
            break;
          }
          body.push(lines[i]);
          i++;
        }
        blocks.push({ t: "code", lang: langTag(fm[2]), v: body.join("\n"), closed: closed });
        continue;
      }
      const mm = MATH_OPEN_RE.exec(ln);
      if (mm) {
        const close = mm[1] === "$$" ? "$$" : "\\]";
        let rest = mm[2];
        const end = rest.indexOf(close);
        if (end >= 0 && rest.slice(end + close.length).trim() === "") {
          flush();
          blocks.push({ t: "math", v: rest.slice(0, end).trim(), display: true });
          i++;
          continue;
        }
        if (end < 0) {
          // multi-line display math until the closing delimiter
          const body = [rest];
          let j = i + 1;
          let found = false;
          while (j < lines.length) {
            const k = lines[j].indexOf(close);
            if (k >= 0) {
              body.push(lines[j].slice(0, k));
              found = lines[j].slice(k + close.length).trim() === "";
              break;
            }
            body.push(lines[j]);
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
        blocks.push({ t: "h", level: hm[1].length, c: parseInline(hm[2], 0) });
        i++;
        continue;
      }
      if (RULE_RE.test(ln)) {
        flush();
        blocks.push({ t: "hr" });
        i++;
        continue;
      }
      if (QUOTE_RE.test(ln) && d < MAX_DEPTH) {
        flush();
        const inner = [];
        while (i < lines.length && QUOTE_RE.test(lines[i])) {
          inner.push(QUOTE_RE.exec(lines[i])[1]);
          i++;
        }
        blocks.push({ t: "quote", c: parseBlocks(inner, d + 1) });
        continue;
      }
      if (
        ln.includes("|") &&
        i + 1 < lines.length &&
        TABLE_SEP_RE.test(lines[i + 1]) &&
        cells(ln).length === cells(lines[i + 1]).length
      ) {
        flush();
        const head = cells(ln).map((c) => parseInline(c, 0));
        const al = aligns(lines[i + 1]);
        const rows = [];
        i += 2;
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
          rows.push(cells(lines[i]).map((c) => parseInline(c, 0)));
          i++;
        }
        blocks.push({ t: "table", head: head, rows: rows, align: al });
        continue;
      }
      const im = ITEM_RE.exec(ln);
      if (im && d < MAX_DEPTH) {
        flush();
        const baseIndent = im[1].length;
        const ordered = /\d/.test(im[2][0]);
        const start = ordered ? parseInt(im[2], 10) : 1;
        const items = [];
        let cur = null;
        while (i < lines.length) {
          const l2 = lines[i];
          const m2 = ITEM_RE.exec(l2);
          if (m2 && m2[1].length <= baseIndent + 1 && /\d/.test(m2[2][0]) === ordered) {
            cur = { lines: [m2[3]] };
            items.push(cur);
            i++;
            continue;
          }
          if (m2 && m2[1].length <= baseIndent + 1) break; // a list of the other kind
          if (!l2.trim()) {
            // blank line: the list continues if the next line is indented or another item
            const next = lines[i + 1];
            if (next !== undefined && cur && (/^\s{2,}\S/.test(next) || (ITEM_RE.exec(next) && ITEM_RE.exec(next)[1].length <= baseIndent + 1))) {
              cur.lines.push("");
              i++;
              continue;
            }
            break;
          }
          if (cur && (/^\s+\S/.test(l2) || !FENCE_RE.test(l2))) {
            if (!/^\s/.test(l2) && (HEADING_RE.test(l2) || QUOTE_RE.test(l2) || RULE_RE.test(l2))) break;
            cur.lines.push(l2.replace(new RegExp("^\\s{0," + (baseIndent + 4) + "}"), ""));
            i++;
            continue;
          }
          break;
        }
        blocks.push({
          t: "list",
          ordered: ordered,
          start: start,
          items: items.map((it) => parseBlocks(it.lines, d + 1)),
        });
        continue;
      }
      para.push(ln.trim());
      i++;
    }
    flush();
    return blocks;
  }

  function parse(text) {
    const src = String(text || "").slice(0, 100000).replace(/\r\n?/g, "\n");
    return parseBlocks(src.split("\n"), 0);
  }

  // ------------------------------------------------------------------ HTML (escaped, tests)

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  function inlineHtml(nodes) {
    return (nodes || [])
      .map((n) => {
        switch (n.t) {
          case "text":
            return esc(n.v);
          case "code":
            return "<code>" + esc(n.v) + "</code>";
          case "math":
            return '<span class="math">' + esc(n.v) + "</span>";
          case "cite":
            return '<button class="cite">' + esc(n.v) + "</button>";
          case "link":
            return '<a href="' + esc(n.href) + '" rel="noopener noreferrer nofollow">' + inlineHtml(n.c) + "</a>";
          case "strong":
            return "<strong>" + inlineHtml(n.c) + "</strong>";
          case "em":
            return "<em>" + inlineHtml(n.c) + "</em>";
          case "del":
            return "<del>" + inlineHtml(n.c) + "</del>";
          default:
            return "";
        }
      })
      .join("");
  }

  function toHtml(blocks) {
    return (blocks || [])
      .map((b) => {
        switch (b.t) {
          case "p":
            return "<p>" + inlineHtml(b.c) + "</p>";
          case "h":
            return "<h" + b.level + ">" + inlineHtml(b.c) + "</h" + b.level + ">";
          case "hr":
            return "<hr>";
          case "code":
            return '<pre><code class="language-' + esc(b.lang) + '">' + esc(b.v) + "</code></pre>";
          case "math":
            return '<div class="math">' + esc(b.v) + "</div>";
          case "quote":
            return "<blockquote>" + toHtml(b.c) + "</blockquote>";
          case "list": {
            const tag = b.ordered ? "ol" : "ul";
            return "<" + tag + ">" + b.items.map((it) => "<li>" + toHtml(it) + "</li>").join("") + "</" + tag + ">";
          }
          case "table":
            return (
              "<table><thead><tr>" +
              b.head.map((c) => "<th>" + inlineHtml(c) + "</th>").join("") +
              "</tr></thead><tbody>" +
              b.rows.map((r) => "<tr>" + r.map((c) => "<td>" + inlineHtml(c) + "</td>").join("") + "</tr>").join("") +
              "</tbody></table>"
            );
          default:
            return "";
        }
      })
      .join("");
  }

  // ------------------------------------------------------------------ DOM

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function copyText(text, button, label) {
    const reset = label || "Copy";
    const done = (ok) => {
      button.textContent = ok ? "Copied" : "Copy failed";
      setTimeout(() => {
        button.textContent = reset;
      }, 1500);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => done(true), () => done(false));
      return;
    }
    const area = el("textarea", "visually-hidden");
    area.value = text;
    area.setAttribute("readonly", "");
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (err) {
      ok = false;
    }
    document.body.removeChild(area);
    done(ok);
  }

  const KATEX_OPTIONS = { throwOnError: false, trust: false, strict: "ignore", maxSize: 20, maxExpand: 500, output: "htmlAndMathml" };

  function mathNode(tex, display) {
    const node = el(display ? "div" : "span", display ? "math-block" : "math-inline");
    const k = root.katex;
    if (k && typeof k.render === "function" && tex.length <= 5000) {
      try {
        k.render(tex, node, Object.assign({ displayMode: !!display }, KATEX_OPTIONS));
        if (display) node.tabIndex = 0;
        return node;
      } catch (err) {
        node.textContent = "";
      }
    }
    node.textContent = display ? tex : "$" + tex + "$";
    node.classList.add("math-raw");
    return node;
  }

  function inlineDom(nodes, opts, parent) {
    for (const n of nodes || []) {
      if (n.t === "text") parent.appendChild(document.createTextNode(n.v));
      else if (n.t === "code") parent.appendChild(el("code", "md-code", n.v));
      else if (n.t === "math") parent.appendChild(mathNode(n.v, false));
      else if (n.t === "strong") inlineDom(n.c, opts, parent.appendChild(el("strong")));
      else if (n.t === "em") inlineDom(n.c, opts, parent.appendChild(el("em")));
      else if (n.t === "del") inlineDom(n.c, opts, parent.appendChild(el("del")));
      else if (n.t === "cite") {
        const made = opts && opts.cite && CITE_RE.test(n.v) ? opts.cite(n.v) : null;
        parent.appendChild(made || document.createTextNode("[" + n.v + "]"));
      } else if (n.t === "link") {
        if (SAFE_URL.test(n.href)) {
          const a = el("a");
          a.href = n.href;
          a.rel = "noopener noreferrer nofollow";
          a.target = "_blank";
          inlineDom(n.c, opts, a);
          parent.appendChild(a);
        } else {
          inlineDom(n.c, opts, parent);
        }
      }
    }
    return parent;
  }

  function codeDom(b, opts) {
    const wrap = el("div", "md-pre");
    const bar = el("div", "md-pre-bar");
    bar.appendChild(el("span", "md-lang", b.lang || "text"));
    const btn = el("button", "md-copy", "Copy");
    btn.type = "button";
    btn.setAttribute("aria-label", "Copy code" + (b.lang ? " (" + b.lang + ")" : ""));
    btn.addEventListener("click", () => copyText(b.v, btn));
    bar.appendChild(btn);
    wrap.appendChild(bar);
    const pre = el("pre");
    pre.tabIndex = 0;
    const code = el("code", b.lang ? "language-" + b.lang : null, b.v);
    pre.appendChild(code);
    wrap.appendChild(pre);
    const hl = root.hljs;
    if (opts && opts.highlight && hl && b.lang && hl.getLanguage && hl.getLanguage(b.lang) && b.v.length < 60000) {
      try {
        hl.highlightElement(code);
      } catch (err) {
        code.textContent = b.v;
      }
    }
    return wrap;
  }

  function tableDom(b, opts) {
    const scroller = el("div", "md-table-wrap");
    scroller.tabIndex = 0;
    const t = el("table", "md-table");
    const thead = el("thead");
    const hr = el("tr");
    b.head.forEach((c, i) => {
      const th = inlineDom(c, opts, el("th"));
      th.scope = "col";
      if (b.align && b.align[i]) th.className = "align-" + b.align[i];
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    t.appendChild(thead);
    const tbody = el("tbody");
    for (const r of b.rows) {
      const tr = el("tr");
      r.forEach((c, i) => {
        const td = inlineDom(c, opts, el("td"));
        if (b.align && b.align[i]) td.className = "align-" + b.align[i];
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
    t.appendChild(tbody);
    scroller.appendChild(t);
    return scroller;
  }

  // opts.cite(label) -> element or null; opts.highlight: run highlight.js (final render)
  function blocksDom(blocks, opts, parent) {
    for (const b of blocks || []) {
      if (b.t === "p") parent.appendChild(inlineDom(b.c, opts, el("p")));
      else if (b.t === "h") parent.appendChild(inlineDom(b.c, opts, el("h" + Math.min(6, b.level + 2), "md-h")));
      else if (b.t === "hr") parent.appendChild(el("hr"));
      else if (b.t === "code") parent.appendChild(codeDom(b, opts));
      else if (b.t === "math") parent.appendChild(mathNode(b.v, true));
      else if (b.t === "quote") parent.appendChild(blocksDom(b.c, opts, el("blockquote")));
      else if (b.t === "table") parent.appendChild(tableDom(b, opts));
      else if (b.t === "list") {
        const list = el(b.ordered ? "ol" : "ul", "md-list");
        if (b.ordered && b.start && b.start !== 1) list.start = b.start;
        for (const item of b.items) {
          const li = el("li");
          if (item.length === 1 && item[0].t === "p") inlineDom(item[0].c, opts, li);
          else blocksDom(item, opts, li);
          list.appendChild(li);
        }
        parent.appendChild(list);
      }
    }
    return parent;
  }

  function render(text, opts) {
    return blocksDom(parse(text), opts || {}, document.createDocumentFragment());
  }

  const api = { parse: parse, parseInline: parseInline, toHtml: toHtml, render: render, copyText: copyText, el: el, SAFE_URL: SAFE_URL };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ChatMarkdown = api;
})(typeof window !== "undefined" ? window : globalThis);
