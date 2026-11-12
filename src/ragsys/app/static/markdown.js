// Safe Markdown rendering for model answers and summary pages. The server parses Markdown into a
// block tree (ragsys/app/markdown.py); this file builds it with createElement and textContent
// only. No HTML strings, no attributes from the text, no links (link targets arrive as text).
"use strict";

(function (global) {
  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function copyText(text, button) {
    const done = (ok) => {
      button.textContent = ok ? "Copied" : "Copy failed";
      setTimeout(() => {
        button.textContent = "Copy";
      }, 1600);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => done(true), () => done(false));
      return;
    }
    const area = el("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.className = "visually-hidden";
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

  function inlines(tokens, opts, parent) {
    for (const t of tokens || []) {
      if (t.t === "text") parent.appendChild(document.createTextNode(t.v));
      else if (t.t === "code") parent.appendChild(el("code", "md-code", t.v));
      else if (t.t === "strong") inlines(t.c, opts, parent.appendChild(el("strong")));
      else if (t.t === "em") inlines(t.c, opts, parent.appendChild(el("em")));
      else if (t.t === "cite") {
        const made = opts && opts.cite ? opts.cite(t.v) : null;
        parent.appendChild(made || document.createTextNode("[" + t.v + "]"));
      }
    }
    return parent;
  }

  function list(block, opts) {
    const node = el(block.ordered ? "ol" : "ul", "md-list");
    for (const item of block.items || []) {
      const li = inlines(item.inlines, opts, el("li"));
      if (item.sub) li.appendChild(list(item.sub, opts));
      node.appendChild(li);
    }
    return node;
  }

  function codeBlock(block) {
    const wrap = el("div", "md-pre");
    const bar = el("div", "md-pre-bar");
    bar.appendChild(el("span", "md-lang", block.lang || "text"));
    const btn = el("button", "btn btn-quiet md-copy", "Copy");
    btn.type = "button";
    btn.setAttribute("aria-label", "Copy code" + (block.lang ? " (" + block.lang + ")" : ""));
    btn.addEventListener("click", () => copyText(block.text, btn));
    bar.appendChild(btn);
    wrap.appendChild(bar);
    const pre = el("pre");
    pre.tabIndex = 0;
    pre.appendChild(el("code", null, block.text));
    wrap.appendChild(pre);
    return wrap;
  }

  function table(block, opts) {
    const scroller = el("div", "md-table-wrap");
    scroller.tabIndex = 0;
    const t = el("table", "md-table");
    const head = el("thead");
    const hr = el("tr");
    for (const c of block.header || []) inlines(c, opts, hr.appendChild(el("th")));
    head.appendChild(hr);
    t.appendChild(head);
    const body = el("tbody");
    for (const r of block.rows || []) {
      const tr = el("tr");
      for (const c of r) inlines(c, opts, tr.appendChild(el("td")));
      body.appendChild(tr);
    }
    t.appendChild(body);
    scroller.appendChild(t);
    return scroller;
  }

  // blocks: the server's block tree; opts.cite(label) -> element for [C1] citations (optional);
  // opts.headingBase: lowest heading level used on the page (default 3)
  function render(blocks, opts) {
    const frag = document.createDocumentFragment();
    const base = (opts && opts.headingBase) || 3;
    for (const b of blocks || []) {
      if (b.type === "heading") {
        const level = Math.min(6, Math.max(base, base + (b.level || 1) - 1));
        frag.appendChild(inlines(b.inlines, opts, el("h" + level, "md-h")));
      } else if (b.type === "paragraph") frag.appendChild(inlines(b.inlines, opts, el("p")));
      else if (b.type === "list") frag.appendChild(list(b, opts));
      else if (b.type === "code") frag.appendChild(codeBlock(b));
      else if (b.type === "table") frag.appendChild(table(b, opts));
      else if (b.type === "rule") frag.appendChild(el("hr"));
    }
    return frag;
  }

  global.SafeMarkdown = { render: render, el: el };
})(window);
