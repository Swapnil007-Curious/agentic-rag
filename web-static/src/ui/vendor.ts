// KaTeX and highlight.js load on first use (an answer with maths or a code fence), from the
// self-hosted vendor/ files; markdown.ts reads them from window.katex / window.hljs.

const loading = new Map<string, Promise<void>>();

function script(src: string): Promise<void> {
  let p = loading.get(src);
  if (!p) {
    p = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.defer = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error(`${src} failed to load`));
      document.head.appendChild(s);
    });
    loading.set(src, p);
  }
  return p;
}

function stylesheet(href: string): void {
  if (document.querySelector(`link[href="${href}"]`)) return;
  const css = document.createElement("link");
  css.rel = "stylesheet";
  css.href = href;
  document.head.appendChild(css);
}

export function needsMath(text: string): boolean {
  return /\$[^\s$][^$\n]*\$|\\\(|\\\[|\$\$/.test(text);
}

export function needsCode(text: string): boolean {
  return /^\s{0,3}(`{3,}|~{3,})/m.test(text);
}

export function ensureKatex(base: string): Promise<void> {
  if (window.katex) return Promise.resolve();
  stylesheet(`${base}vendor/katex/katex.min.css`);
  return script(`${base}vendor/katex/katex.min.js`);
}

export function ensureHljs(base: string): Promise<void> {
  if (window.hljs) return Promise.resolve();
  return script(`${base}vendor/highlight/highlight.min.js`);
}
