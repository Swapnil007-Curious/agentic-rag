// Excerpts with the question's words marked (brass-soft), built from text nodes only.

const STOP = new Set(["what", "which", "when", "where", "does", "that", "this", "with", "from", "have", "under", "about", "there", "their", "them", "than", "then", "into", "will", "would", "should", "could", "between", "being", "were", "your", "whose", "how", "why", "the", "and", "for", "are", "can", "you"]);

export function queryTerms(question: string): string[] {
  const words = question.toLowerCase().match(/[a-z][a-z0-9'-]{3,}/g) ?? [];
  const out: string[] = [];
  for (const w of words) {
    // a light stem so "rebase" also marks "rebasing" and "rebased"
    const bare = w.replace(/'s$/, "");
    const stem = bare.replace(/(ing|es|s|e)$/, "");
    const base = stem.length >= 4 ? stem : bare;
    if (STOP.has(w) || base.length < 4 || out.includes(base)) continue;
    out.push(base);
  }
  return out.slice(0, 12);
}

const esc = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Text with <mark> around the terms (prefix match), as a fragment of text and mark nodes. */
export function markTerms(text: string, terms: readonly string[]): DocumentFragment {
  const frag = document.createDocumentFragment();
  if (!terms.length) {
    frag.appendChild(document.createTextNode(text));
    return frag;
  }
  const re = new RegExp(`\\b(${terms.map(esc).join("|")})[a-z]*`, "gi");
  let pos = 0;
  for (const m of text.matchAll(re)) {
    const i = m.index ?? 0;
    if (i > pos) frag.appendChild(document.createTextNode(text.slice(pos, i)));
    const mark = document.createElement("mark");
    mark.textContent = m[0];
    frag.appendChild(mark);
    pos = i + m[0].length;
  }
  if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
  return frag;
}

/** A window of about ``size`` characters around the first matched term. */
export function excerpt(text: string, terms: readonly string[], size = 300): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= size) return clean;
  let at = -1;
  const lower = clean.toLowerCase();
  for (const t of terms) {
    const i = lower.indexOf(t);
    if (i >= 0 && (at < 0 || i < at)) at = i;
  }
  let start = at < 0 ? 0 : Math.max(0, at - Math.floor(size / 3));
  if (start > 0) {
    const sp = clean.indexOf(" ", start);
    if (sp > 0 && sp - start < 40) start = sp + 1;
  }
  const end = Math.min(clean.length, start + size);
  return `${start > 0 ? "…" : ""}${clean.slice(start, end).trim()}${end < clean.length ? "…" : ""}`;
}
