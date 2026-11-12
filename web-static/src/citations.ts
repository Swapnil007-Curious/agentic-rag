// Citation labels in a chat reply (port of src/ragsys/assistant/citations.py): labels in the
// prose only (never in code), invalid labels removed, and the source line under the reply.

import { normalizeCitations } from "./text.ts";

const STRIP_RE = /[ \t]*\[(C\d{1,3})\]/g;
const LABEL_RE = /\[(C\d{1,3})\]/g;
const FENCE_RE = /^\s{0,3}(`{3,}|~{3,})/;
const INLINE_CODE_RE = /(`+)(?:.+?)\1/g;

export const GENERAL_LINE = "General knowledge — no matching documents";

/** Apply ``fn`` to the prose of a Markdown text; fenced code and inline code stay as they are. */
export function mapProse(text: string, fn: (segment: string) => string): string {
  const out: string[] = [];
  let fence: string | null = null;
  for (const line of text.split("\n")) {
    const m = FENCE_RE.exec(line);
    if (fence !== null) {
      out.push(line);
      const f = m?.[1];
      if (f && f[0] === fence[0] && f.length >= fence.length) fence = null;
      continue;
    }
    if (m?.[1]) {
      fence = m[1];
      out.push(line);
      continue;
    }
    let pos = 0;
    const parts: string[] = [];
    for (const cm of line.matchAll(INLINE_CODE_RE)) {
      const idx = cm.index ?? 0;
      parts.push(fn(line.slice(pos, idx)), cm[0]);
      pos = idx + cm[0].length;
    }
    parts.push(fn(line.slice(pos)));
    out.push(parts.join(""));
  }
  return out.join("\n");
}

export function findLabels(text: string): string[] {
  const seen: string[] = [];
  mapProse(text, (segment) => {
    for (const m of normalizeCitations(segment).matchAll(LABEL_RE)) {
      const label = m[1] ?? "";
      if (!seen.includes(label)) seen.push(label);
    }
    return segment;
  });
  return seen;
}

export interface CitationResult {
  text: string;
  used: string[];
  removed: string[];
}

export function sourceLine(nSources: number): string {
  if (nSources <= 0) return GENERAL_LINE;
  return `Answered with documents (${nSources} ${nSources === 1 ? "source" : "sources"})`;
}

/** Normalise citation variants in the prose and remove every label not in ``valid``. */
export function cleanCitations(text: string, valid: readonly string[]): CitationResult {
  const used: string[] = [];
  const removed: string[] = [];
  const cleaned = mapProse(text, (segment) =>
    normalizeCitations(segment).replace(STRIP_RE, (whole: string, label: string) => {
      if (valid.includes(label)) {
        if (!used.includes(label)) used.push(label);
        return whole;
      }
      if (!removed.includes(label)) removed.push(label);
      return "";
    }),
  );
  return { text: cleaned, used, removed };
}
