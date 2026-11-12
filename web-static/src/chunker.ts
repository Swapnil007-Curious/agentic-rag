// Heading-aware chunking of an uploaded document's lines. Headings follow the generic parser's
// numbered-heading rule only (src/ragsys/structure/generic.py: "2 Results", "3.1 Method"); each
// heading's text becomes chunks of at most ``maxTokens`` (estimated at four characters per token),
// headed by the heading path, as the structural chunker's contextual header does.

import type { Chunk } from "./types.ts";

export const NUMBERED_HEADING_RE = /^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z].{0,120})$/;

export interface Line {
  text: string;
  page: number;
}

export function isHeading(text: string): { number: string; title: string; level: number } | null {
  const t = text.trim();
  if (t.length > 150 || /[.:;,]$/.test(t)) return null;
  const m = NUMBERED_HEADING_RE.exec(t);
  if (!m) return null;
  const number = m[1] ?? "";
  return { number, title: m[2] ?? "", level: number.split(".").length };
}

const tokens = (s: string): number => Math.ceil(s.length / 4);

interface Section {
  id: string;
  path: string[];
  lines: Line[];
}

function sections(lines: readonly Line[], title: string, docId: string): Section[] {
  const out: Section[] = [];
  const stack: { level: number; label: string }[] = [];
  let cur: Section = { id: `${docId}:preamble`, path: [title], lines: [] };
  for (const ln of lines) {
    const h = isHeading(ln.text);
    if (h) {
      if (cur.lines.length) out.push(cur);
      while (stack.length && (stack[stack.length - 1]?.level ?? 0) >= h.level) stack.pop();
      stack.push({ level: h.level, label: `${h.number} ${h.title}` });
      cur = { id: `${docId}:h${h.number}`, path: [title, ...stack.map((s) => s.label)], lines: [] };
      continue;
    }
    if (ln.text.trim()) cur.lines.push(ln);
  }
  if (cur.lines.length) out.push(cur);
  return out;
}

function pages(lines: readonly Line[]): string {
  const a = lines[0]?.page ?? 1;
  const b = lines[lines.length - 1]?.page ?? a;
  return a === b ? String(a) : `${a}-${b}`;
}

/** Chunks of an uploaded document; ``text`` is the display text, ``embedText`` adds the path. */
export function chunkLines(
  lines: readonly Line[],
  title: string,
  docId: string,
  maxTokens = 384,
  minTokens = 48,
): (Chunk & { embedText: string })[] {
  const out: (Chunk & { embedText: string })[] = [];
  for (const sec of sections(lines, title, docId)) {
    const path = sec.path.join(" > ");
    let piece: Line[] = [];
    let size = 0;
    const flush = (): void => {
      if (!piece.length) return;
      const text = piece.map((l) => l.text.trim()).join("\n");
      const prev = out[out.length - 1];
      // a tiny tail joins the previous piece of the same section
      if (tokens(text) < minTokens && prev && prev.section_ids[0] === sec.id && prev.n_tokens + tokens(text) <= maxTokens * 1.25) {
        prev.text = `${prev.text}\n${text}`;
        prev.embedText = `${path}\n${prev.text}`;
        prev.n_tokens = tokens(prev.embedText);
        const first = prev.pages.split("-")[0] ?? "1";
        const last = String(piece[piece.length - 1]?.page ?? first);
        prev.pages = first === last ? first : `${first}-${last}`;
      } else {
        out.push({
          chunk_id: `structural:${docId}:${String(out.length).padStart(5, "0")}`,
          doc_id: docId,
          heading_path: path,
          section_ids: [sec.id],
          pages: pages(piece),
          n_tokens: tokens(`${path}\n${text}`),
          text,
          embedText: `${path}\n${text}`,
        });
      }
      piece = [];
      size = 0;
    };
    for (const ln of sec.lines) {
      // very long lines (no line breaks in the PDF) are cut at word boundaries
      const parts = tokens(ln.text) > maxTokens ? splitLong(ln.text, maxTokens * 4) : [ln.text];
      for (const part of parts) {
        const n = tokens(part);
        if (size + n > maxTokens && piece.length) flush();
        piece.push({ text: part, page: ln.page });
        size += n;
      }
    }
    flush();
  }
  return out;
}

function splitLong(text: string, maxChars: number): string[] {
  const out: string[] = [];
  let rest = text.trim();
  while (rest.length > maxChars) {
    let cut = rest.lastIndexOf(" ", maxChars);
    if (cut < maxChars / 2) cut = maxChars;
    out.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trim();
  }
  if (rest) out.push(rest);
  return out;
}
