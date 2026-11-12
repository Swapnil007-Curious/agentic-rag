// Answer text handling, ported from src/ragsys/agents/text.py and context.py: citation tags
// ([C2], [C1][C3], [C1, C3]), sentence splitting that respects legal abbreviations, tag removal
// and the neutralising of passage-delimiter look-alikes in untrusted text.

const GROUP = String.raw`\[\s*[Cc]\s*\d+(?:\s*[,;]\s*(?:[Cc]\s*)?\d+)*\s*\]`;
export const CITATION_GROUP_RE = new RegExp(GROUP, "g");
const FULLWIDTH_RE = /【\s*([Cc]\s*\d+(?:\s*[,;]\s*(?:[Cc]\s*)?\d+)*)(?:\s*†[^】]*)?\s*】/g;
const CITE_LIST_RE = /\[\s*(C\d{1,3}(?:\s*[,;]\s*C\d{1,3})+)\s*\]/g;
const DELIM_RE = /<{3,}|>{3,}/g;

const ABBREVIATIONS = new Set([
  "rs", "no", "nos", "s", "ss", "r", "rr", "sec", "secs", "cl", "art", "arts", "sub", "vs", "v",
  "viz", "e.g", "i.e", "eg", "ie", "mr", "mrs", "ms", "dr", "govt", "ltd", "co", "inc", "para",
  "paras", "ch", "sch", "u/s", "approx", "fig", "p", "pp",
]);
const END_RE = new RegExp(
  String.raw`[.!?]+["'”’)\]]*(?<cites>(?:\s*${GROUP})*)(?=\s+["'“‘(A-Z0-9]|\s*$)`,
  "g",
);
const BULLET_RE = /^\s*(?:[-*•]|\d{1,2}[.)])\s+/;
const WORD_BEFORE_RE = /([A-Za-z][A-Za-z./]*)$/;

/** Neutralise runs of three or more '<' or '>' so a passage cannot close its own block. */
export function sanitize(text: string): string {
  return text.replace(DELIM_RE, (m) => (m.startsWith("<") ? "‹" : "›"));
}

/** 【C3】 / 【C3†L1-L4】 -> [C3]; text without fullwidth brackets is returned unchanged. */
export function normalizeCitationBrackets(text: string): string {
  if (!text.includes("【")) return text;
  return text.replace(FULLWIDTH_RE, (_m, inner: string) => `[${inner}]`);
}

/** Citation variants as single [Cn] tags: "【C3】", "[C3, C5]" -> "[C3]", "[C3][C5]". */
export function normalizeCitations(text: string): string {
  return normalizeCitationBrackets(text).replace(CITE_LIST_RE, (_m, inner: string) =>
    inner
      .split(/[,;]/)
      .map((c) => `[${c.trim()}]`)
      .join(""),
  );
}

/** Citation labels in order of first appearance, e.g. ["C2", "C1"]. */
export function labelsIn(text: string): string[] {
  const out: string[] = [];
  for (const group of text.match(CITATION_GROUP_RE) ?? []) {
    for (const num of group.match(/\d+/g) ?? []) {
      const label = `C${Number.parseInt(num, 10)}`;
      if (!out.includes(label)) out.push(label);
    }
  }
  return out;
}

export function stripCitations(text: string): string {
  return text
    .replace(CITATION_GROUP_RE, "")
    .replace(/\s+([.,;:!?])/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

/** 'C2', '[C2]', 'c2', 2 or '2' -> 'C2'; anything else -> null. */
export function normalizeLabel(value: unknown): string | null {
  if (typeof value === "number" && Number.isInteger(value)) return value > 0 ? `C${value}` : null;
  if (typeof value !== "string") return null;
  const m = /^\s*\[?\s*[Cc]?\s*(\d+)\s*\]?\s*$/.exec(value);
  if (!m || Number.parseInt(m[1] ?? "0", 10) <= 0) return null;
  return `C${Number.parseInt(m[1] ?? "0", 10)}`;
}

export interface Sentence {
  raw: string;
  text: string;
  labels: string[];
}

function isAbbreviation(before: string): boolean {
  const m = WORD_BEFORE_RE.exec(before);
  if (!m) return false;
  const word = (m[1] ?? "").toLowerCase().replace(/\.+$/, "");
  return ABBREVIATIONS.has(word) || (word.length === 1 && /[a-z]/.test(word));
}

function splitLine(line: string): string[] {
  const parts: string[] = [];
  let start = 0;
  END_RE.lastIndex = 0;
  for (const m of line.matchAll(END_RE)) {
    const idx = m.index ?? 0;
    if (isAbbreviation(line.slice(start, idx))) continue;
    const end = idx + m[0].length;
    const piece = line.slice(start, end).trim();
    if (piece) parts.push(piece);
    start = end;
  }
  const rest = line.slice(start).trim();
  if (rest) parts.push(rest);
  return parts;
}

/** Sentences of an answer; each keeps the citation tags that end it (see text.py). */
export function splitSentences(answer: string): Sentence[] {
  const lines = answer
    .replace(/\r/g, "")
    .split("\n")
    .map((ln) => ln.replace(BULLET_RE, "").trim())
    .filter((ln) => ln);
  const merged: string[] = [];
  for (const ln of lines) {
    const last = merged[merged.length - 1];
    if (last !== undefined && last.endsWith(":")) merged[merged.length - 1] = `${last} ${ln}`;
    else merged.push(ln);
  }
  const pieces: string[] = [];
  for (const ln of merged) {
    for (const piece of splitLine(ln)) {
      const last = pieces[pieces.length - 1];
      if (last !== undefined && !stripCitations(piece) && labelsIn(piece).length) {
        pieces[pieces.length - 1] = `${last} ${piece}`;
      } else {
        pieces.push(piece);
      }
    }
  }
  const out: Sentence[] = [];
  for (const p of pieces) {
    const text = stripCitations(p);
    const labels = labelsIn(p);
    if (text || labels.length) out.push({ raw: p, text, labels });
  }
  return out;
}

/** Rough token count (four characters per token), as the Python history cap uses. */
export function estimateTokens(text: string): number {
  return Math.floor(text.length / 4) + 1;
}
