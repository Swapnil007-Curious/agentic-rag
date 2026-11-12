// Deterministic guards (port of src/ragsys/agents/guards.py and the number extraction of
// src/ragsys/eval/checks.py): citation validity and number grounding by value, so "30 days" and
// "thirty days", or "50 lakh" and "50,00,000", match.

import { labelsIn, splitSentences, stripCitations } from "./text.ts";

export interface PassageLike {
  label: string;
  heading_path: string;
  section_ids: string[];
  text: string;
}

const UNITS: Record<string, number> = {
  zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9,
  ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16,
  seventeen: 17, eighteen: 18, nineteen: 19,
};
const TENS: Record<string, number> = {
  twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90,
};
const SCALES: Record<string, number> = {
  hundred: 100, thousand: 1_000, lakh: 100_000, lakhs: 100_000, lac: 100_000, lacs: 100_000,
  million: 1_000_000, crore: 10_000_000, crores: 10_000_000, billion: 1_000_000_000,
};
const TOKEN_RE = /\d[\d,]*(?:\.\d+)?|[a-z]+/g;

export interface NumberMention {
  value: number;
  base: number;
  text: string;
}

const has = (o: Record<string, number>, k: string): boolean => Object.hasOwn(o, k);

/** Numbers as digits or English words, each optionally followed by a multiplier word. */
export function extractNumbers(text: string): NumberMention[] {
  const s = text.toLowerCase().replace(/(?<=[a-z])-(?=[a-z])/g, " ");
  const toks = [...s.matchAll(TOKEN_RE)].map((m) => ({
    t: m[0],
    start: m.index ?? 0,
    end: (m.index ?? 0) + m[0].length,
  }));
  const out: NumberMention[] = [];
  let i = 0;
  while (i < toks.length) {
    const tok = toks[i];
    if (!tok) break;
    if (/\d/.test(tok.t[0] ?? "")) {
      const value0 = Number.parseFloat(tok.t.replace(/,+$/, "").replace(/,/g, ""));
      if (Number.isNaN(value0)) {
        i += 1;
        continue;
      }
      let value = value0;
      let j = i + 1;
      while (j < toks.length && has(SCALES, toks[j]?.t ?? "")) {
        value *= SCALES[toks[j]?.t ?? ""] ?? 1;
        j += 1;
      }
      out.push({ value, base: value0, text: s.slice(tok.start, toks[j - 1]?.end ?? tok.end) });
      i = j;
      continue;
    }
    if (has(UNITS, tok.t) || has(TENS, tok.t)) {
      let total = 0;
      let current = 0;
      let j = i;
      let wordBase: number | null = null;
      while (j < toks.length) {
        const t = toks[j]?.t ?? "";
        if (has(UNITS, t)) current += UNITS[t] ?? 0;
        else if (has(TENS, t)) current += TENS[t] ?? 0;
        else if (t === "hundred") current = (current || 1) * 100;
        else if (has(SCALES, t)) {
          if (wordBase === null) wordBase = total + (current || 1);
          total += (current || 1) * (SCALES[t] ?? 1);
          current = 0;
        } else if (t === "and" && j + 1 < toks.length && (has(UNITS, toks[j + 1]?.t ?? "") || has(TENS, toks[j + 1]?.t ?? ""))) {
          // "one hundred and five"
        } else break;
        j += 1;
      }
      const value = total + current;
      out.push({ value, base: wordBase ?? value, text: s.slice(tok.start, toks[j - 1]?.end ?? tok.end) });
      i = j;
      continue;
    }
    i += 1;
  }
  return out;
}

const key = (v: number): number => Math.round(v * 10_000) / 10_000;

export function numberForms(texts: readonly string[]): Set<number> {
  const out = new Set<number>();
  for (const t of texts) {
    for (const m of extractNumbers(t)) {
      out.add(key(m.value));
      out.add(key(m.base));
    }
  }
  return out;
}

export interface GuardResult {
  name: string;
  passed: boolean;
  reasons: string[];
}

export interface GuardReport {
  passed: boolean;
  status: "supported" | "unsupported";
  reasons: string[];
  results: GuardResult[];
}

export function citedLabels(answer: string, citations: readonly string[] = []): string[] {
  const out = labelsIn(answer);
  for (const c of citations) if (!out.includes(c)) out.push(c);
  return out;
}

const clip = (t: string, n = 80): string => (t.length <= n ? t : `${t.slice(0, n - 3)}...`);

export function citationValidity(answer: string, citations: readonly string[], known: readonly string[]): GuardResult {
  const reasons: string[] = [];
  const sentences = splitSentences(answer);
  if (!sentences.length) reasons.push("the answer is empty");
  sentences.forEach((s, i) => {
    if (!s.labels.length) reasons.push(`sentence ${i + 1} has no citation: "${clip(s.text)}"`);
  });
  const unknown = citedLabels(answer, citations).filter((l) => !known.includes(l));
  if (unknown.length) reasons.push(`cited label(s) not in the context: ${unknown.join(", ")}`);
  return { name: "citation_validity", passed: reasons.length === 0, reasons };
}

export function grounding(p: PassageLike): string {
  return `${p.heading_path}\n${p.section_ids.join(" ")}\n${p.text}`;
}

export function answerNumbers(answer: string): NumberMention[] {
  return extractNumbers(stripCitations(answer)).filter((m) => m.text.trim() !== "one");
}

export function numberGrounding(answer: string, cited: readonly PassageLike[]): GuardResult {
  const forms = numberForms(cited.map(grounding));
  const labels = cited.map((p) => p.label).join(", ") || "none";
  const reasons: string[] = [];
  const seen = new Set<number>();
  for (const m of answerNumbers(answer)) {
    const k = key(m.value);
    if (forms.has(k) || seen.has(k)) continue;
    seen.add(k);
    reasons.push(`number "${m.text}" does not occur in the cited passages (${labels})`);
  }
  return { name: "number_grounding", passed: reasons.length === 0, reasons };
}

export function subset<T extends { label: string }>(passages: readonly T[], labels: readonly string[]): T[] {
  const out: T[] = [];
  for (const l of labels) {
    const p = passages.find((x) => x.label === l);
    if (p && !out.includes(p)) out.push(p);
  }
  return out;
}

export function runGuards(answer: string, citations: readonly string[], passages: readonly PassageLike[]): GuardReport {
  const cited = subset(passages, citedLabels(answer, citations));
  const results = [
    citationValidity(answer, citations, passages.map((p) => p.label)),
    numberGrounding(answer, cited),
  ];
  const passed = results.every((r) => r.passed);
  return {
    passed,
    status: passed ? "supported" : "unsupported",
    reasons: results.flatMap((r) => r.reasons.map((m) => `${r.name}: ${m}`)),
    results,
  };
}
