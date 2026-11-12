// On-demand verification of a grounded reply (port of src/ragsys/assistant/verify.py): the
// citation and number guards run here on the prose (code is not a claim) and per sentence with
// the passages it cites; the critic checks the whole reply through the proxy. A sentence is
// supported when it cites a given passage, its numbers occur in those passages and the critic did
// not flag it.

import { readCritic, type CriticOutput } from "./jsonx.ts";
import { citedLabels, numberGrounding, runGuards, subset, type GuardReport, type PassageLike } from "./guards.ts";
import { labelsIn, splitSentences, stripCitations } from "./text.ts";

const FENCE_BLOCK_RE = /^\s{0,3}(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:^\s{0,3}\1[^\n]*$|(?![\s\S]))/gm;
const MATH_BLOCK_RE = /\$\$[\s\S]*?\$\$/g;
const HEADING_RE = /^\s{0,3}#{1,6}\s+.*$/gm;
const EMPHASIS_RE = /(\*\*|__)(.+?)\1/g;

export function proseOf(answer: string): string {
  return answer.replace(FENCE_BLOCK_RE, "\n").replace(MATH_BLOCK_RE, " ").replace(HEADING_RE, "").replace(EMPHASIS_RE, "$2");
}

const norm = (t: string): string => stripCitations(t).replace(/\s+/g, " ").trim().toLowerCase().replace(/\.$/, "");

function similarity(a: string, b: string): number {
  // Dice coefficient on character bigrams (difflib's ratio is not available in the browser)
  if (a === b) return 1;
  const grams = (s: string): Map<string, number> => {
    const m = new Map<string, number>();
    for (let i = 0; i < s.length - 1; i++) m.set(s.slice(i, i + 2), (m.get(s.slice(i, i + 2)) ?? 0) + 1);
    return m;
  };
  const ga = grams(a);
  const gb = grams(b);
  let inter = 0;
  for (const [g, n] of ga) inter += Math.min(n, gb.get(g) ?? 0);
  const total = Math.max(1, a.length - 1 + (b.length - 1));
  return (2 * inter) / total;
}

export function flagged(sentence: string, flaggedList: readonly string[]): boolean {
  const s = norm(sentence);
  return flaggedList.some((f) => {
    const g = norm(f);
    return g !== "" && (s.includes(g) || g.includes(s) || similarity(s, g) >= 0.8);
  });
}

export interface VerifiedSentence {
  text: string;
  citations: string[];
  status: "supported" | "unsupported";
  reasons: string[];
}

export interface VerifyResult {
  sentences: VerifiedSentence[];
  supported: number;
  total: number;
  guards: GuardReport;
  critic: CriticOutput;
}

export async function verifyAnswer(
  answer: string,
  passages: readonly PassageLike[],
  critic: (cited: PassageLike[], prose: string) => Promise<string>,
): Promise<VerifyResult> {
  const prose = proseOf(answer);
  const report = runGuards(prose, labelsIn(prose), passages);
  const cited = subset(passages, citedLabels(prose));
  const verdict = readCritic(await critic(cited, prose));
  const known = passages.map((p) => p.label);
  const rows: VerifiedSentence[] = [];
  for (const s of splitSentences(prose)) {
    if (!s.text) continue;
    const reasons: string[] = [];
    const valid = s.labels.filter((l) => known.includes(l));
    if (!s.labels.length) reasons.push("no citation");
    else if (valid.length < s.labels.length) reasons.push("cites a passage that was not given");
    if (valid.length) reasons.push(...numberGrounding(s.raw, subset(passages, valid)).reasons);
    if (flagged(s.text, verdict.unsupported_sentences)) reasons.push("the critic found it unsupported by the cited passages");
    rows.push({ text: s.text, citations: s.labels, status: reasons.length ? "unsupported" : "supported", reasons });
  }
  if (verdict.verdict === "abstain") {
    for (const r of rows) {
      if (r.status === "supported") {
        r.status = "unsupported";
        r.reasons.push("the critic found that the passages do not answer the question");
      }
    }
  }
  const supported = rows.filter((r) => r.status === "supported").length;
  return { sentences: rows, supported, total: rows.length, guards: report, critic: verdict };
}
