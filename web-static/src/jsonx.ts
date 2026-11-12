// Tolerant JSON reading of model replies (ports of extract_json, parse_synth and parse_critic).

import { labelsIn, normalizeCitationBrackets, normalizeLabel, stripCitations } from "./text.ts";

export function extractJson(text: string): Record<string, unknown> {
  const candidates = [text.trim()];
  const fence = /```(?:json)?\s*([\s\S]*?)```/i.exec(text);
  if (fence?.[1]) candidates.push(fence[1].trim());
  let start = text.indexOf("{");
  while (start !== -1) {
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === "\\") esc = true;
        else if (ch === '"') inStr = false;
      } else if (ch === '"') inStr = true;
      else if (ch === "{") depth++;
      else if (ch === "}") {
        depth--;
        if (depth === 0) {
          candidates.push(text.slice(start, i + 1));
          break;
        }
      }
    }
    start = text.indexOf("{", start + 1);
  }
  for (const c of candidates) {
    try {
      const data: unknown = JSON.parse(c);
      if (data && typeof data === "object" && !Array.isArray(data)) return data as Record<string, unknown>;
    } catch {
      /* next candidate */
    }
  }
  throw new Error(`no JSON object in the reply: ${JSON.stringify(text.slice(0, 200))}`);
}

export interface SynthOutput {
  abstain: boolean;
  answer: string;
  citations: string[];
  note: string | null;
}

const REFUSAL_RE =
  /\b(?:passages?|context|documents?|provided text)\b.{0,40}\b(?:do(?:es)? not|don't|doesn't)\s+(?:contain|mention|specify|provide|say|state|include|cover)|\b(?:no|not enough) information\b|\bcannot (?:be )?(?:answer|determine)/i;

function asBool(v: unknown): boolean {
  if (typeof v === "boolean") return v;
  if (typeof v === "string") return ["true", "yes", "1"].includes(v.trim().toLowerCase());
  return Boolean(v);
}

export function parseSynth(data: Record<string, unknown>): SynthOutput {
  if (!("abstain" in data) && !("answer" in data)) throw new Error("reply needs 'abstain' and 'answer'");
  let abstain = asBool(data.abstain ?? false);
  let answer = data.answer ?? "";
  if (answer === null) answer = "";
  if (typeof answer !== "string") throw new Error("'answer' must be a string");
  let raw = data.citations ?? [];
  if (raw === null) raw = [];
  if (typeof raw === "string") raw = raw.trim() ? raw.trim().split(/[,;\s]+/) : [];
  if (!Array.isArray(raw)) throw new Error("'citations' must be a list");
  let cites: string[] = [];
  for (const c of raw) {
    const lab = normalizeLabel(c);
    if (lab === null) throw new Error(`bad citation label ${JSON.stringify(c)}`);
    if (!cites.includes(lab)) cites.push(lab);
  }
  let text = normalizeCitationBrackets(answer.trim());
  let note: string | null = null;
  if (abstain) {
    text = "";
    cites = [];
  } else if (!text) {
    abstain = true;
    note = "empty answer";
  } else if (!labelsIn(text).length && REFUSAL_RE.test(stripCitations(text))) {
    abstain = true;
    note = "answer text is a refusal";
    text = "";
    cites = [];
  }
  return { abstain, answer: text, citations: cites, note };
}

export interface CriticOutput {
  verdict: "accept" | "revise" | "abstain";
  unsupported_sentences: string[];
  feedback: string;
  error: string | null;
}

const VERDICTS: Record<string, CriticOutput["verdict"]> = {
  accept: "accept", accepted: "accept", approve: "accept", supported: "accept",
  revise: "revise", revision: "revise", reject: "revise", rejected: "revise", abstain: "abstain",
};

export function parseCritic(data: Record<string, unknown>): CriticOutput {
  const raw = data.verdict;
  const verdict = typeof raw === "string" ? VERDICTS[raw.trim().toLowerCase()] : undefined;
  if (!verdict) throw new Error("'verdict' must be accept, revise or abstain");
  let unsupported = data.unsupported_sentences ?? [];
  if (unsupported === null) unsupported = [];
  if (typeof unsupported === "string") unsupported = unsupported.trim() ? [unsupported] : [];
  if (!Array.isArray(unsupported)) throw new Error("'unsupported_sentences' must be a list");
  const feedback = data.feedback == null ? "" : String(data.feedback);
  return {
    verdict,
    unsupported_sentences: unsupported.map((s) => String(s).trim()).filter(Boolean),
    feedback: feedback.trim(),
    error: null,
  };
}

/** The critic's verdict; an unreadable reply counts as no objection (the guards passed already). */
export function readCritic(text: string): CriticOutput {
  try {
    return parseCritic(extractJson(text));
  } catch (err) {
    return { verdict: "accept", unsupported_sentences: [], feedback: "", error: err instanceof Error ? err.message : String(err) };
  }
}
