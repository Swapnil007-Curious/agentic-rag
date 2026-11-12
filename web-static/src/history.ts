// Conversation context for one request (port of src/ragsys/assistant/history.py): the last
// max_turns turns verbatim when they fit the budget; when they do not, the oldest messages are
// summarised by one short LLM call and only the newest messages that fit ``targetShare`` of the
// budget stay verbatim. The summary is stored with the conversation and reused while the
// messages it covers are unchanged (digest).

import { estimateTokens } from "./text.ts";

export interface Turn {
  role: "user" | "assistant";
  content: string;
}

export interface Summary {
  text: string;
  upto: number;
  digest: string;
}

export interface HistoryPlan {
  kept: Turn[];
  summary: Summary | null;
  newSummary: boolean;
  dropped: number;
  tokens: number;
  notes: string[];
}

export type Summarizer = (previous: string, turns: Turn[]) => Promise<string>;

/** A stable digest of messages (FNV-1a over role and content; only compared in this browser). */
export function messagesDigest(turns: readonly Turn[]): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  const text = JSON.stringify(turns.map((t) => [t.role, t.content]));
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ (c + 0x9e37), 0x5bd1e995) >>> 0;
  }
  return h1.toString(16).padStart(8, "0") + h2.toString(16).padStart(8, "0");
}

function startWithUser(turns: Turn[]): Turn[] {
  let i = 0;
  while (i < turns.length && turns[i]?.role !== "user") i++;
  return turns.slice(i);
}

const sum = (turns: readonly Turn[]): number => turns.reduce((s, t) => s + estimateTokens(t.content), 0);

export async function planHistory(
  turns: readonly Turn[],
  budgetTokens: number,
  opts: { maxTurns?: number; summary?: Summary | null; summarize?: Summarizer; targetShare?: number } = {},
): Promise<HistoryPlan> {
  const maxTurns = opts.maxTurns ?? 10;
  const targetShare = opts.targetShare ?? 0.5;
  const msgs = turns.filter((t) => t.content.trim());
  const latest = msgs[msgs.length - 1];
  if (!latest || latest.role !== "user") throw new Error("the conversation must end with a user message");
  let start = 0;
  let prior: Summary | null = null;
  const s = opts.summary;
  if (s && s.upto > 0 && s.upto < msgs.length && s.digest === messagesDigest(msgs.slice(0, s.upto))) {
    start = s.upto;
    prior = s;
  }
  const earlier = msgs.slice(start, -1);
  const window = maxTurns > 1 ? earlier.slice(-2 * (maxTurns - 1)) : [];
  const latestTokens = estimateTokens(latest.content);
  const priorTokens = prior ? estimateTokens(prior.text) : 0;
  const avail = budgetTokens - latestTokens - priorTokens;
  if (sum(window) <= avail) {
    const fits = startWithUser([...window]);
    const k = msgs.length - 1 - fits.length;
    return { kept: [...fits, latest], summary: prior, newSummary: false, dropped: k - start, tokens: sum(fits) + latestTokens + priorTokens, notes: [] };
  }
  const target = Math.max(0, Math.floor(budgetTokens * targetShare) - latestTokens);
  let kept: Turn[] = [];
  let used = 0;
  for (let i = window.length - 1; i >= 0; i--) {
    const t = window[i] as Turn;
    const n = estimateTokens(t.content);
    if (used + n > target) break;
    kept.unshift(t);
    used += n;
  }
  kept = startWithUser(kept);
  const k = msgs.length - 1 - kept.length;
  const toSummarise = msgs.slice(start, k);
  let planSummary = prior;
  let calls = 0;
  const notes: string[] = [];
  if (toSummarise.length && opts.summarize) {
    calls = 1;
    let text = "";
    try {
      text = await opts.summarize(prior?.text ?? "", toSummarise);
    } catch {
      text = "";
    }
    if (text) planSummary = { text, upto: k, digest: messagesDigest(msgs.slice(0, k)) };
    else notes.push("the summary of earlier messages could not be written");
  } else if (toSummarise.length) {
    notes.push(`${toSummarise.length} earlier message(s) were left out`);
  }
  const dropped = planSummary && planSummary.upto === k ? 0 : k - start;
  return {
    kept: [...kept, latest],
    summary: planSummary,
    newSummary: calls > 0 && planSummary !== prior,
    dropped,
    tokens: sum(kept) + latestTokens + (planSummary ? estimateTokens(planSummary.text) : 0),
    notes,
  };
}

/** Plain-text transcript for the summariser (each message clipped). */
export function transcript(turns: readonly Turn[], maxChars: number, perMessage = 2000): string {
  const out = turns
    .map((t) => {
      let text = t.content.trim();
      if (text.length > perMessage) text = `${text.slice(0, perMessage)} [...]`;
      return `${t.role === "user" ? "User" : "Assistant"}: ${text}`;
    })
    .join("\n\n");
  return out.slice(-maxChars);
}
