// Automatic grounding without an LLM call (port of src/ragsys/assistant/gate.py) and context
// packing (port of src/ragsys/agents/context.py): the latest message, joined with the previous
// user message for a short follow-up, is embedded once; every source whose top-1 cosine reaches
// its threshold contributes hits; the pooled hits, best first, become labelled passages within
// the token budget.

import { sanitize } from "./text.ts";
import type { Hit, Passage } from "./types.ts";
import type { SearchSource } from "./vectors.ts";

const BACK_REFERENCE_RE =
  /\b(it|its|it's|that|this|those|these|they|them|their|there|above|previous|same|again|more|another|instead|example|examples|why|how come)\b/i;
const FOLLOW_UP_START_RE =
  /^\s*(and|but|so|also|then|ok|okay|what about|how about|why|elaborate|explain more|tell me more|continue|go on)\b/i;
const MAX_QUERY_CHARS = 2000;

export function isFollowUp(latest: string, maxWords: number): boolean {
  if (latest.split(/\s+/).filter(Boolean).length > maxWords) return false;
  return FOLLOW_UP_START_RE.test(latest) || BACK_REFERENCE_RE.test(latest);
}

export function gateQueries(latest: string, previous: string | null, maxWords: number): string[] {
  const out = [latest.trim().slice(0, MAX_QUERY_CHARS)];
  if (previous && previous.trim() && isFollowUp(latest, maxWords)) {
    out.push(`${previous.trim().slice(0, MAX_QUERY_CHARS / 2)}\n${latest.trim()}`.slice(0, MAX_QUERY_CHARS));
  }
  return out;
}

/** Pool hits of several sources, best score first (stable), one entry per chunk. */
export function mergeHits(groups: readonly (readonly Hit[])[]): Hit[] {
  const pooled = new Map<string, Hit>();
  for (const group of groups) {
    for (const h of group) {
      const cur = pooled.get(h.chunk.chunk_id);
      if (!cur || h.score > cur.score) pooled.set(h.chunk.chunk_id, h);
    }
  }
  return [...pooled.values()].sort((a, b) => b.score - a.score);
}

export interface Packed {
  passages: Passage[];
  usedTokens: number;
  nextIndex: number;
  budget: number;
}

function toPassage(h: Hit, label: string, titles: ReadonlyMap<string, string>): Passage {
  return { ...h.chunk, label, source: h.source, score: h.score, title: titles.get(h.chunk.doc_id) ?? "" };
}

/** Rank order until the next chunk would exceed the budget (at least ``minPassages``). */
export function pack(
  ranked: readonly Hit[],
  budget: number,
  minPassages = 1,
  titles: ReadonlyMap<string, string> = new Map(),
  start: Packed | null = null,
): Packed {
  const passages = start ? [...start.passages] : [];
  let used = start ? start.usedTokens : 0;
  let i = start ? start.nextIndex : 0;
  while (i < ranked.length) {
    const h = ranked[i] as Hit;
    if (used + h.chunk.n_tokens > budget && passages.length >= minPassages) break;
    passages.push(toPassage(h, `C${passages.length + 1}`, titles));
    used += h.chunk.n_tokens;
    i += 1;
  }
  return { passages, usedTokens: used, nextIndex: i, budget };
}

/** The delimited passage block, exactly as the Python agents see it. */
export function passageBlock(p: Pick<Passage, "label" | "heading_path" | "section_ids" | "text">): string {
  return [
    `<<<PASSAGE ${p.label}>>>`,
    `Source: ${sanitize(p.heading_path)}`,
    `Sections: ${p.section_ids.join(", ") || "-"}`,
    "<<<TEXT>>>",
    sanitize(p.text.trim()),
    `<<<END ${p.label}>>>`,
  ].join("\n");
}

export interface SourceCheck {
  source: string;
  topScore: number | null;
  threshold: number;
  passed: boolean;
}

export interface GateDecision {
  grounded: boolean;
  queries: string[];
  checks: SourceCheck[];
  passages: Passage[];
  embedMs: number;
  searchMs: number;
}

export interface GateSettings {
  follow_up_max_words: number;
  search_k: number;
  max_passages: number;
}

export type Embed = (texts: string[]) => Promise<Float32Array[]>;

export async function decide(
  latest: string,
  previous: string | null,
  sources: readonly SearchSource[],
  budget: number,
  settings: GateSettings,
  embed: Embed,
  titles: ReadonlyMap<string, string> = new Map(),
): Promise<GateDecision> {
  const queries = gateQueries(latest, previous, settings.follow_up_max_words);
  if (!sources.length) return { grounded: false, queries, checks: [], passages: [], embedMs: 0, searchMs: 0 };
  const t0 = performance.now();
  const vectors = await embed(queries);
  const embedMs = performance.now() - t0;
  const t1 = performance.now();
  const checks: SourceCheck[] = [];
  const passing: Hit[][] = [];
  for (const src of sources) {
    const hits = src.search(vectors, settings.search_k);
    const top = hits[0]?.score ?? null;
    const passed = top !== null && top >= src.threshold;
    checks.push({ source: src.id, topScore: top, threshold: src.threshold, passed });
    if (passed) passing.push(hits);
  }
  const searchMs = performance.now() - t1;
  if (!passing.length) return { grounded: false, queries, checks, passages: [], embedMs, searchMs };
  const pooled = mergeHits(passing).slice(0, settings.max_passages);
  const packed = pack(pooled, budget, 1, titles);
  return { grounded: true, queries, checks, passages: packed.passages, embedMs, searchMs };
}
