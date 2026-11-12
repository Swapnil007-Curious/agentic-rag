// "Grounded only" (port of the orchestrator in src/ragsys/agents/orchestrator.py, as system G4
// runs it): synthesizer -> deterministic guards -> critic, with at most one revision round on an
// extended context; abstains when the answer is not supported. Retrieval and the guards run in
// the browser; the synthesizer and critic calls go through the LLM proxy.

import { extractJson, parseSynth, readCritic, type CriticOutput, type SynthOutput } from "./jsonx.ts";
import { pack, type Packed } from "./gate.ts";
import { citedLabels, runGuards, subset, type GuardReport } from "./guards.ts";
import { splitSentences } from "./text.ts";
import type { GroundedSettings, Hit, Passage } from "./types.ts";
import type { ProxyPassage } from "./api.ts";

export interface GroundedLLM {
  synth(req: { question: string; passages: ProxyPassage[]; revision?: { previous: string; problems: string[]; new_labels: string[] } }): Promise<string>;
  critic(req: { question: string; passages: ProxyPassage[]; answer: string }): Promise<string>;
}

export interface ClosestSection {
  chunk_id: string;
  heading_path: string;
  pages: string;
}

export interface GroundedResult {
  outcome: "answered" | "abstained";
  reason: "synth_abstain" | "guard_fail" | "critic_abstain" | null;
  detail: string;
  answer: string;
  citations: string[];
  passages: Passage[]; // the context of the last round
  closest: ClosestSection[];
  sentences: { text: string; citations: string[] }[];
  llmCalls: number;
  guards: GuardReport | null;
  critic: CriticOutput | null;
}

export function toProxyPassages(passages: readonly Passage[]): ProxyPassage[] {
  return passages.map((p) => ({ label: p.label, heading: p.heading_path, sections: p.section_ids, text: p.text }));
}

export function closestSections(ranked: readonly Hit[], n = 3): ClosestSection[] {
  const out: ClosestSection[] = [];
  const seen = new Set<string>();
  for (const h of ranked) {
    const key = h.chunk.section_ids.join(",") || h.chunk.chunk_id;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ chunk_id: h.chunk.chunk_id, heading_path: h.chunk.heading_path, pages: h.chunk.pages });
    if (out.length === n) break;
  }
  return out;
}

export async function groundedAnswer(
  question: string,
  ranked: readonly Hit[],
  settings: GroundedSettings,
  llm: GroundedLLM,
  titles: ReadonlyMap<string, string> = new Map(),
): Promise<GroundedResult> {
  let ctx: Packed = pack(ranked, settings.budget_tokens, settings.min_passages, titles);
  let calls = 0;
  let criticCalls = 0;
  let revision: { previous: string; problems: string[]; new_labels: string[] } | undefined;
  let lastGuards: GuardReport | null = null;
  let lastCritic: CriticOutput | null = null;
  const finish = (outcome: GroundedResult["outcome"], reason: GroundedResult["reason"], detail: string, out?: SynthOutput): GroundedResult => {
    const answered = outcome === "answered" && out !== undefined;
    return {
      outcome,
      reason,
      detail,
      answer: answered ? out.answer : "",
      citations: answered ? citedLabels(out.answer, out.citations) : [],
      passages: ctx.passages,
      closest: answered ? [] : closestSections(ranked),
      sentences: answered ? splitSentences(out.answer).map((s) => ({ text: s.text, citations: s.labels })) : [],
      llmCalls: calls,
      guards: lastGuards,
      critic: lastCritic,
    };
  };
  const rounds = settings.max_synth_calls;
  for (let rnd = 1; rnd <= rounds; rnd++) {
    if (rnd > 1 && revision) {
      const before = ctx.passages.length;
      ctx = pack(ranked, settings.revision_budget_tokens, settings.min_passages, titles, ctx);
      revision.new_labels = ctx.passages.slice(before).map((p) => p.label);
    }
    calls += 1;
    const raw = await llm.synth({ question, passages: toProxyPassages(ctx.passages), ...(revision ? { revision } : {}) });
    let out: SynthOutput;
    try {
      out = parseSynth(extractJson(raw));
    } catch (err) {
      return finish("abstained", "synth_abstain", `unreadable synthesizer reply: ${err instanceof Error ? err.message : String(err)}`);
    }
    if (out.abstain) {
      return finish("abstained", "synth_abstain", `the synthesizer found no answer in the passages${out.note ? ` (${out.note})` : ""}`);
    }
    const report = runGuards(out.answer, out.citations, ctx.passages);
    lastGuards = report;
    let verdict: CriticOutput | null = null;
    if (criticCalls < settings.max_critic_calls && (report.passed || settings.critic_on_guard_fail)) {
      const cited = subset(ctx.passages, citedLabels(out.answer, out.citations));
      calls += 1;
      criticCalls += 1;
      verdict = readCritic(await llm.critic({ question, passages: toProxyPassages(cited), answer: out.answer }));
      lastCritic = verdict;
      if (verdict.verdict === "abstain") {
        return finish("abstained", "critic_abstain", verdict.feedback || "the critic found no answer in the passages");
      }
    }
    if (report.passed && (verdict === null || verdict.verdict === "accept")) {
      return finish("answered", null, verdict ? "guards passed and the critic accepted" : "guards passed", out);
    }
    if (rnd === rounds) {
      if (!report.passed) return finish("abstained", "guard_fail", report.reasons.join("; "));
      if (settings.on_final_revise === "answer") return finish("answered", null, "released although the critic asked for a revision", out);
      return finish("abstained", "critic_abstain", `the critic still found unsupported content after the revision${verdict?.feedback ? `: ${verdict.feedback}` : ""}`);
    }
    const problems = [...report.reasons];
    if (verdict) {
      problems.push(...verdict.unsupported_sentences.map((s) => `unsupported sentence: ${s}`));
      if (verdict.feedback) problems.push(`reviewer feedback: ${verdict.feedback}`);
    }
    revision = { previous: out.answer, problems, new_labels: [] };
  }
  return finish("abstained", "synth_abstain", "no round returned");
}
