// One chat request in the static app (the browser counterpart of src/ragsys/assistant/service.py):
// the gate searches the selected collection and the uploads here, the history is capped (and
// summarised through the proxy when needed), and the reply streams from the LLM proxy. Grounded
// only and Verify reuse the grounded pipeline's logic (grounded.ts, verify.ts).

import { callProxy, ProxyError, type ProxyPassage } from "./api.ts";
import { cleanCitations, sourceLine } from "./citations.ts";
import type { CollectionStore, Progress } from "./data.ts";
import type { Embedder } from "./embedder.ts";
import { decide, gateQueries, mergeHits } from "./gate.ts";
import { groundedAnswer, toProxyPassages, type ClosestSection, type GroundedResult } from "./grounded.ts";
import type { GuardReport } from "./guards.ts";
import type { CriticOutput } from "./jsonx.ts";
import { planHistory, transcript, type Summary, type Turn } from "./history.ts";
import type { Manifest, Passage } from "./types.ts";
import type { UploadStore } from "./uploads.ts";
import { verifyAnswer, type VerifyResult } from "./verify.ts";
import type { SearchSource } from "./vectors.ts";

export interface ChatRequest {
  turns: Turn[];
  collection: string | null;
  groundedOnly: boolean;
  summary: Summary | null;
}

/** A pipeline stage as it starts and finishes (for the UI's verification trace; informational only). */
export type StageId = "embed" | "search" | "draft" | "synth" | "guards" | "critic";
export interface StageEvent {
  stage: StageId;
  status: "active" | "done" | "skipped";
  ms?: number;
  note?: string;
}

export interface ChatCallbacks {
  onContext?(grounded: boolean, passages: Passage[]): void;
  onToken?(t: string): void;
  onStatus?(text: string): void;
  onIndexProgress?: Progress;
  onStage?(event: StageEvent): void;
}

/** What the grounded pipeline checked (the fields groundedAnswer already returns). */
export interface GroundedCheck {
  guards: GuardReport | null;
  critic: CriticOutput | null;
  reason: GroundedResult["reason"];
  detail: string;
  llmCalls: number;
}

export interface ChatOutcome {
  mode: "chat" | "grounded_only";
  answer: string;
  citations: string[];
  passages: Passage[]; // the cited passages (shown as chips; kept for Verify)
  grounded: boolean;
  sourceLine: string;
  stopped: boolean;
  summary: Summary | null;
  outcome?: "answered" | "abstained";
  closest?: ClosestSection[];
  abstentionMessage?: string;
  check?: GroundedCheck;
  timings: { gate_ms: number; ttft_ms: number | null; total_ms: number; embed_ms?: number; search_ms?: number };
}

export class Assistant {
  constructor(
    readonly manifest: Manifest,
    readonly store: CollectionStore,
    readonly embedder: Embedder,
    readonly uploads: UploadStore,
    readonly workerUrl: string,
  ) {}

  async sources(collection: string | null, onProgress?: Progress): Promise<SearchSource[]> {
    const out: SearchSource[] = [];
    if (collection) out.push(await this.store.load(collection, onProgress));
    out.push(...this.uploads.sources());
    return out;
  }

  private profileOf(collection: string | null): string {
    return (collection && this.store.meta(collection)?.profile) || "uploads";
  }

  private summarizer(signal?: AbortSignal) {
    return async (previous: string, turns: Turn[]): Promise<string> => {
      const r = await callProxy(
        this.workerUrl,
        { mode: "summary", ...(previous ? { previous_summary: previous.slice(0, 4000) } : {}), transcript: transcript(turns, this.manifest.history.summary_input_chars) },
        signal ? { signal } : {},
      );
      return r.text.trim();
    };
  }

  async chat(req: ChatRequest, cb: ChatCallbacks = {}, signal?: AbortSignal): Promise<ChatOutcome> {
    const t0 = performance.now();
    if (req.groundedOnly) return this.grounded(req, cb, signal, t0);
    const latest = req.turns[req.turns.length - 1]?.content.trim() ?? "";
    const previous = [...req.turns.slice(0, -1)].reverse().find((t) => t.role === "user")?.content ?? null;
    const sources = await this.sources(req.collection, cb.onIndexProgress);
    await this.embedder.ready;
    const titles = this.uploads.titles();
    const g = this.manifest.gate;
    cb.onStage?.({ stage: "embed", status: "active" });
    const decision = await decide(latest, previous, sources, this.manifest.context_tokens, g, (q) => this.embedder.queries(q), titles);
    if (sources.length) {
      cb.onStage?.({ stage: "embed", status: "done", ms: decision.embedMs });
      cb.onStage?.({ stage: "search", status: "done", ms: decision.searchMs, note: decision.grounded ? `${decision.passages.length} passages` : "no match" });
    } else {
      cb.onStage?.({ stage: "embed", status: "skipped", note: "no documents" });
      cb.onStage?.({ stage: "search", status: "skipped", note: "no documents" });
    }
    cb.onContext?.(decision.grounded, decision.passages);
    const plan = await planHistory(req.turns, this.manifest.history.budget_tokens, {
      maxTurns: this.manifest.history.max_turns,
      summary: req.summary,
      summarize: this.summarizer(signal),
      targetShare: this.manifest.history.summary_target_share,
    });
    const meta = req.collection ? this.store.meta(req.collection) : undefined;
    const tReq = performance.now();
    let first: number | null = null;
    let text = "";
    let stopped = false;
    cb.onStage?.({ stage: "draft", status: "active" });
    try {
      await callProxy(
        this.workerUrl,
        {
          mode: "chat",
          messages: plan.kept.map((t) => ({ role: t.role, content: t.content.slice(0, this.manifest.limits.max_message_chars) })),
          ...(decision.grounded ? { passages: toProxyPassages(decision.passages) } : {}),
          ...(plan.summary ? { summary: plan.summary.text.slice(0, 4000) } : {}),
          ...(meta ? { collection: { label: meta.label, notice: meta.notice } } : {}),
        },
        {
          ...(signal ? { signal } : {}),
          onToken: (t) => {
            first ??= performance.now();
            text += t;
            cb.onToken?.(t);
          },
        },
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") stopped = true;
      else throw err;
    }
    cb.onStage?.({ stage: "draft", status: "done", ms: performance.now() - tReq, ...(first === null ? {} : { note: `first token ${Math.round(first - tReq)} ms` }) });
    const valid = decision.passages.map((p) => p.label);
    const cleaned = cleanCitations(text, valid);
    const cited = decision.passages.filter((p) => cleaned.used.includes(p.label));
    return {
      mode: "chat",
      answer: cleaned.text,
      citations: cleaned.used,
      passages: cited,
      grounded: cleaned.used.length > 0,
      sourceLine: sourceLine(cleaned.used.length),
      stopped,
      summary: plan.newSummary ? plan.summary : null,
      timings: {
        gate_ms: decision.embedMs + decision.searchMs,
        ttft_ms: first === null ? null : first - tReq,
        total_ms: performance.now() - t0,
        embed_ms: decision.embedMs,
        search_ms: decision.searchMs,
      },
    };
  }

  private async grounded(req: ChatRequest, cb: ChatCallbacks, signal: AbortSignal | undefined, t0: number): Promise<ChatOutcome> {
    if (!req.collection && !this.uploads.docs.length) {
      throw new ProxyError("no_documents", "Grounded only needs documents: choose a collection or upload a PDF.", 400);
    }
    cb.onStatus?.("Searching and checking the documents…");
    const latest = req.turns[req.turns.length - 1]?.content.trim() ?? "";
    const previous = [...req.turns.slice(0, -1)].reverse().find((t) => t.role === "user")?.content ?? null;
    const sources = await this.sources(req.collection, cb.onIndexProgress);
    await this.embedder.ready;
    const g = this.manifest.gate;
    cb.onStage?.({ stage: "embed", status: "active" });
    const tEmbed = performance.now();
    const vectors = await this.embedder.queries(gateQueries(latest, previous, g.follow_up_max_words));
    const embedMs = performance.now() - tEmbed;
    cb.onStage?.({ stage: "embed", status: "done", ms: embedMs });
    const tSearch = performance.now();
    const ranked = mergeHits(sources.map((s) => s.search(vectors, g.search_k * 2))).slice(0, g.search_k * 2);
    const searchMs = performance.now() - tSearch;
    cb.onStage?.({ stage: "search", status: "done", ms: searchMs, note: `${ranked.length} candidates` });
    const meta = req.collection ? this.store.meta(req.collection) : undefined;
    const settings = meta?.grounded ?? this.manifest.uploads_grounded;
    const profile = this.profileOf(req.collection);
    const opts = signal ? { signal } : {};
    let synthCalls = 0;
    let guardsReported = false;
    const result = await groundedAnswer(
      latest,
      ranked,
      settings,
      {
        synth: async (r) => {
          synthCalls += 1;
          const note = synthCalls > 1 ? { note: "revision" } : {};
          cb.onStage?.({ stage: "synth", status: "active", ...note });
          const t = performance.now();
          const text = (await callProxy(this.workerUrl, { mode: "synth", profile, ...r }, opts)).text;
          cb.onStage?.({ stage: "synth", status: "done", ms: performance.now() - t, ...note });
          cb.onStage?.({ stage: "guards", status: "active" });
          guardsReported = false;
          return text;
        },
        critic: async (r) => {
          // the guards ran between the synthesizer reply and this call (inside groundedAnswer)
          cb.onStage?.({ stage: "guards", status: "done" });
          guardsReported = true;
          cb.onStage?.({ stage: "critic", status: "active" });
          const t = performance.now();
          const text = (await callProxy(this.workerUrl, { mode: "critic", profile, ...r }, opts)).text;
          cb.onStage?.({ stage: "critic", status: "done", ms: performance.now() - t });
          return text;
        },
      },
      this.uploads.titles(),
    );
    if (!guardsReported && result.guards) cb.onStage?.({ stage: "guards", status: "done", note: result.guards.passed ? "passed" : "failed" });
    else if (!guardsReported) cb.onStage?.({ stage: "guards", status: "skipped" });
    if (!result.critic) cb.onStage?.({ stage: "critic", status: "skipped" });
    const answered = result.outcome === "answered";
    const cited = result.passages.filter((p) => result.citations.includes(p.label));
    const message = meta?.abstention_message ?? "Not found in these documents";
    return {
      mode: "grounded_only",
      answer: result.answer,
      citations: answered ? cited.map((p) => p.label) : [],
      passages: answered ? cited : [],
      grounded: answered,
      sourceLine: answered ? sourceLine(cited.length) : message,
      stopped: false,
      summary: null,
      outcome: result.outcome,
      closest: result.closest,
      abstentionMessage: message,
      check: { guards: result.guards, critic: result.critic, reason: result.reason, detail: result.detail, llmCalls: result.llmCalls },
      timings: { gate_ms: embedMs + searchMs, ttft_ms: null, total_ms: performance.now() - t0, embed_ms: embedMs, search_ms: searchMs },
    };
  }

  /** Guards here, critic through the proxy, on the passages the reply cites. */
  verify(question: string, answer: string, passages: readonly Passage[], collection: string | null): Promise<VerifyResult> {
    const profile = this.profileOf(collection);
    return verifyAnswer(answer, passages, async (cited, prose) => {
      const body: ProxyPassage[] = cited.map((p) => ({ label: p.label, heading: p.heading_path, sections: p.section_ids, text: p.text }));
      const r = await callProxy(this.workerUrl, { mode: "critic", profile, question: question.slice(0, 4000), passages: body, answer: prose.slice(0, 8000) });
      return r.text;
    });
  }
}
