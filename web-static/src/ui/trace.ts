// The verification trace: the real pipeline stages of one answer, each pending -> active -> done
// with its measured duration. Connectors light up left to right as stages finish.

import { el } from "./dom.ts";
import type { StageEvent } from "../engine.ts";
import type { Verdict } from "./seal.ts";

export type StepStatus = "pending" | "active" | "done" | "skipped";

export interface TraceStep {
  id: string;
  label: string;
  status: StepStatus;
  ms: number | null;
  note?: string;
  verdict?: Verdict;
}

const LABELS: Record<string, string> = {
  embed: "Embed",
  search: "Search",
  draft: "Draft",
  synth: "Synthesize",
  guards: "Guards",
  critic: "Critic",
  verify: "Verify",
};

/** The expected stages of a mode, all pending, so the strip reserves its space before the first event. */
export function initialTrace(groundedOnly: boolean, hasDocuments: boolean): TraceStep[] {
  const ids = groundedOnly ? ["embed", "search", "synth", "guards", "critic"] : hasDocuments ? ["embed", "search", "draft", "verify"] : ["draft"];
  return ids.map((id) => ({ id, label: LABELS[id] ?? id, status: "pending", ms: null }));
}

/** Apply a stage event; a stage that repeats (a revision round) gets a new step. */
export function applyStage(steps: TraceStep[], e: StageEvent): TraceStep[] {
  const out = steps.map((s) => ({ ...s }));
  let idx = out.findIndex((s) => s.id === e.stage && s.status !== "done" && s.status !== "skipped");
  if (idx < 0) {
    const label = `${LABELS[e.stage] ?? e.stage}${e.note === "revision" ? " (revision)" : ""}`;
    // a repeated stage (the revision round) is appended in order of occurrence
    const step: TraceStep = { id: e.stage, label, status: "pending", ms: null };
    idx = out.length;
    out.push(step);
  }
  const step = out[idx] as TraceStep;
  step.status = e.status;
  if (e.ms !== undefined) step.ms = e.ms;
  if (e.note && e.note !== "revision") step.note = e.note;
  if (e.note === "revision" && !step.label.includes("revision")) step.label = `${step.label} (revision)`;
  return out;
}

export function finishTrace(steps: TraceStep[]): TraceStep[] {
  return steps.map((s) => (s.status === "active" ? { ...s, status: "done" as const } : s));
}

function ms(n: number | null): string {
  if (n === null) return "";
  return n >= 1000 ? `${(n / 1000).toFixed(1)} s` : `${Math.round(n)} ms`;
}

export function renderTrace(host: HTMLElement, steps: TraceStep[], opts: { onVerify?: () => void; verifying?: boolean } = {}): void {
  host.textContent = "";
  host.className = "trace";
  host.setAttribute("role", "list");
  host.setAttribute("aria-label", "Verification trace");
  steps.forEach((s, i) => {
    if (i > 0) {
      const link = el("span", "trace-link");
      const prev = steps[i - 1];
      if (prev && (prev.status === "done" || prev.status === "skipped")) link.classList.add("is-lit");
      host.appendChild(link);
    }
    const step = el("span", "trace-step");
    step.setAttribute("role", "listitem");
    step.dataset.status = s.status;
    if (s.verdict) step.dataset.verdict = s.verdict;
    step.appendChild(el("span", "trace-node"));
    if (s.id === "verify" && s.status === "pending" && opts.onVerify) {
      const b = el("button", "trace-btn", opts.verifying ? "Verifying…" : "Verify");
      b.type = "button";
      b.disabled = Boolean(opts.verifying);
      b.setAttribute("aria-label", "Verify this answer against its sources");
      b.addEventListener("click", opts.onVerify);
      step.appendChild(b);
    } else {
      step.appendChild(el("span", "trace-label", s.label));
    }
    const t = el("span", "trace-ms mono", s.status === "skipped" ? (s.note ?? "skipped") : ms(s.ms) || (s.note ?? ""));
    step.appendChild(t);
    const title = [s.label, s.status, s.ms !== null ? ms(s.ms) : "", s.note ?? ""].filter(Boolean).join(" · ");
    step.title = title;
    host.appendChild(step);
  });
}
