// Error and limit copy, in the interface's voice: specific, never apologetic. The Worker's JSON
// gives a code and retry_after; the client gives "network" when the fetch itself failed.

import { ProxyError } from "../api.ts";

export interface ErrorView {
  kind: "rate" | "quota" | "busy" | "network" | "setup" | "upstream" | "other";
  title: string;
  detail: string | null;
  retryAfter: number | null; // seconds
  countdown: boolean;
}

export function formatCountdown(seconds: number): string {
  const s = Math.max(0, Math.ceil(seconds));
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

export function describeError(err: unknown): ErrorView {
  if (err instanceof ProxyError) {
    const retry = err.retryAfter;
    switch (err.code) {
      case "rate_limited":
        return { kind: "rate", title: "Limit reached: 20 questions per 10 minutes.", detail: retry === null ? "Try again in a few minutes." : null, retryAfter: retry, countdown: retry !== null };
      case "daily_cap": {
        const hours = retry === null ? null : Math.max(1, Math.round(retry / 3600));
        return { kind: "quota", title: "Today's free capacity is used up.", detail: hours === null ? "It resets at midnight UTC." : `It resets in about ${hours} hour${hours === 1 ? "" : "s"}.`, retryAfter: retry, countdown: false };
      }
      case "busy":
        return { kind: "busy", title: "The answer service is busy.", detail: retry === null ? "Try again in a minute." : null, retryAfter: retry, countdown: retry !== null };
      case "network":
        return { kind: "network", title: "Can't reach the answer service.", detail: "Check your connection.", retryAfter: null, countdown: false };
      case "not_configured":
        return { kind: "setup", title: "This site has no answer service configured.", detail: "See the README: Public demo.", retryAfter: null, countdown: false };
      case "no_documents":
        return { kind: "other", title: "Grounded only needs documents.", detail: "Choose a collection or attach a PDF.", retryAfter: null, countdown: false };
      case "loading":
        return { kind: "other", title: "Still preparing the search index.", detail: "The question will send when it is ready.", retryAfter: null, countdown: false };
      case "upstream":
      case "unavailable":
        return { kind: "upstream", title: "The model stopped with an error.", detail: "Try again in a moment.", retryAfter: null, countdown: false };
      default:
        return { kind: "other", title: err.message || "The answer could not be produced.", detail: err.status ? `HTTP ${err.status}` : null, retryAfter: null, countdown: false };
    }
  }
  const message = err instanceof Error ? err.message : String(err);
  return { kind: "other", title: message || "Something went wrong.", detail: null, retryAfter: null, countdown: false };
}
