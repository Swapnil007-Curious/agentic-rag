// @vitest-environment happy-dom
// UI units that need a DOM: storage-blocked mode, share-link prefill, citation chip <-> source
// card linking, reduced-motion class switching, error copy, fuzzy search and the trace.

import { afterEach, describe, expect, it, vi } from "vitest";
import { ProxyError } from "../../src/api.ts";
import { storage } from "../../src/storage.ts";
import { describeError, formatCountdown } from "../../src/ui/errors.ts";
import { excerpt, markTerms, queryTerms } from "../../src/ui/highlight.ts";
import { linkLabels } from "../../src/ui/linking.ts";
import { reducedMotion, syncReducedMotion } from "../../src/ui/motion.ts";
import { fuzzyScore } from "../../src/ui/palette.ts";
import { buildShareUrl, parseShareParams } from "../../src/ui/share.ts";
import { applyStage, finishTrace, initialTrace } from "../../src/ui/trace.ts";

afterEach(() => {
  vi.restoreAllMocks();
  document.documentElement.className = "";
});

describe("storage blocked", () => {
  it("works without localStorage: probe, get and set never throw", () => {
    const blocked = {
      getItem: () => {
        throw new DOMException("blocked", "SecurityError");
      },
      setItem: () => {
        throw new DOMException("blocked", "SecurityError");
      },
      removeItem: () => {
        throw new DOMException("blocked", "SecurityError");
      },
    };
    vi.stubGlobal("localStorage", blocked);
    Object.defineProperty(window, "localStorage", { value: blocked, configurable: true });
    storage.ok = true;
    expect(storage.probe()).toBe(false);
    expect(storage.get("x")).toBeNull();
    expect(storage.ok).toBe(false);
    expect(storage.set("x", { a: 1 })).toBe(false);
  });
});

describe("share links", () => {
  it("parses ?q= and ?c=, ignoring bad collection ids", () => {
    expect(parseShareParams("?q=What%20is%20a%20heap%3F&c=cs_ai")).toEqual({ q: "What is a heap?", c: "cs_ai" });
    expect(parseShareParams("?q=hello&c=../etc")).toEqual({ q: "hello", c: null });
    expect(parseShareParams("?c=indian_law")).toEqual({ q: "", c: "indian_law" });
    expect(parseShareParams("")).toBeNull();
    expect(parseShareParams("?q=" + "a".repeat(5000))?.q.length).toBe(4000);
  });
  it("builds a link under the site's base path", () => {
    const url = buildShareUrl("https://example.github.io", "/agentic-rag/", "Who may file a complaint?", "consumer_law");
    expect(url).toBe("https://example.github.io/agentic-rag/?q=Who+may+file+a+complaint%3F&c=consumer_law");
    expect(parseShareParams(new URL(url).search)).toEqual({ q: "Who may file a complaint?", c: "consumer_law" });
    expect(buildShareUrl("https://example.github.io", "/", "x", null)).toBe("https://example.github.io/?q=x");
  });
});

describe("citation chip <-> source card linking", () => {
  it("marks the chips and cards of one label and clears them", () => {
    document.body.innerHTML = `
      <div id="msg"><button class="cite-inline" data-label="C1">C1</button><button class="cite-inline" data-label="C2">C2</button></div>
      <div id="evidence"><article class="source-card" data-label="C1"></article><article class="source-card" data-label="C2"></article></div>`;
    const msg = document.getElementById("msg") as HTMLElement;
    const ev = document.getElementById("evidence") as HTMLElement;
    const card = linkLabels(ev, "C2", ".source-card");
    linkLabels(msg, "C2");
    expect(card?.dataset.label).toBe("C2");
    expect([...ev.querySelectorAll(".is-linked")].map((n) => (n as HTMLElement).dataset.label)).toEqual(["C2"]);
    expect([...msg.querySelectorAll(".is-linked")].map((n) => (n as HTMLElement).dataset.label)).toEqual(["C2"]);
    expect(linkLabels(ev, null, ".source-card")).toBeNull();
    expect(ev.querySelectorAll(".is-linked")).toHaveLength(0);
  });
});

describe("reduced motion", () => {
  it("mirrors prefers-reduced-motion onto html.reduce-motion and follows changes", () => {
    let matches = true;
    const listeners: (() => void)[] = [];
    const mql = {
      get matches() {
        return matches;
      },
      addEventListener: (_: string, fn: () => void) => listeners.push(fn),
    };
    Object.defineProperty(window, "matchMedia", { value: () => mql, configurable: true });
    syncReducedMotion();
    expect(document.documentElement.classList.contains("reduce-motion")).toBe(true);
    expect(reducedMotion()).toBe(true);
    matches = false;
    for (const fn of listeners) fn();
    expect(document.documentElement.classList.contains("reduce-motion")).toBe(false);
    expect(reducedMotion()).toBe(false);
  });
});

describe("error copy", () => {
  it("names the limit and the countdown; keeps network errors specific", () => {
    const rate = describeError(new ProxyError("rate_limited", "x", 429, 125));
    expect(rate.title).toBe("Limit reached: 20 questions per 10 minutes.");
    expect(rate.countdown).toBe(true);
    expect(formatCountdown(rate.retryAfter ?? 0)).toBe("02:05");
    const quota = describeError(new ProxyError("daily_cap", "x", 429, 7200));
    expect(quota.title).toBe("Today's free capacity is used up.");
    expect(quota.detail).toBe("It resets in about 2 hours.");
    const net = describeError(new ProxyError("network", "x", 0));
    expect(net.title).toBe("Can't reach the answer service.");
    expect(net.detail).toBe("Check your connection.");
    expect(describeError(new Error("boom")).title).toBe("boom");
  });
});

describe("palette fuzzy search", () => {
  it("prefers word starts and consecutive letters", () => {
    expect(fuzzyScore("hb", "Human Biology")).toBeGreaterThan(fuzzyScore("hb", "The hub"));
    expect(fuzzyScore("zzz", "Human Biology")).toBe(0);
    expect(fuzzyScore("", "anything")).toBe(1);
  });
});

describe("excerpts", () => {
  it("marks the question's words and windows around the first hit", () => {
    const terms = queryTerms("What does git rebase do to history?");
    expect(terms).toEqual(["rebas", "history"]);
    const text = `${"filler ".repeat(80)}Rebasing replays commits, which rewrites history.${" more".repeat(40)}`;
    const ex = excerpt(text, terms, 160);
    expect(ex.startsWith("…")).toBe(true);
    expect(ex).toContain("Rebasing");
    const frag = markTerms("Rebasing rewrites history.", terms);
    expect([...frag.childNodes].filter((n) => n.nodeName === "MARK").map((n) => n.textContent)).toEqual(["Rebasing", "history"]);
  });
});

describe("trace", () => {
  it("builds the stages of each mode and applies real events, adding revision rounds", () => {
    expect(initialTrace(false, true).map((s) => s.id)).toEqual(["embed", "search", "draft", "verify"]);
    expect(initialTrace(false, false).map((s) => s.id)).toEqual(["draft"]);
    let steps = initialTrace(true, true);
    expect(steps.map((s) => s.id)).toEqual(["embed", "search", "synth", "guards", "critic"]);
    steps = applyStage(steps, { stage: "embed", status: "done", ms: 12.4 });
    steps = applyStage(steps, { stage: "synth", status: "active" });
    steps = applyStage(steps, { stage: "synth", status: "done", ms: 900 });
    steps = applyStage(steps, { stage: "critic", status: "done", ms: 400 });
    steps = applyStage(steps, { stage: "synth", status: "active", note: "revision" });
    expect(steps.map((s) => `${s.id}:${s.status}`)).toEqual(["embed:done", "search:pending", "synth:done", "guards:pending", "critic:done", "synth:active"]);
    expect(steps[5]?.label).toBe("Synthesize (revision)");
    expect(finishTrace(steps)[5]?.status).toBe("done");
  });
});
