import { describe, expect, it } from "vitest";
import { messagesDigest, planHistory, transcript, type Turn } from "../../src/history.ts";
import { extractJson, parseCritic, parseSynth, readCritic } from "../../src/jsonx.ts";

function conversation(n: number, words = 10): Turn[] {
  const out: Turn[] = [];
  for (let i = 0; i < n; i++) {
    out.push({ role: "user", content: `question ${i} ${"word ".repeat(words)}` });
    out.push({ role: "assistant", content: `answer ${i} ${"text ".repeat(words)}` });
  }
  out.push({ role: "user", content: "latest question" });
  return out;
}

describe("history", () => {
  it("keeps the last ten turns without a summary when they fit", async () => {
    const calls: string[] = [];
    const plan = await planHistory(conversation(15), 100000, { maxTurns: 10, summarize: async (p) => (calls.push(p), "s") });
    expect(calls).toEqual([]);
    expect(plan.summary).toBeNull();
    expect(plan.kept).toHaveLength(19);
    expect(plan.kept[0]?.content.startsWith("question 6")).toBe(true);
    expect(plan.dropped).toBe(12);
  });
  it("summarises once when over budget and reuses the summary", async () => {
    const turns = conversation(8, 60);
    let calls = 0;
    const summarize = async (): Promise<string> => (calls++, "summary text");
    const first = await planHistory(turns, 600, { summarize, targetShare: 0.5 });
    expect(calls).toBe(1);
    expect(first.newSummary).toBe(true);
    expect(first.tokens).toBeLessThanOrEqual(600);
    expect(first.kept[0]?.role).toBe("user");
    const k = first.summary?.upto ?? 0;
    expect(first.summary?.digest).toBe(messagesDigest(turns.slice(0, k)));
    const more: Turn[] = [...turns, { role: "assistant", content: "short" }, { role: "user", content: "next" }];
    const second = await planHistory(more, 600, { summary: first.summary, summarize });
    expect(calls).toBe(1);
    expect(second.summary).toEqual(first.summary);
    const edited: Turn[] = [{ role: "user", content: "changed" }, ...more.slice(1)];
    const third = await planHistory(edited, 600, { summary: first.summary, summarize });
    expect(calls).toBe(2);
    expect(third.newSummary).toBe(true);
  });
  it("drops old messages with a note when no summary can be written", async () => {
    const plan = await planHistory(conversation(8, 60), 600, { summarize: async () => "" });
    expect(plan.summary).toBeNull();
    expect(plan.dropped).toBeGreaterThan(0);
    expect(plan.notes[0]).toContain("could not be written");
    await expect(planHistory([{ role: "user", content: "a" }, { role: "assistant", content: "b" }], 100)).rejects.toThrow();
  });
  it("clips the transcript", () => {
    const t = transcript([{ role: "user", content: "a".repeat(5000) }, { role: "assistant", content: "b" }], 3000);
    expect(t.length).toBeLessThanOrEqual(3000);
    expect(t.endsWith("Assistant: b")).toBe(true);
  });
});

describe("model JSON", () => {
  it("extracts JSON from fences and prose", () => {
    expect(extractJson('Sure:\n```json\n{"abstain": false, "answer": "x [C1].", "citations": ["C1"]}\n```')).toEqual({ abstain: false, answer: "x [C1].", citations: ["C1"] });
    expect(extractJson('text {"a": {"b": "}"}} tail')).toEqual({ a: { b: "}" } });
    expect(() => extractJson("no json")).toThrow();
  });
  it("parses synthesizer replies like the Python version", () => {
    expect(parseSynth({ abstain: "true", answer: "ignored" })).toMatchObject({ abstain: true, answer: "", citations: [] });
    expect(parseSynth({ answer: "Three months 【C1】.", citations: "C1, c2" })).toMatchObject({ abstain: false, answer: "Three months [C1].", citations: ["C1", "C2"] });
    expect(parseSynth({ answer: "The passages do not mention this." })).toMatchObject({ abstain: true, note: "answer text is a refusal" });
    expect(parseSynth({ answer: "" })).toMatchObject({ abstain: true, note: "empty answer" });
    expect(() => parseSynth({ answer: "x", citations: ["zz"] })).toThrow();
  });
  it("parses critic replies and treats unreadable ones as no objection", () => {
    expect(parseCritic({ verdict: "Rejected", unsupported_sentences: "x", feedback: null })).toEqual({ verdict: "revise", unsupported_sentences: ["x"], feedback: "", error: null });
    expect(readCritic("garbage").verdict).toBe("accept");
    expect(readCritic("garbage").error).not.toBeNull();
  });
});
