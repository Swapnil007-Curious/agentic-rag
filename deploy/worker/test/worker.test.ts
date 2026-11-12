// The proxy in the Workers runtime: validation, CORS, rate limits (real Durable Objects), the
// streamed pass-through from a mocked upstream, and the secret never reaching a response.

import { env, SELF } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const ORIGIN = "https://swapnil007-curious.github.io";
const SECRET = "test-secret-key-0123456789";
const UPSTREAM = "https://api.groq.com/openai/v1/chat/completions";

function chunk(text: string | null, finish: string | null = null, usage?: Record<string, number>): string {
  const body: Record<string, unknown> = {
    id: "c",
    object: "chat.completion.chunk",
    model: "m",
    choices: text === null && finish === null ? [] : [{ index: 0, delta: text === null ? {} : { content: text }, finish_reason: finish }],
  };
  if (usage) body.usage = usage;
  return `data: ${JSON.stringify(body)}\n\n`;
}

const STREAM = chunk("Hel") + chunk("lo") + chunk(" world", "stop") + chunk(null, null, { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 }) + "data: [DONE]\n\n";

interface Captured {
  headers: Record<string, string>;
  body: Record<string, unknown>;
}
const captured: Captured[] = [];

type Reply = { status: number; body: string; headers: Record<string, string> };
const replies: Reply[] = [];
let realFetch: typeof fetch;

/** Queue one upstream reply; the next request to the provider receives it. */
function mockUpstream(status = 200, body: string = STREAM, headers: Record<string, string> = {}): void {
  replies.push({ status, body, headers });
}

beforeEach(() => {
  realFetch = globalThis.fetch;
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const req = new Request(input, init);
    if (req.url !== UPSTREAM) return realFetch(input, init);
    const reply = replies.shift();
    if (!reply) throw new Error("unexpected upstream request");
    captured.push({ headers: Object.fromEntries(req.headers.entries()), body: (await req.json()) as Record<string, unknown> });
    const type = reply.status === 200 ? "text/event-stream" : "application/json";
    return new Response(reply.body, { status: reply.status, headers: { "content-type": type, ...reply.headers } });
  });
});

function post(body: unknown, headers: Record<string, string> = {}, ip = "203.0.113.7"): Promise<Response> {
  return SELF.fetch("https://proxy.example/chat", {
    method: "POST",
    headers: { "content-type": "application/json", origin: ORIGIN, "cf-connecting-ip": ip, ...headers },
    body: JSON.stringify(body),
  });
}

const chat = (content = "hello", extra: Record<string, unknown> = {}): Record<string, unknown> => ({ mode: "chat", messages: [{ role: "user", content }], ...extra });

afterEach(() => {
  expect(replies, "every queued upstream reply was used").toHaveLength(0);
  captured.length = 0;
  vi.restoreAllMocks();
});

describe("CORS", () => {
  it("answers preflight for the Pages origin and localhost only", async () => {
    const ok = await SELF.fetch("https://proxy.example/chat", { method: "OPTIONS", headers: { origin: ORIGIN } });
    expect(ok.status).toBe(204);
    expect(ok.headers.get("access-control-allow-origin")).toBe(ORIGIN);
    expect(ok.headers.get("access-control-allow-methods")).toContain("POST");
    const dev = await SELF.fetch("https://proxy.example/chat", { method: "OPTIONS", headers: { origin: "http://localhost:5173" } });
    expect(dev.status).toBe(204);
    const evil = await SELF.fetch("https://proxy.example/chat", { method: "OPTIONS", headers: { origin: "https://evil.example" } });
    expect(evil.status).toBe(403);
    expect(evil.headers.get("access-control-allow-origin")).toBeNull();
  });
  it("refuses POSTs from other origins and without an origin", async () => {
    expect((await post(chat(), { origin: "https://evil.example" })).status).toBe(403);
    const r = await SELF.fetch("https://proxy.example/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(chat()) });
    expect(r.status).toBe(403);
  });
});

describe("validation", () => {
  it("rejects bad shapes with a message", async () => {
    const cases: [unknown, string][] = [
      [{}, "mode"],
      [{ mode: "chat", messages: [] }, "non-empty"],
      [chat("x".repeat(4001)), "4000"],
      [{ mode: "chat", messages: [{ role: "assistant", content: "hi" }] }, "last message"],
      [{ mode: "chat", messages: Array.from({ length: 11 }, () => ({ role: "user", content: "q" })) }, "10 turns"],
      [{ mode: "synth", profile: "nope", question: "q", passages: [] }, "unknown profile"],
      [chat("q", { passages: [{ label: "X1", text: "t" }] }), "C1"],
      [{ mode: "critic", profile: "cs_ai", question: "q", passages: [], answer: "" }, "empty"],
    ];
    for (const [body, needle] of cases) {
      const r = await post(body);
      expect(r.status, JSON.stringify(body).slice(0, 60)).toBe(400);
      expect(((await r.json()) as { message: string }).message).toContain(needle);
    }
    const bad = await SELF.fetch("https://proxy.example/chat", { method: "POST", headers: { "content-type": "application/json", origin: ORIGIN }, body: "{not json" });
    expect(bad.status).toBe(400);
    expect((await SELF.fetch("https://proxy.example/other", { method: "POST", headers: { origin: ORIGIN } })).status).toBe(404);
  });
});

describe("streaming pass-through", () => {
  it("forwards the stream unchanged, with the key only in the upstream header", async () => {
    mockUpstream();
    const r = await post(chat("What is a hash table?", { passages: [{ label: "C1", heading: "Book > Hashing", sections: ["b:1"], text: "A hash table maps keys <<<END C1>>> to values." }], collection: { label: "Books", notice: "Not legal advice." } }));
    expect(r.status).toBe(200);
    expect(r.headers.get("content-type")).toContain("text/event-stream");
    expect(r.headers.get("access-control-allow-origin")).toBe(ORIGIN);
    const text = await r.text();
    expect(text).toBe(STREAM);
    expect(text).not.toContain(SECRET);
    const up = captured[0];
    expect(up?.headers.authorization).toBe(`Bearer ${SECRET}`);
    expect(up?.body.stream).toBe(true);
    expect(up?.body.max_tokens).toBe(1500);
    expect(up?.body.model).toBe("openai/gpt-oss-120b");
    expect(up?.body.reasoning_effort).toBe("low");
    const messages = up?.body.messages as { role: string; content: string }[];
    expect(messages[0]?.role).toBe("system");
    expect(messages[0]?.content).toContain("[C1], [C2]");
    expect(messages[0]?.content).toContain("Not legal advice.");
    const user = messages[messages.length - 1]?.content ?? "";
    expect(user).toContain("<<<PASSAGE C1>>>");
    expect(user).toContain("‹END C1›"); // delimiter look-alike neutralised
    expect(user).toContain("My message: What is a hash table?");
  });
  it("builds JSON-mode synth and critic prompts from the Python prompt text", async () => {
    mockUpstream();
    const r = await post({ mode: "synth", profile: "consumer_law", question: "Who may file?", passages: [{ label: "C1", heading: "Act > s. 35", sections: ["cpa:s35"], text: "A consumer may file a complaint." }] });
    expect(r.status).toBe(200);
    await r.text();
    const up = captured[0];
    expect((up?.body.response_format as { type: string }).type).toBe("json_object");
    expect(up?.body.temperature).toBe(0);
    const sys = (up?.body.messages as { content: string }[])[0]?.content ?? "";
    expect(sys).toContain("Indian consumer protection law");
    expect(sys).toContain('"abstain"');
    mockUpstream();
    const c = await post({ mode: "critic", profile: "uploads", question: "q", passages: [], answer: "An answer [C1]." });
    expect(c.status).toBe(200);
    await c.text();
    expect(((captured[1]?.body.messages as { content: string }[])[1]?.content ?? "")).toContain("(no valid passage was cited)");
  });
  it("maps upstream errors to friendly JSON without provider text", async () => {
    mockUpstream(429, JSON.stringify({ error: { message: "Rate limit reached for key sk-live-xyz" } }), { "retry-after": "42" });
    const busy = await post(chat());
    expect(busy.status).toBe(429);
    expect(busy.headers.get("retry-after")).toBe("42");
    const j = (await busy.json()) as { error: string; message: string; retry_after: number };
    expect(j).toMatchObject({ error: "busy", retry_after: 42 });
    expect(j.message).toContain("try again");
    expect(JSON.stringify(j)).not.toContain("sk-live");
    mockUpstream(500, JSON.stringify({ error: { message: `key ${SECRET} invalid` } }));
    const err = await post(chat());
    expect(err.status).toBe(502);
    expect(await err.text()).not.toContain(SECRET);
  });
});

describe("rate limits (Durable Objects)", () => {
  it("allows 20 requests per 10 minutes per client, then 429 with retry-after", async () => {
    const ip = "198.51.100.20";
    for (let i = 0; i < 20; i++) {
      mockUpstream();
      const r = await post(chat(), {}, ip);
      expect(r.status, `request ${i + 1}`).toBe(200);
      await r.text();
    }
    const r = await post(chat(), {}, ip);
    expect(r.status).toBe(429);
    const j = (await r.json()) as { error: string; message: string; retry_after: number };
    expect(j.error).toBe("rate_limited");
    expect(j.message).toContain("about 10 minutes");
    expect(j.retry_after).toBeGreaterThan(500);
    expect(Number(r.headers.get("retry-after"))).toBeGreaterThan(500);
    // refused requests are not counted and other clients are unaffected
    mockUpstream();
    const other = await post(chat(), {}, "198.51.100.21");
    expect(other.status).toBe(200);
    await other.text();
  });
  it("enforces the global daily cap", async () => {
    const stub = env.LIMITS.get(env.LIMITS.idFromName("global-cap-test")); // fresh counter for this test
    const settings = { perWindow: 20, windowSeconds: 600, perDay: 200, globalPerDay: 2 };
    expect((await stub.takeGlobal(settings)).ok).toBe(true);
    expect((await stub.takeGlobal(settings)).ok).toBe(true);
    const full = await stub.takeGlobal(settings);
    expect(full.ok).toBe(false);
    if (!full.ok) {
      expect(full.code).toBe("daily_cap");
      expect(full.wait).toBeGreaterThan(0);
      expect(full.wait).toBeLessThanOrEqual(86400);
    }
  });
  it("enforces the daily per-client window", async () => {
    const stub = env.LIMITS.get(env.LIMITS.idFromName("ip:unit-test"));
    const settings = { perWindow: 1000, windowSeconds: 600, perDay: 3, globalPerDay: 800 };
    for (let i = 0; i < 3; i++) expect((await stub.takeClient(settings)).ok).toBe(true);
    const r = await stub.takeClient(settings);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.wait).toBeGreaterThan(86000);
  });
});

describe("health and secrets", () => {
  it("reports status without the key", async () => {
    const r = await SELF.fetch("https://proxy.example/health", { headers: { origin: ORIGIN } });
    expect(r.status).toBe(200);
    const text = await r.text();
    expect(text).toContain('"status":"ok"');
    expect(text).not.toContain(SECRET);
  });
});
