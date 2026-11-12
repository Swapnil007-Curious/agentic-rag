// LLM proxy for the static study assistant (web-static/), a Cloudflare Worker.
//
// POST /chat takes one of four structured requests (chat, summary, synth, critic), builds the
// prompts here from prompts.json (the same text as the Python app), forwards the result to the
// provider's OpenAI-compatible chat completions with stream: true and passes the Server-Sent
// Events back unchanged. The API key lives only in the LLM_API_KEY secret; it is read for the
// upstream Authorization header and nowhere else (never logged, never echoed). Message contents
// are not stored or logged. Per-client limits (CF-Connecting-IP): a short window and a daily one;
// a global daily cap shields the key when the site is busy (see limits.ts for the storage choice).

import { buildMessages, validate, type ProxyRequest, type ValidationError } from "./request.ts";
import { clientLimiter, globalCounter, LimitError, type LimitSettings } from "./limits.ts";

export interface Env {
  LLM_API_KEY: string;
  LLM_BASE_URL?: string;
  LLM_MODEL?: string;
  LLM_EXTRA_BODY?: string; // JSON merged into the upstream request, e.g. {"reasoning_effort":"low"}
  ALLOWED_ORIGINS?: string; // comma separated; default: the Pages site and localhost
  MAX_OUTPUT_TOKENS?: string;
  LIMIT_PER_10MIN?: string;
  LIMIT_PER_DAY?: string;
  GLOBAL_PER_DAY?: string;
  LIMITS: DurableObjectNamespace;
}

export { RateLimiter } from "./limits.ts";

const DEFAULT_ORIGINS = ["https://swapnil007-curious.github.io"];
const DEFAULT_BASE_URL = "https://api.groq.com/openai/v1";
const DEFAULT_MODEL = "openai/gpt-oss-120b";
const BUSY = "The assistant is busy right now. Please try again in a minute.";

function settings(env: Env): LimitSettings {
  const num = (raw: string | undefined, fallback: number): number => {
    const n = Number.parseInt(raw ?? "", 10);
    return Number.isFinite(n) && n > 0 ? n : fallback;
  };
  return {
    perWindow: num(env.LIMIT_PER_10MIN, 20),
    windowSeconds: 600,
    perDay: num(env.LIMIT_PER_DAY, 200),
    globalPerDay: num(env.GLOBAL_PER_DAY, 800),
  };
}

export function allowedOrigins(env: Env): string[] {
  const extra = (env.ALLOWED_ORIGINS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  return extra.length ? extra : DEFAULT_ORIGINS;
}

/** Development origins: any http://localhost or http://127.0.0.1 port. */
export function originAllowed(origin: string | null, env: Env): boolean {
  if (!origin) return false;
  if (/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)) return true;
  return allowedOrigins(env).includes(origin);
}

function corsHeaders(origin: string): Record<string, string> {
  return {
    "access-control-allow-origin": origin,
    "access-control-allow-methods": "POST, OPTIONS",
    "access-control-allow-headers": "content-type",
    "access-control-max-age": "86400",
    vary: "origin",
  };
}

function json(status: number, body: Record<string, unknown>, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...headers },
  });
}

function clientAddress(request: Request): string {
  return request.headers.get("cf-connecting-ip") ?? "unknown";
}

async function upstream(env: Env, body: ProxyRequest, maxTokens: number): Promise<Response> {
  const built = buildMessages(body);
  const extra = env.LLM_EXTRA_BODY ? (JSON.parse(env.LLM_EXTRA_BODY) as Record<string, unknown>) : {};
  const payload: Record<string, unknown> = {
    model: env.LLM_MODEL || DEFAULT_MODEL,
    messages: built.messages,
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: Math.min(maxTokens, built.maxTokens),
    temperature: built.temperature,
    ...extra,
  };
  if (built.json) payload.response_format = { type: "json_object" };
  return fetch(`${(env.LLM_BASE_URL || DEFAULT_BASE_URL).replace(/\/+$/, "")}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${env.LLM_API_KEY}` },
    body: JSON.stringify(payload),
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const origin = request.headers.get("origin");
    const allowed = originAllowed(origin, env);
    const cors = allowed && origin ? corsHeaders(origin) : {};
    if (request.method === "OPTIONS") {
      return allowed ? new Response(null, { status: 204, headers: cors }) : json(403, { error: "forbidden", message: "This origin is not allowed." });
    }
    if (url.pathname === "/health" && request.method === "GET") {
      return json(200, { status: "ok", model: env.LLM_MODEL || DEFAULT_MODEL, limits: settings(env) }, cors);
    }
    if (url.pathname !== "/chat" || request.method !== "POST") {
      return json(404, { error: "not_found", message: "POST /chat" }, cors);
    }
    if (!allowed) return json(403, { error: "forbidden", message: "This origin is not allowed." });
    if (!env.LLM_API_KEY) return json(503, { error: "not_configured", message: "The proxy has no API key configured." }, cors);

    let body: ProxyRequest;
    try {
      body = validate(await request.json());
    } catch (err) {
      const e = err as ValidationError;
      return json(400, { error: "bad_request", message: e.message ?? "Invalid request." }, cors);
    }

    const lim = settings(env);
    const ip = clientAddress(request);
    try {
      await clientLimiter(env.LIMITS, ip).take(lim);
      await globalCounter(env.LIMITS).take(lim);
    } catch (err) {
      if (err instanceof LimitError) {
        return json(
          429,
          { error: err.code, message: err.message, retry_after: err.retryAfter },
          { ...cors, "retry-after": String(Math.ceil(err.retryAfter)) },
        );
      }
      throw err;
    }

    let res: Response;
    try {
      res = await upstream(env, body, Number.parseInt(env.MAX_OUTPUT_TOKENS ?? "1500", 10) || 1500);
    } catch {
      return json(503, { error: "unavailable", message: "The model could not be reached. Please try again in a moment." }, cors);
    }
    if (res.status === 429) {
      const retry = Number.parseFloat(res.headers.get("retry-after") ?? "") || 60;
      return json(429, { error: "busy", message: BUSY, retry_after: retry }, { ...cors, "retry-after": String(Math.ceil(retry)) });
    }
    if (!res.ok) {
      // the provider's error text could carry request details; only the status is passed on
      return json(502, { error: "upstream", message: `The model answered with an error (HTTP ${res.status}). Please try again.` }, cors);
    }
    return new Response(res.body, {
      status: 200,
      headers: { ...cors, "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-store", "x-accel-buffering": "no" },
    });
  },
} satisfies ExportedHandler<Env>;
