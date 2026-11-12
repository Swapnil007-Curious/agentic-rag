// Rate limits in Durable Objects (one per client address, one global).
//
// Decision (Cloudflare docs read on 2026-11-13):
// * The Workers Rate Limiting binding only accepts a 10 s or 60 s period, counts per location
//   and is eventually consistent, so it cannot express "20 per 10 minutes" or "200 per day".
// * Workers KV on the Free plan allows 1,000 writes per day in total and one write per second
//   per key, and is eventually consistent: counting requests in KV would exhaust the write
//   quota and still over-count.
// * Durable Objects are on the Free plan (SQLite backend only: 100,000 requests and 100,000
//   row writes per day), and one object per key gives a single, strictly ordered counter.
// Each limiter keeps a sliding window of request times in its storage: two windows for a client
// (10 minutes and a day), one daily window for the global cap. Only timestamps are stored; no
// message content, no address beyond the object's own name.

import { DurableObject } from "cloudflare:workers";

export interface LimitSettings {
  perWindow: number;
  windowSeconds: number;
  perDay: number;
  globalPerDay: number;
}

export class LimitError extends Error {
  constructor(
    readonly code: "rate_limited" | "daily_cap",
    message: string,
    readonly retryAfter: number,
  ) {
    super(message);
  }
}

const DAY = 86_400;

function minutes(seconds: number): string {
  const m = Math.max(1, Math.round(seconds / 60));
  return m === 1 ? "about a minute" : `about ${m} minutes`;
}

interface Window {
  max: number;
  seconds: number;
}

/** Sliding windows over stored request times; returns the wait when a window is full. */
export function check(times: number[], now: number, windows: Window[]): { wait: number; kept: number[] } {
  const horizon = Math.max(...windows.map((w) => w.seconds));
  const kept = times.filter((t) => now - t < horizon);
  let wait = 0;
  for (const w of windows) {
    const recent = kept.filter((t) => now - t < w.seconds);
    if (recent.length >= w.max) {
      const oldest = recent[recent.length - w.max] ?? now;
      wait = Math.max(wait, w.seconds - (now - oldest));
    }
  }
  return { wait, kept };
}

export class RateLimiter extends DurableObject {
  /** Records one request for a client, or returns the seconds to wait. */
  async takeClient(settings: LimitSettings): Promise<{ ok: true } | { ok: false; code: "rate_limited"; wait: number }> {
    const now = Date.now() / 1000;
    const times = (await this.ctx.storage.get<number[]>("t")) ?? [];
    const { wait, kept } = check(times, now, [
      { max: settings.perWindow, seconds: settings.windowSeconds },
      { max: settings.perDay, seconds: DAY },
    ]);
    if (wait > 0) {
      if (kept.length !== times.length) await this.ctx.storage.put("t", kept);
      return { ok: false, code: "rate_limited", wait };
    }
    kept.push(now);
    await this.ctx.storage.put("t", kept);
    await this.ctx.storage.setAlarm((now + DAY) * 1000);
    return { ok: true };
  }

  /** The global daily cap (one counter for everyone, reset at midnight UTC). */
  async takeGlobal(settings: LimitSettings): Promise<{ ok: true } | { ok: false; code: "daily_cap"; wait: number }> {
    const now = Date.now();
    const day = Math.floor(now / (DAY * 1000));
    const state = (await this.ctx.storage.get<{ day: number; n: number }>("g")) ?? { day, n: 0 };
    if (state.day !== day) {
      state.day = day;
      state.n = 0;
    }
    if (state.n >= settings.globalPerDay) {
      return { ok: false, code: "daily_cap", wait: ((day + 1) * DAY * 1000 - now) / 1000 };
    }
    state.n += 1;
    await this.ctx.storage.put("g", state);
    return { ok: true };
  }

  /** Housekeeping: drop the stored times once they are all older than a day. */
  override async alarm(): Promise<void> {
    const now = Date.now() / 1000;
    const times = (await this.ctx.storage.get<number[]>("t")) ?? [];
    const kept = times.filter((t) => now - t < DAY);
    if (kept.length) await this.ctx.storage.put("t", kept);
    else await this.ctx.storage.deleteAll();
  }
}

export function clientLimiter(ns: DurableObjectNamespace, ip: string): { take(s: LimitSettings): Promise<void> } {
  const stub = ns.get(ns.idFromName(`ip:${ip}`)) as unknown as RateLimiter;
  return {
    async take(s) {
      const r = await stub.takeClient(s);
      if (!r.ok) throw new LimitError("rate_limited", `You have sent many messages in a short time. Please try again in ${minutes(r.wait)}.`, r.wait);
    },
  };
}

export function globalCounter(ns: DurableObjectNamespace): { take(s: LimitSettings): Promise<void> } {
  const stub = ns.get(ns.idFromName("global")) as unknown as RateLimiter;
  return {
    async take(s) {
      const r = await stub.takeGlobal(s);
      if (!r.ok) throw new LimitError("daily_cap", "The assistant is very busy today and has reached its daily limit. Please try again tomorrow.", r.wait);
    },
  };
}
