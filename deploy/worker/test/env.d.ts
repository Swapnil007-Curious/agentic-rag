// Bindings as the tests see them (cloudflare:test's env).
import type { RateLimiter } from "../src/limits.ts";

declare global {
  namespace Cloudflare {
    interface Env {
      LLM_API_KEY: string;
      LIMITS: DurableObjectNamespace<RateLimiter>;
    }
  }
}

export {};
