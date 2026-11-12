import { cloudflareTest } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

// Tests run inside the Workers runtime (workerd via Miniflare) with the real Durable Object
// binding; the upstream provider is mocked by spying on fetch, so nothing leaves the machine and
// no real key is needed.
export default defineConfig({
  plugins: [
    cloudflareTest({
      wrangler: { configPath: "./wrangler.jsonc" },
      miniflare: { bindings: { LLM_API_KEY: "test-secret-key-0123456789" } },
    }),
  ],
  test: {
    include: ["test/**/*.test.ts"],
  },
});
