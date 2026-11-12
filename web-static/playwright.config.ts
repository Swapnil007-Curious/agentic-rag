import { defineConfig, devices } from "@playwright/test";

// Tests run against `vite preview` of a build with a test worker URL (the LLM proxy is mocked with
// page.route); BASE_PATH must match the build.
const base = "/agentic-rag/";
const port = 4174;
const origin = `http://localhost:${port}`;
export const TEST_WORKER_URL = "https://proxy.test.invalid";

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 240_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: { baseURL: `${origin}${base}`, trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npx vite build --logLevel warn && npx vite preview --port ${port} --strictPort`,
    env: { BASE_PATH: base, VITE_WORKER_URL: TEST_WORKER_URL, OUT_DIR: "dist-test" },
    url: `${origin}${base}`,
    reuseExistingServer: !process.env.CI,
    timeout: 300_000,
  },
});
