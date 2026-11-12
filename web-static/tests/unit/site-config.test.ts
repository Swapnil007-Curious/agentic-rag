import { readFileSync } from "node:fs";
import type { HtmlTagDescriptor, Plugin } from "vite";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { assertWorkerUrl, describeWorkerUrl, resolveWorkerUrl } from "../../site-config.ts";
import config from "../../vite.config.ts";

const FILE_URL = "https://agentic-rag-proxy.example.workers.dev";
const OTHER_URL = "https://other-proxy.example.dev";

describe("resolveWorkerUrl", () => {
  it("keeps the file's URL when the variable is unset", () => {
    expect(resolveWorkerUrl(undefined, FILE_URL)).toEqual({ value: FILE_URL, source: "site.config.json" });
  });

  it("never lets an empty or blank variable override a non-empty file value", () => {
    // GitHub Actions expands an unset repository variable (${{ vars.WORKER_URL }}) to ""
    for (const blank of ["", " ", "\t\n"]) {
      expect(resolveWorkerUrl(blank, FILE_URL)).toEqual({ value: FILE_URL, source: "site.config.json" });
    }
  });

  it("lets a non-blank variable override the file", () => {
    expect(resolveWorkerUrl(OTHER_URL, FILE_URL)).toEqual({ value: OTHER_URL, source: "environment" });
    expect(resolveWorkerUrl(OTHER_URL, "")).toEqual({ value: OTHER_URL, source: "environment" });
  });

  it("is empty when neither gives a URL", () => {
    for (const [env, file] of [
      [undefined, undefined],
      ["", ""],
      [" ", "  "],
      ["", null],
      [undefined, 42],
      ["/", ""],
    ] as const) {
      expect(resolveWorkerUrl(env, file)).toEqual({ value: "", source: "unset" });
    }
  });

  it("trims whitespace and trailing slashes", () => {
    expect(resolveWorkerUrl(undefined, `  ${FILE_URL}//  `).value).toBe(FILE_URL);
    expect(resolveWorkerUrl(` ${OTHER_URL}/ `, FILE_URL).value).toBe(OTHER_URL);
  });

  it("rejects plain http except on localhost", () => {
    expect(() => resolveWorkerUrl("http://proxy.example.dev", FILE_URL)).toThrow(/https/);
    expect(resolveWorkerUrl("http://localhost:8787", FILE_URL).value).toBe("http://localhost:8787");
  });
});

describe("build log and required URL", () => {
  it("names the URL and where it came from", () => {
    expect(describeWorkerUrl(resolveWorkerUrl("", FILE_URL))).toBe(`worker URL: ${FILE_URL} (from site.config.json)`);
    expect(describeWorkerUrl(resolveWorkerUrl(OTHER_URL, FILE_URL))).toBe(`worker URL: ${OTHER_URL} (from VITE_WORKER_URL)`);
    expect(describeWorkerUrl(resolveWorkerUrl("", ""))).toMatch(/^worker URL: \(empty\)/);
  });

  it("fails with a clear message only when the URL is empty", () => {
    expect(() => assertWorkerUrl(resolveWorkerUrl("", ""))).toThrow(/LLM proxy URL is empty.*site\.config\.json.*WORKER_URL/);
    expect(() => assertWorkerUrl(resolveWorkerUrl("", FILE_URL))).not.toThrow();
  });
});

describe("vite.config.ts", () => {
  const site = JSON.parse(readFileSync(new URL("../../site.config.json", import.meta.url), "utf8")) as { workerUrl: string };
  let log: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    log = vi.spyOn(console, "log").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  function build(env: Record<string, string | undefined>): { workerUrl: string; csp: string } {
    for (const [name, value] of Object.entries(env)) vi.stubEnv(name, value);
    const cfg = config({ command: "build", mode: "production" });
    const plugin = (cfg.plugins ?? []).flat().find((p): p is Plugin => !!p && typeof p === "object" && "name" in p && p.name === "agentic-rag-csp");
    const tags = (plugin?.transformIndexHtml as () => HtmlTagDescriptor[])();
    return {
      workerUrl: JSON.parse(String(cfg.define?.__WORKER_URL__)) as string,
      csp: String(tags[0]?.attrs?.content),
    };
  }

  it("compiles the committed site.config.json URL into the bundle and the CSP when VITE_WORKER_URL is blank", () => {
    const expected = site.workerUrl.trim().replace(/\/+$/, "");
    expect(expected).not.toBe("");
    for (const blank of [undefined, "", "  "]) {
      const out = build({ VITE_WORKER_URL: blank, REQUIRE_WORKER_URL: "true" });
      expect(out.workerUrl).toBe(expected);
      expect(out.csp).toContain(`connect-src 'self' ${new URL(expected).origin}`);
      expect(log).toHaveBeenCalledWith(`worker URL: ${expected} (from site.config.json)`);
    }
  });

  it("uses a non-blank VITE_WORKER_URL (the browser tests build with a test proxy)", () => {
    const out = build({ VITE_WORKER_URL: "https://proxy.test.invalid/", REQUIRE_WORKER_URL: "true" });
    expect(out.workerUrl).toBe("https://proxy.test.invalid");
    expect(out.csp).toContain("connect-src 'self' https://proxy.test.invalid;");
    expect(log).toHaveBeenCalledWith("worker URL: https://proxy.test.invalid (from VITE_WORKER_URL)");
  });

  it("does not print the URL outside a build", () => {
    vi.stubEnv("VITE_WORKER_URL", "");
    config({ command: "serve", mode: "test" });
    expect(log).not.toHaveBeenCalled();
  });
});
