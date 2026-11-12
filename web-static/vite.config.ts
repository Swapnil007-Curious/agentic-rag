import { readFileSync } from "node:fs";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";
import { assertWorkerUrl, describeWorkerUrl, pick, resolveWorkerUrl, workerOrigin, type SiteFile } from "./site-config.ts";

const site = JSON.parse(readFileSync(new URL("./site.config.json", import.meta.url), "utf8")) as SiteFile;

/**
 * Content-Security-Policy as a meta tag (GitHub Pages cannot send headers): every request stays
 * on this origin except calls to the LLM proxy; 'wasm-unsafe-eval' allows WebAssembly only.
 */
export function contentSecurityPolicy(origin: string): string {
  return [
    "default-src 'self'",
    "script-src 'self' 'wasm-unsafe-eval'",
    "style-src 'self'",
    "img-src 'self' data: blob:",
    "font-src 'self'",
    `connect-src 'self'${origin ? ` ${origin}` : ""}`,
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'none'",
  ].join("; ");
}

function csp(origin: string): Plugin {
  return {
    name: "agentic-rag-csp",
    apply: "build",
    transformIndexHtml() {
      return [
        {
          tag: "meta",
          attrs: { "http-equiv": "Content-Security-Policy", content: contentSecurityPolicy(origin) },
          injectTo: "head-prepend",
        },
      ];
    },
  };
}

export default defineConfig(({ command }) => {
  // GitHub Pages serves the site under /agentic-rag/: the workflow sets BASE_PATH=/agentic-rag/.
  const base = process.env.BASE_PATH ?? "/";
  // The LLM proxy's URL: VITE_WORKER_URL when non-blank (the Pages workflow passes the repository
  // variable WORKER_URL, which is "" when unset), else site.config.json. Empty: the site loads, but
  // sending a message reports that it is not set up; with REQUIRE_WORKER_URL=true the build fails.
  const worker = resolveWorkerUrl(process.env.VITE_WORKER_URL, site.workerUrl);
  const modelDtype = pick(process.env.MODEL_DTYPE, site.modelDtype).value || "fp16";
  if (command === "build") {
    console.log(describeWorkerUrl(worker));
    if (process.env.REQUIRE_WORKER_URL === "true") assertWorkerUrl(worker);
  }
  return {
    base,
    define: {
      __WORKER_URL__: JSON.stringify(worker.value),
      __MODEL_DTYPE__: JSON.stringify(modelDtype),
    },
    plugins: [csp(workerOrigin(worker.value))],
    build: {
      target: "es2022",
      outDir: process.env.OUT_DIR ?? "dist",
      chunkSizeWarningLimit: 2500,
    },
    worker: { format: "es" },
    // onnxruntime-web offers an export condition that selects its small entry point and loads the
    // WASM from env.wasmPaths (the self-hosted ort/ files) instead of bundling a 27 MB build
    resolve: { conditions: ["onnxruntime-web-use-extern-wasm"] },
    // pre-bundling would move transformers.js away from its import.meta.url-relative files
    optimizeDeps: { exclude: ["@huggingface/transformers"] },
    // the page reuses the Python app's stylesheet (../src/ragsys/app/static/chat.css)
    server: { fs: { allow: [".."] } },
    preview: { port: 4174, strictPort: true },
    test: {
      include: ["tests/unit/**/*.test.ts"],
      environment: "node",
    },
  };
});
