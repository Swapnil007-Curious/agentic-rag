import { readFileSync } from "node:fs";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

// GitHub Pages serves the site under /agentic-rag/: the workflow sets BASE_PATH=/agentic-rag/.
const base = process.env.BASE_PATH ?? "/";
const site = JSON.parse(readFileSync(new URL("./site.config.json", import.meta.url), "utf8")) as {
  workerUrl: string;
  modelDtype: string;
};
// The LLM proxy's URL: VITE_WORKER_URL (e.g. a repository variable in the Pages workflow), else
// site.config.json. Empty: the site loads, but sending a message reports that it is not set up.
const workerUrl = (process.env.VITE_WORKER_URL ?? site.workerUrl ?? "").replace(/\/+$/, "");
const modelDtype = process.env.MODEL_DTYPE ?? site.modelDtype ?? "fp16";

function workerOrigin(url: string): string {
  if (!url) return "";
  const u = new URL(url);
  if (u.protocol !== "https:" && u.hostname !== "localhost" && u.hostname !== "127.0.0.1") {
    throw new Error(`the worker URL must use https: ${url}`);
  }
  return u.origin;
}

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

export default defineConfig({
  base,
  define: {
    __WORKER_URL__: JSON.stringify(workerUrl),
    __MODEL_DTYPE__: JSON.stringify(modelDtype),
  },
  plugins: [csp(workerOrigin(workerUrl))],
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
});
