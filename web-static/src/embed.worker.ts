// Web Worker: bge-small-en-v1.5 (ONNX, self-hosted under models/) with transformers.js on ONNX
// Runtime Web's WASM backend (self-hosted under ort/). CLS pooling and L2 normalisation, as the
// sentence-transformers model is configured. Nothing is fetched from another origin: remote
// models are off and the WASM paths point at this site.
//
// The model files are downloaded once, with byte progress, into the Cache API entry that
// transformers.js reads ("transformers-cache", keyed by the local path); the pipeline then loads
// them from there. (Passing a progress callback to the pipeline instead would make transformers.js
// fetch every local file twice: once for its size, once for its content.)

import { env, pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

export interface ModelFile {
  path: string; // relative to the model directory, e.g. "onnx/model_fp16.onnx"
  bytes: number;
}

export type WorkerRequest =
  | { type: "init"; base: string; dtype: string; model: string; files: ModelFile[] }
  | { type: "embed"; id: number; texts: string[] };

export type WorkerResponse =
  | { type: "progress"; loaded: number; total: number; cached: boolean }
  | { type: "ready"; ms: number }
  | { type: "result"; id: number; vectors: Float32Array[] }
  | { type: "error"; id: number | null; message: string };

const scope = self as unknown as DedicatedWorkerGlobalScope;
let extractor: Promise<FeatureExtractionPipeline> | null = null;

function post(msg: WorkerResponse, transfer: Transferable[] = []): void {
  scope.postMessage(msg, transfer);
}

async function prefetch(urls: { url: string; bytes: number }[]): Promise<void> {
  let cache: Cache;
  try {
    cache = await caches.open(env.cacheKey);
  } catch {
    return; // no Cache API (private window): transformers.js downloads the files itself
  }
  const total = urls.reduce((s, u) => s + u.bytes, 0);
  let done = 0;
  let allCached = true;
  for (const { url, bytes } of urls) {
    if (await cache.match(url)) {
      done += bytes;
      post({ type: "progress", loaded: done, total, cached: allCached });
      continue;
    }
    allCached = false;
    const res = await fetch(url);
    if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
    const reader = res.body.getReader();
    const parts: Uint8Array[] = [];
    let got = 0;
    for (;;) {
      const { done: end, value } = await reader.read();
      if (end) break;
      parts.push(value);
      got += value.byteLength;
      post({ type: "progress", loaded: done + Math.min(got, bytes), total, cached: false });
    }
    const blob = new Blob(parts as BlobPart[]);
    const headers = { "content-type": res.headers.get("content-type") ?? "application/octet-stream", "content-length": String(blob.size) };
    await cache.put(url, new Response(blob, { headers })).catch(() => undefined);
    done += bytes;
  }
  post({ type: "progress", loaded: total, total, cached: allCached });
}

async function load(msg: Extract<WorkerRequest, { type: "init" }>): Promise<FeatureExtractionPipeline> {
  const root = new URL(msg.base, scope.location.origin).href;
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  // a path, not a full URL: transformers.js only checks local files for paths
  env.localModelPath = `${msg.base}models/`;
  env.useBrowserCache = true;
  env.useWasmCache = false; // the WASM files are same-origin and cached by the browser
  const wasm = env.backends.onnx.wasm;
  if (wasm) {
    wasm.wasmPaths = { mjs: `${root}ort/ort-wasm-simd-threaded.mjs`, wasm: `${root}ort/ort-wasm-simd-threaded.wasm` };
    wasm.numThreads = 1; // GitHub Pages cannot send the headers that multi-threading needs
  }
  const t0 = performance.now();
  await prefetch(msg.files.map((f) => ({ url: `${root}models/${msg.model}/${f.path}`, bytes: f.bytes })));
  const ex = (await pipeline("feature-extraction", msg.model, {
    dtype: msg.dtype === "q8" ? "q8" : "fp16",
    device: "wasm",
  })) as FeatureExtractionPipeline;
  post({ type: "ready", ms: performance.now() - t0 });
  return ex;
}

scope.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const msg = ev.data;
  if (msg.type === "init") {
    extractor ??= load(msg);
    extractor.catch((err: unknown) => post({ type: "error", id: null, message: err instanceof Error ? err.message : String(err) }));
    return;
  }
  try {
    if (!extractor) throw new Error("the embedding model is not initialised");
    const ex = await extractor;
    const out = await ex(msg.texts, { pooling: "cls", normalize: true });
    const tensor = out.type === "float32" ? out : out.to("float32");
    const dim = tensor.dims[tensor.dims.length - 1] ?? 0;
    const data = tensor.data as Float32Array;
    const vectors = msg.texts.map((_, i) => data.slice(i * dim, (i + 1) * dim));
    post({ type: "result", id: msg.id, vectors }, vectors.map((v) => v.buffer));
  } catch (err) {
    post({ type: "error", id: msg.id, message: err instanceof Error ? err.message : String(err) });
  }
};
