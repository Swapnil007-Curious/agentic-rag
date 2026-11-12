// Main-thread client of the embedding worker: one worker for the whole page; queries get the bge
// query instruction, document chunks (uploads) do not.

import lock from "../assets.lock.json" with { type: "json" };
import type { ModelFile, WorkerRequest, WorkerResponse } from "./embed.worker.ts";

export type ProgressListener = (loaded: number, total: number, cached: boolean) => void;

const MODEL_FILES: Record<string, string> = { fp16: "onnx/model_fp16.onnx", q8: "onnx/model_quantized.onnx" };

/** The files the pipeline reads (sizes from assets.lock.json, the file the build verified). */
export function modelFiles(dtype: string): ModelFile[] {
  const files = lock.model.files as Record<string, { bytes: number }>;
  const weights = MODEL_FILES[dtype] ?? "onnx/model_fp16.onnx";
  return ["config.json", "tokenizer_config.json", "tokenizer.json", weights].map((path) => ({
    path,
    bytes: files[path]?.bytes ?? 0,
  }));
}

export class Embedder {
  private readonly worker: Worker;
  private nextId = 1;
  private readonly pending = new Map<number, { resolve: (v: Float32Array[]) => void; reject: (e: Error) => void }>();
  private readonly progress = new Set<ProgressListener>();
  readonly ready: Promise<number>;

  constructor(
    base: string,
    dtype: string,
    private readonly instruction: string,
  ) {
    this.worker = new Worker(new URL("./embed.worker.ts", import.meta.url), { type: "module" });
    let resolveReady: (ms: number) => void = () => {};
    let rejectReady: (e: Error) => void = () => {};
    this.ready = new Promise<number>((res, rej) => {
      resolveReady = res;
      rejectReady = rej;
    });
    this.ready.catch(() => undefined); // reported through the callers
    this.worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      const msg = ev.data;
      if (msg.type === "progress") for (const fn of this.progress) fn(msg.loaded, msg.total, msg.cached);
      else if (msg.type === "ready") resolveReady(msg.ms);
      else if (msg.type === "result") {
        this.pending.get(msg.id)?.resolve(msg.vectors);
        this.pending.delete(msg.id);
      } else if (msg.type === "error") {
        const err = new Error(msg.message);
        if (msg.id === null) {
          rejectReady(err);
          for (const p of this.pending.values()) p.reject(err);
          this.pending.clear();
        } else {
          this.pending.get(msg.id)?.reject(err);
          this.pending.delete(msg.id);
        }
      }
    };
    const init: WorkerRequest = { type: "init", base, dtype, model: lock.model.dir, files: modelFiles(dtype) };
    this.worker.postMessage(init);
  }

  onProgress(fn: ProgressListener): () => void {
    this.progress.add(fn);
    return () => this.progress.delete(fn);
  }

  private run(texts: string[]): Promise<Float32Array[]> {
    if (!texts.length) return Promise.resolve([]);
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "embed", id, texts } satisfies WorkerRequest);
    });
  }

  /** Query vectors (with the retrieval instruction, as the Python index was queried). */
  queries(texts: string[]): Promise<Float32Array[]> {
    return this.run(texts.map((t) => this.instruction + t));
  }

  /** Passage vectors in batches (no instruction), reporting progress. */
  async passages(texts: string[], onBatch?: (done: number, total: number) => void, batch = 16): Promise<Float32Array[]> {
    const out: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += batch) {
      out.push(...(await this.run(texts.slice(i, i + batch))));
      onBatch?.(Math.min(i + batch, texts.length), texts.length);
    }
    return out;
  }
}
