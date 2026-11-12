// Vector search in the browser. A collection's vectors.bin holds int8 values with one float32 scale
// per vector (v ~= scale * q); the score of a query x is scale * (q . x), the same arithmetic as
// scripts/build_static_indexes.py's search_int8. Uploaded documents keep float32 vectors.

import type { Chunk, Hit } from "./types.ts";

const MAGIC = "RAGV";

export interface VectorIndex {
  readonly size: number;
  readonly dim: number;
  /** Score of row ``i`` for the (normalised) query ``x``. */
  score(i: number, x: Float32Array): number;
}

export class Int8Index implements VectorIndex {
  readonly size: number;
  readonly dim: number;
  private readonly scales: Float32Array;
  private readonly values: Int8Array;

  constructor(buffer: ArrayBuffer) {
    const view = new DataView(buffer);
    const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (magic !== MAGIC || view.getUint8(4) !== 1) throw new Error("not a vectors.bin file (version 1)");
    this.dim = view.getUint32(8, true);
    this.size = view.getUint32(12, true);
    this.scales = new Float32Array(buffer, 16, this.size);
    this.values = new Int8Array(buffer, 16 + 4 * this.size, this.size * this.dim);
  }

  score(i: number, x: Float32Array): number {
    const off = i * this.dim;
    let dot = 0;
    for (let d = 0; d < this.dim; d++) dot += (this.values[off + d] ?? 0) * (x[d] ?? 0);
    return dot * (this.scales[i] ?? 0);
  }
}

export class Float32Index implements VectorIndex {
  readonly size: number;
  readonly dim: number;

  constructor(private readonly rows: Float32Array[]) {
    this.size = rows.length;
    this.dim = rows[0]?.length ?? 0;
  }

  score(i: number, x: Float32Array): number {
    const row = this.rows[i];
    if (!row) return 0;
    let dot = 0;
    for (let d = 0; d < row.length; d++) dot += (row[d] ?? 0) * (x[d] ?? 0);
    return dot;
  }
}

/** Top ``k`` rows by the best score over the query vectors (ties: earlier row first). */
export function topK(index: VectorIndex, queries: readonly Float32Array[], k: number): { row: number; score: number }[] {
  const n = index.size;
  const best = new Float32Array(n).fill(Number.NEGATIVE_INFINITY);
  for (const q of queries) {
    for (let i = 0; i < n; i++) {
      const s = index.score(i, q);
      if (s > (best[i] ?? Number.NEGATIVE_INFINITY)) best[i] = s;
    }
  }
  const order = Array.from({ length: n }, (_, i) => i);
  // partial selection would be faster; n is at most a few thousand rows per collection
  order.sort((a, b) => (best[b] ?? 0) - (best[a] ?? 0) || a - b);
  return order.slice(0, Math.min(k, n)).map((row) => ({ row, score: best[row] ?? 0 }));
}

export class SearchSource {
  constructor(
    readonly id: string,
    readonly chunks: readonly Chunk[],
    readonly index: VectorIndex,
    readonly threshold: number,
  ) {
    if (chunks.length !== index.size) throw new Error(`${id}: ${chunks.length} chunks but ${index.size} vectors`);
  }

  search(queries: readonly Float32Array[], k: number): Hit[] {
    return topK(this.index, queries, k).map(({ row, score }) => ({
      chunk: this.chunks[row] as Chunk,
      score,
      source: this.id,
    }));
  }

  get(chunkId: string): Chunk | undefined {
    return this.chunks.find((c) => c.chunk_id === chunkId);
  }
}

export function normalise(v: Float32Array): Float32Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n);
  return n > 0 ? v.map((x) => x / n) : v;
}
