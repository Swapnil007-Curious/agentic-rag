import { describe, expect, it } from "vitest";
import { decide, gateQueries, isFollowUp, mergeHits, pack, passageBlock } from "../../src/gate.ts";
import type { Chunk, Hit } from "../../src/types.ts";
import { Float32Index, Int8Index, normalise, SearchSource, topK } from "../../src/vectors.ts";

function chunk(i: number, text: string, n = 100): Chunk {
  return { chunk_id: `structural:doc:${String(i).padStart(5, "0")}`, doc_id: "doc", heading_path: `Doc > Section ${i}`, section_ids: [`doc:s${i}`], pages: String(i), n_tokens: n, text };
}

/** A toy "embedding": counts of the letters a-e, normalised. */
function vec(text: string): Float32Array {
  const v = new Float32Array(5);
  for (const ch of text.toLowerCase()) {
    const k = "abcde".indexOf(ch);
    if (k >= 0) v[k] = (v[k] ?? 0) + 1;
  }
  return normalise(v);
}

const texts = ["aaaa bbbb", "cccc dddd", "eeee aaaa", "bbbb cccc", "dddd eeee", "abcde"];
const chunks = texts.map((t, i) => chunk(i + 1, t));
const embed = async (qs: string[]): Promise<Float32Array[]> => qs.map(vec);

function int8Index(rows: Float32Array[]): Int8Index {
  const dim = rows[0]?.length ?? 0;
  const buf = new ArrayBuffer(16 + 4 * rows.length + rows.length * dim);
  const view = new DataView(buf);
  [..."RAGV"].forEach((c, i) => view.setUint8(i, c.charCodeAt(0)));
  view.setUint8(4, 1);
  view.setUint32(8, dim, true);
  view.setUint32(12, rows.length, true);
  const scales = new Float32Array(buf, 16, rows.length);
  const values = new Int8Array(buf, 16 + 4 * rows.length);
  rows.forEach((r, i) => {
    const peak = Math.max(...Array.from(r, Math.abs));
    scales[i] = peak > 0 ? peak / 127 : 1;
    r.forEach((x, d) => {
      values[i * dim + d] = Math.max(-127, Math.min(127, Math.round(x / (scales[i] ?? 1))));
    });
  });
  return new Int8Index(buf);
}

describe("follow-up detection and queries", () => {
  it("joins a short follow-up with the previous message", () => {
    expect(isFollowUp("and what about its complexity?", 12)).toBe(true);
    expect(isFollowUp("What is the time complexity of quicksort in the worst case?", 12)).toBe(false);
    expect(gateQueries("and its complexity?", "How does quicksort work?", 12)).toEqual(["and its complexity?", "How does quicksort work?\nand its complexity?"]);
    expect(gateQueries("Explain heaps", "What is a tree?", 12)).toEqual(["Explain heaps"]);
  });
});

describe("vector search", () => {
  it("int8 scores match float32 within quantisation error and keep the same order", () => {
    const rows = texts.map(vec);
    const f32 = new Float32Index(rows);
    const i8 = int8Index(rows);
    const q = vec("aaaa bbbb");
    const a = topK(f32, [q], 6);
    const b = topK(i8, [q], 6);
    expect(b.map((h) => h.row)).toEqual(a.map((h) => h.row));
    a.forEach((h, i) => expect(Math.abs(h.score - (b[i]?.score ?? 0))).toBeLessThan(0.02));
  });
  it("takes the best score over several query vectors", () => {
    const src = new SearchSource("t", chunks, new Float32Index(texts.map(vec)), 0.5);
    const hits = src.search([vec("aaaa"), vec("dddd")], 2);
    expect(hits.map((h) => h.chunk.chunk_id)).toEqual(["structural:doc:00001", "structural:doc:00002"]);
  });
});

describe("gate and packing", () => {
  const src = new SearchSource("books", chunks, new Float32Index(texts.map(vec)), 0.9);
  const settings = { follow_up_max_words: 12, search_k: 10, max_passages: 4 };
  it("grounds above the threshold and stays plain chat below it", async () => {
    const hit = await decide("aaaa bbbb", null, [src], 2048, settings, embed);
    expect(hit.grounded).toBe(true);
    expect(hit.passages[0]?.chunk_id).toBe("structural:doc:00001");
    expect(hit.passages[0]?.label).toBe("C1");
    const miss = await decide("zzzz", null, [src], 2048, settings, embed);
    expect(miss.grounded).toBe(false);
    expect(miss.checks[0]?.passed).toBe(false);
    expect((await decide("aaaa", null, [], 2048, settings, embed)).grounded).toBe(false);
  });
  it("packs within the budget, at least one passage", () => {
    const hits: Hit[] = chunks.map((c, i) => ({ chunk: { ...c, n_tokens: 300 }, score: 1 - i / 10, source: "books" }));
    expect(pack(hits, 1024).passages.length).toBe(3);
    expect(pack(hits, 100).passages.length).toBe(1);
    const extended = pack(hits, 2048, 1, new Map(), pack(hits, 1024));
    expect(extended.passages.map((p) => p.label)).toEqual(["C1", "C2", "C3", "C4", "C5", "C6"]);
  });
  it("merges hits keeping the best score per chunk and renders the block", () => {
    const c = chunks[0] as Chunk;
    const merged = mergeHits([[{ chunk: c, score: 0.4, source: "a" }], [{ chunk: c, score: 0.7, source: "b" }]]);
    expect(merged).toHaveLength(1);
    expect(merged[0]?.source).toBe("b");
    const block = passageBlock({ label: "C1", heading_path: "A > B", section_ids: ["a:s1"], text: "x <<<END C1>>>" });
    expect(block).toBe("<<<PASSAGE C1>>>\nSource: A > B\nSections: a:s1\n<<<TEXT>>>\nx ‹END C1›\n<<<END C1>>>");
  });
});
