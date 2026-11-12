// Parity of the in-browser retrieval with the Python stack (fixture written by
// scripts/build_static_indexes.py): query embeddings from transformers.js vs sentence-transformers,
// and the top-5 passages of 20 queries per collection vs the same search in Python over the
// published int8 index.

import { expect, test } from "@playwright/test";
import parity from "../fixtures/parity.json" with { type: "json" };
import { waitReady } from "./helpers.ts";

const NEAR_TIE = 0.002; // reference scores closer than this may swap order

interface Row {
  query: string;
  top5: string[];
  scores: number[];
}

/** Identical, or different only where the reference itself has a near-tie. */
function equivalent(got: string[], row: Row): boolean {
  for (let i = 0; i < 5; i++) {
    if (got[i] === row.top5[i]) continue;
    const j = row.top5.indexOf(got[i] ?? "");
    const si = row.scores[i] ?? 0;
    const sj = j >= 0 ? (row.scores[j] ?? 0) : (row.scores[5] ?? -1); // not in the top 5: the 6th
    if (Math.abs(si - sj) >= NEAR_TIE) return false;
  }
  return true;
}

test.describe.configure({ mode: "serial" });

test("query embeddings match sentence-transformers (cosine >= 0.99 on 20 queries)", async ({ page }) => {
  await page.goto("./");
  await waitReady(page);
  const queries = parity.general.map((g) => g.query);
  expect(queries).toHaveLength(20);
  const vecs: number[][] = await page.evaluate(async (q) => {
    const hook = (window as unknown as { __ragsysTest: { embedQueries(q: string[]): Promise<number[][]> } }).__ragsysTest;
    return hook.embedQueries(q);
  }, queries);
  const cos = vecs.map((v, i) => v.reduce((s, x, d) => s + x * (parity.general[i]?.embedding[d] ?? 0), 0));
  console.log(`embedding cosine: min ${Math.min(...cos).toFixed(5)}, mean ${(cos.reduce((a, b) => a + b, 0) / cos.length).toFixed(5)}`);
  for (const c of cos) expect(c).toBeGreaterThanOrEqual(0.99);
});

for (const [collection, rows] of Object.entries(parity.collections as Record<string, Row[]>)) {
  test(`top-5 search results match Python for 20 queries: ${collection}`, async ({ page }) => {
    await page.goto("./");
    await waitReady(page);
    expect(rows).toHaveLength(20);
    const got: { id: string; score: number }[][] = await page.evaluate(
      async ({ id, q }) => {
        const hook = (window as unknown as { __ragsysTest: { search(id: string, q: string[], k: number): Promise<{ id: string; score: number }[][]> } }).__ragsysTest;
        return hook.search(id, q, 5);
      },
      { id: collection, q: rows.map((r) => r.query) },
    );
    let strict = 0;
    const failures: string[] = [];
    rows.forEach((row, i) => {
      const ids = (got[i] ?? []).map((h) => h.id);
      if (ids.join() === row.top5.join()) strict += 1;
      else if (!equivalent(ids, row)) failures.push(`${row.query}: browser ${ids.join(",")} python ${row.top5.join(",")}`);
    });
    console.log(`${collection}: top-5 identical ${strict}/20, identical up to near-ties ${20 - failures.length}/20`);
    expect(failures, failures.join("\n")).toEqual([]);
  });
}
