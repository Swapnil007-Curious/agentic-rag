// Prints the size of the built site (dist/, or OUT_DIR): total, per area and per collection, raw
// and gzip (GitHub Pages compresses text responses; the model and WASM are cached by the browser
// after the first visit).

import { readFile, readdir, stat } from "node:fs/promises";
import { join, relative, resolve, sep } from "node:path";
import { gzipSync } from "node:zlib";

const out = resolve(process.env.OUT_DIR ?? "dist");

async function files(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const nested = await Promise.all(entries.map((e) => (e.isDirectory() ? files(join(dir, e.name)) : [join(dir, e.name)])));
  return nested.flat();
}

function area(rel) {
  const parts = rel.split("/");
  if (parts[0] === "data" && parts.length > 2) return `data/${parts[1]} (collection)`;
  if (parts[0] === "models") return "models (bge-small, ONNX)";
  return parts.length > 1 ? parts[0] : "(root)";
}

const groups = new Map();
let total = 0;
let totalGz = 0;
for (const path of await files(out)) {
  const rel = relative(out, path).split(sep).join("/");
  const bytes = (await stat(path)).size;
  const gz = /\.(onnx|wasm|woff2|png)$/.test(rel) ? bytes : gzipSync(await readFile(path), { level: 9 }).length;
  const g = groups.get(area(rel)) ?? { n: 0, bytes: 0, gz: 0 };
  g.n += 1;
  g.bytes += bytes;
  g.gz += gz;
  groups.set(area(rel), g);
  total += bytes;
  totalGz += gz;
}
const mb = (n) => `${(n / 1e6).toFixed(2).padStart(7)} MB`;
console.log(`\nsite size (${relative(process.cwd(), out) || "."}):`);
console.log(`  ${"area".padEnd(34)} ${"files".padStart(5)} ${"raw".padStart(10)} ${"gzip".padStart(10)}`);
for (const [name, g] of [...groups].sort((a, b) => a[0].localeCompare(b[0]))) {
  console.log(`  ${name.padEnd(34)} ${String(g.n).padStart(5)} ${mb(g.bytes)} ${mb(g.gz)}`);
}
console.log(`  ${"total".padEnd(34)} ${"".padStart(5)} ${mb(total)} ${mb(totalGz)}`);
console.log("  first visit downloads the app, the model and ONNX Runtime, then only the selected collection");
