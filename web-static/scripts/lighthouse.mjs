// Lighthouse (mobile preset) against the production preview. Builds dist/ first unless
// SKIP_BUILD=1, serves it with vite preview and writes the scores to docs/lighthouse.json plus a
// Markdown table on stdout. Uses Playwright's Chromium. For experiments: LH_BLOCK=<patterns>
// blocks URLs (comma separated) and skips writing the JSON; LH_DEBUG=1 prints the LCP element,
// the main-thread breakdown and the long tasks.

import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import lighthouse from "lighthouse";
import { launch } from "chrome-launcher";
import { chromium } from "playwright";

const PORT = 4176;
const root = process.cwd();

if (!process.env.SKIP_BUILD) {
  const r = spawnSync("npx", ["vite", "build", "--logLevel", "warn"], { cwd: root, shell: true, stdio: "inherit" });
  if (r.status) process.exit(r.status);
}
const server = spawn("npx", ["vite", "preview", "--port", String(PORT), "--strictPort"], { cwd: root, shell: true, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 2500));

const chrome = await launch({ chromePath: chromium.executablePath(), chromeFlags: ["--headless=new", "--no-sandbox"] });
try {
  const result = await lighthouse(`http://localhost:${PORT}/`, {
    port: chrome.port,
    output: "json",
    logLevel: "error",
    onlyCategories: ["performance", "accessibility", "best-practices", "seo"],
    formFactor: "mobile",
    screenEmulation: { mobile: true, width: 412, height: 823, deviceScaleFactor: 1.75, disabled: false },
    throttlingMethod: "simulate",
    ...(process.env.LH_BLOCK ? { blockedUrlPatterns: process.env.LH_BLOCK.split(",") } : {}),
  });
  const cats = result.lhr.categories;
  const rows = Object.entries(cats).map(([id, c]) => [c.title, Math.round((c.score ?? 0) * 100), id]);
  const audits = result.lhr.audits;
  const metrics = ["first-contentful-paint", "largest-contentful-paint", "total-blocking-time", "cumulative-layout-shift", "speed-index"].map((id) => [audits[id]?.title ?? id, audits[id]?.displayValue ?? "n/a"]);
  mkdirSync(join(root, "..", "docs"), { recursive: true });
  if (!process.env.LH_BLOCK) writeFileSync(join(root, "..", "docs", "lighthouse.json"), JSON.stringify({ fetchTime: result.lhr.fetchTime, lighthouseVersion: result.lhr.lighthouseVersion, scores: Object.fromEntries(rows.map(([t, s]) => [t, s])), metrics: Object.fromEntries(metrics) }, null, 2));
  console.log("| category | score |\n|---|---|");
  for (const [title, score] of rows) console.log(`| ${title} | ${score} |`);
  console.log("\n| metric | value |\n|---|---|");
  for (const [title, value] of metrics) console.log(`| ${title} | ${value} |`);
  if (process.env.LH_DEBUG) {
    const lcp = audits["largest-contentful-paint-element"]?.details?.items ?? [];
    console.log("lcp element:", JSON.stringify(lcp).slice(0, 900));
    console.log("lcp breakdown:", JSON.stringify(audits["lcp-breakdown-insight"]?.details ?? null).slice(0, 900));
    for (const id of ["bootup-time", "mainthread-work-breakdown", "long-tasks", "forced-reflow-insight"]) {
      console.log(`${id}:`);
      for (const it of audits[id]?.details?.items ?? []) console.log("  ", JSON.stringify(it).slice(0, 300));
    }
  }
  const failing = Object.values(audits).filter((a) => a.score !== null && a.score < 0.9 && a.scoreDisplayMode !== "informative" && a.scoreDisplayMode !== "notApplicable");
  if (failing.length) {
    console.log("\naudits below 0.9:");
    for (const a of failing) console.log(`  - ${a.id}: ${a.title} (${a.displayValue ?? ""})`);
  }
} finally {
  await chrome.kill();
  server.kill();
}
process.exit(0);
