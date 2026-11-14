// Brand assets for Veridex, all generated from one seal definition (no external assets):
//   public/favicon.svg                 the seal mark (reads at 16 px)
//   public/icons/icon-192.png, icon-512.png, apple-touch-icon.png   rendered from the SVG with sharp
//   public/og-image.png                1200x630: wordmark, tagline and seal, rendered with the site's
//                                      own fonts through Playwright's Chromium
// Run: node scripts/brand.mjs  (the PNGs are committed; rerun after changing the mark).

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import sharp from "sharp";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pub = join(root, "public");

const LAPIS = "#0B1230";
const BRASS = "#D9A441";

/** The seal: a thin circular ring and a V whose right stroke runs on into a check tick. */
export function sealSvg({ size = 64, background = true, ring = 2.5, stroke = 6 } = {}) {
  const disc = background ? `<circle cx="32" cy="32" r="32" fill="${LAPIS}"/>` : "";
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="${size}" height="${size}" role="img" aria-label="Veridex">` +
    disc +
    `<circle cx="32" cy="32" r="27.5" fill="none" stroke="${BRASS}" stroke-width="${ring}"/>` +
    `<path d="M17 23 L28 44 L49 17" fill="none" stroke="${BRASS}" stroke-width="${stroke}" stroke-linecap="round" stroke-linejoin="round"/>` +
    `</svg>`
  );
}

async function png(svg, size, file) {
  await sharp(Buffer.from(svg), { density: 384 }).resize(size, size).png({ compressionLevel: 9 }).toFile(file);
}

async function ogImage(file) {
  const { chromium } = await import("playwright");
  const fonts = {
    serif: pathToFileURL(join(root, "node_modules/@fontsource/instrument-serif/files/instrument-serif-latin-400-normal.woff2")).href,
    sans: pathToFileURL(join(root, "node_modules/@fontsource-variable/geist/files/geist-latin-wght-normal.woff2")).href,
  };
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>
    @font-face { font-family: "Instrument Serif"; src: url("${fonts.serif}") format("woff2"); }
    @font-face { font-family: "Geist"; src: url("${fonts.sans}") format("woff2-variations"); font-weight: 100 900; }
    html, body { margin: 0; }
    body { width: 1200px; height: 630px; overflow: hidden; background: ${LAPIS}; color: #E7ECFF; font-family: Geist, system-ui, sans-serif; position: relative; }
    .aurora { position: absolute; inset: 0;
      background:
        radial-gradient(60% 55% at 78% 22%, rgba(78,104,210,.42), transparent 70%),
        radial-gradient(55% 50% at 18% 85%, rgba(37,58,150,.45), transparent 70%); }
    .seal { position: absolute; left: 96px; top: 165px; width: 300px; height: 300px; }
    .text { position: absolute; left: 452px; top: 196px; }
    h1 { font: 400 148px/1.05 "Instrument Serif", serif; letter-spacing: -0.02em; margin: 0; color: #E7ECFF; }
    p { font: 500 36px/1.3 Geist, sans-serif; margin: 14px 0 0 6px; color: #B9C2E6; }
    .proof { position: absolute; left: 458px; top: 436px; font: 500 20px/1 Geist, sans-serif; letter-spacing: .14em; text-transform: uppercase; color: ${BRASS}; }
  </style></head><body>
    <div class="aurora"></div>
    <div class="seal">${sealSvg({ size: 300, background: false, ring: 1.6, stroke: 4.6 })}</div>
    <div class="text"><h1>Veridex</h1><p>Ask anything. Every answer verified.</p></div>
    <div class="proof">Hybrid-retrieval · self-verifying · in your browser</div>
  </body></html>`;
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
    await page.setContent(html, { waitUntil: "load" });
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({ path: file, type: "png" });
  } finally {
    await browser.close();
  }
}

await mkdir(join(pub, "icons"), { recursive: true });
await writeFile(join(pub, "favicon.svg"), `${sealSvg({ size: 64, ring: 3.5, stroke: 7 })}\n`);
await png(sealSvg({ size: 512, ring: 2.5, stroke: 6 }), 192, join(pub, "icons", "icon-192.png"));
await png(sealSvg({ size: 512, ring: 2.5, stroke: 6 }), 512, join(pub, "icons", "icon-512.png"));
await png(sealSvg({ size: 512, ring: 2.5, stroke: 6 }), 180, join(pub, "icons", "apple-touch-icon.png"));
await ogImage(join(pub, "og-image.png"));
await writeFile(
  join(pub, "site.webmanifest"),
  `${JSON.stringify(
    {
      name: "Veridex",
      short_name: "Veridex",
      description: "Ask anything. Every answer verified.",
      start_url: "./",
      scope: "./",
      display: "standalone",
      background_color: LAPIS,
      theme_color: LAPIS,
      icons: [
        { src: "icons/icon-192.png", sizes: "192x192", type: "image/png" },
        { src: "icons/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any maskable" },
      ],
    },
    null,
    2,
  )}\n`,
);
console.log("brand assets written to public/");
