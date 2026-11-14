// WCAG 2.2 contrast of the Veridex colour tokens (DESIGN.md lists the output). Pairs that carry
// text must reach 4.5:1 (AA); large text and non-text marks 3:1. Exit code 1 when a pair fails.

const THEMES = {
  dark: {
    bg: "#0B1230", surface: "#121A3D", surface2: "#1A2350",
    text: "#E7ECFF", muted: "#8C96C0", brass: "#D9A441", brassText: "#D9A441", brassInk: "#0B1230",
    supported: "#4FD1A5", partial: "#F2C14E", unsupported: "#F07A7A",
  },
  light: {
    bg: "#F5F7FC", surface: "#FFFFFF", surface2: "#EEF1F9",
    text: "#0B1230", muted: "#5A6390", brass: "#A87414", brassText: "#8B5F0B", brassInk: "#0B1230",
    supported: "#1C7A5B", partial: "#8A5A00", unsupported: "#B9353A",
  },
};

function lum(hex) {
  const n = parseInt(hex.slice(1), 16);
  const c = [16, 8, 0].map((s) => ((n >> s) & 255) / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
export function ratio(a, b) {
  const [l1, l2] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
}

const PAIRS = [
  ["text", "bg", 4.5, "body text"],
  ["text", "surface", 4.5, "text on cards"],
  ["text", "surface2", 4.5, "text in user bubble"],
  ["muted", "bg", 4.5, "muted text (hints, timings)"],
  ["muted", "surface", 4.5, "muted text on cards"],
  ["brassText", "bg", 4.5, "brass text: citation chips, labels"],
  ["brassText", "surface", 4.5, "brass text on cards"],
  ["brass", "bg", 3, "brass marks (seal, bars, focus ring): non-text 3:1"],
  ["brassInk", "brass", 4.5, "text on the primary (brass) button"],
  ["supported", "bg", 4.5, "verified label"],
  ["supported", "surface", 4.5, "verified label on cards"],
  ["partial", "bg", 4.5, "partially supported label"],
  ["partial", "surface", 4.5, "partial label on cards"],
  ["unsupported", "bg", 4.5, "not enough evidence label"],
  ["unsupported", "surface", 4.5, "abstained label on cards"],
];

let failed = false;
const rows = [];
for (const [theme, t] of Object.entries(THEMES)) {
  for (const [fg, bg, min, use] of PAIRS) {
    const r = ratio(t[fg], t[bg]);
    const ok = r >= min;
    if (!ok) failed = true;
    rows.push(`| ${theme} | ${use} | ${t[fg]} on ${t[bg]} | ${r.toFixed(2)}:1 | ${min}:1 | ${ok ? "pass" : "FAIL"} |`);
  }
}
console.log("| theme | use | pair | ratio | AA (4.5:1) |\n|---|---|---|---|---|\n" + rows.join("\n"));
if (failed) process.exit(1);
