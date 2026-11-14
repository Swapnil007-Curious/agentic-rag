// DOM helpers: element lookup, SVG icons (1.5 px line icons, built with createElementNS, never
// from markup strings) and the five collection marks.

export const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} is missing`);
  return node as T;
};

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string | null, text?: string | null): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

const NS = "http://www.w3.org/2000/svg";

export function svgEl<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

/** A 24-grid line icon from one or more path strings. */
export function icon(paths: string | readonly string[], size = 18, extra: Record<string, string | number> = {}): SVGSVGElement {
  const svg = svgEl("svg", { viewBox: "0 0 24 24", width: size, height: size, "aria-hidden": "true", ...extra });
  for (const d of Array.isArray(paths) ? paths : [paths]) {
    svg.appendChild(svgEl("path", { d, fill: "none", stroke: "currentColor", "stroke-width": 1.5, "stroke-linecap": "round", "stroke-linejoin": "round" }));
  }
  return svg;
}

export const ICONS = {
  copy: ["M9 9h10v10H9z", "M5 15V5h10"],
  refresh: ["M4 12a8 8 0 0 1 13.7-5.6L20 8.5", "M20 4v4.5h-4.5", "M20 12a8 8 0 0 1-13.7 5.6L4 15.5", "M4 20v-4.5h4.5"],
  share: ["M4 12v7a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-7", "M12 15V4", "M8 8l4-4 4 4"],
  evidence: ["M4 5h16v14H4z", "M15 5v14", "M7 9h5M7 12h5"],
  verify: ["M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6z", "M9 12l2 2 4-4"],
  close: ["M6 6l12 12", "M18 6L6 18"],
  trash: ["M5 7h14", "M9 7V4h6v3", "M7 7l1 13h8l1-13"],
  undo: ["M9 14 4 9l5-5", "M4 9h11a5 5 0 0 1 0 10h-3"],
  search: ["M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13z", "M20 20l-4.5-4.5"],
  plus: ["M12 5v14", "M5 12h14"],
  sun: ["M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z", "M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"],
  moon: ["M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"],
  chat: ["M4 5h16v11H9l-5 4z"],
  book: ["M5 5.5A2.5 2.5 0 0 1 7.5 3H19v15H7.5A2.5 2.5 0 0 0 5 20.5z", "M5 18.5V5.5", "M19 18H7.5"],
  flow: ["M4 12h4l2-5 4 10 2-5h4"],
  github: ["M9 19c-4 1.5-4-2-6-2m12 4v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.2 4.2 0 0 0-.1-3.2s-1-.3-3.4 1.3a11.8 11.8 0 0 0-6.2 0C6.5 2.8 5.5 3.1 5.5 3.1a4.2 4.2 0 0 0-.1 3.2A4.6 4.6 0 0 0 4 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21"],
  chevron: ["M7 10l5 5 5-5"],
  keyboard: ["M3 6h18v12H3z", "M7 10h.01M11 10h.01M15 10h.01M7 14h10"],
  none: ["M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z", "M5.6 5.6l12.8 12.8"],
  globe: ["M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z", "M3 12h18", "M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18"],
} as const;

/** Collection marks: circuit, scale, courthouse, heart, atom. */
export const COLLECTION_ICONS: Record<string, string[]> = {
  cs_ai: ["M8 8h8v8H8z", "M10 8V5M14 8V5M10 19v-3M14 19v-3M5 10h3M5 14h3M16 10h3M16 14h3"],
  consumer_law: ["M12 4v16", "M5 8h14", "M8 8l-3 6a3 3 0 0 0 6 0zM16 8l-3 6a3 3 0 0 0 6 0z", "M9 20h6"],
  indian_law: ["M3 9l9-5 9 5", "M5 9v9M9 9v9M15 9v9M19 9v9", "M3 18h18v2H3z"],
  human_biology: ["M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.5A4 4 0 0 1 19 10c0 5.5-7 10-7 10z", "M6 12h3l1.5-2.5 2 5 1.5-2.5h4"],
  physics_chemistry: ["M12 12m-1.5 0a1.5 1.5 0 1 0 3 0 1.5 1.5 0 1 0-3 0", "M12 3c2.5 0 4.5 4 4.5 9s-2 9-4.5 9-4.5-4-4.5-9 2-9 4.5-9z", "M4.2 7.5c1.3-2.2 5.7-1.5 10 1s7 6.3 5.6 8.5-5.7 1.5-10-1-7-6.3-5.6-8.5z", "M19.8 7.5c-1.3-2.2-5.7-1.5-10 1s-7 6.3-5.6 8.5 5.7 1.5 10-1 7-6.3 5.6-8.5z"],
};

export function collectionIcon(id: string, size = 20): SVGSVGElement {
  return icon(COLLECTION_ICONS[id] ?? ICONS.book, size);
}

/** Short collection names for tiles and pills. */
export const SHORT_NAMES: Record<string, string> = {
  cs_ai: "CS & AI",
  consumer_law: "Consumer Law",
  indian_law: "Indian Law",
  human_biology: "Human Biology",
  physics_chemistry: "Physics & Chemistry",
};

export function shortName(id: string, fallback: string): string {
  return SHORT_NAMES[id] ?? fallback;
}

export const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
export const modKey = isMac ? "⌘" : "Ctrl";
