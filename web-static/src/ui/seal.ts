// The seal: a thin ring and a V whose right stroke runs on into a check tick. Three uses:
//   - the hero mark, whose ring closes with the real model-download progress and idles slowly
//   - the stamp under an answer, driven only by the real verdict (section 4 of the brief)
//   - small marks (wordmark, favicon) drawn once

import { svgEl } from "./dom.ts";
import { coarsePointer, reducedMotion, springTo } from "./motion.ts";

export type Verdict = "supported" | "partial" | "unsupported" | "none";

const RING = 173; // 2 * pi * 27.5
const CHECK = 64; // length of the V-check path

export function sealSvg(opts: { ticks?: boolean; ripple?: boolean } = {}): SVGSVGElement {
  const svg = svgEl("svg", { viewBox: "0 0 64 64", "aria-hidden": "true" });
  if (opts.ripple) {
    svg.appendChild(svgEl("circle", { class: "seal-ripple", cx: 32, cy: 32, r: 27.5, fill: "none", stroke: "currentColor", "stroke-width": 1.5 }));
  }
  if (opts.ticks) {
    const g = svgEl("g", { class: "seal-ticks", fill: "none", stroke: "currentColor", "stroke-width": 1, "stroke-linecap": "round" });
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      const r1 = 30.5;
      const r2 = i % 6 === 0 ? 27.5 : 29;
      g.appendChild(svgEl("line", { x1: (32 + r1 * Math.cos(a)).toFixed(2), y1: (32 + r1 * Math.sin(a)).toFixed(2), x2: (32 + r2 * Math.cos(a)).toFixed(2), y2: (32 + r2 * Math.sin(a)).toFixed(2) }));
    }
    svg.appendChild(g);
  }
  svg.appendChild(svgEl("circle", { class: "seal-ring", cx: 32, cy: 32, r: 27.5, fill: "none", stroke: "currentColor", "stroke-width": 2.5, "stroke-linecap": "round", transform: "rotate(-90 32 32)" }));
  svg.appendChild(svgEl("path", { class: "seal-check", d: "M17 23 L28 44 L49 17", fill: "none", stroke: "currentColor", "stroke-width": 6, "stroke-linecap": "round", "stroke-linejoin": "round" }));
  return svg;
}

/** Hero: the ring's dashoffset follows the real download fraction (0..1). */
export function setSealProgress(host: HTMLElement, fraction: number): void {
  const ring = host.querySelector<SVGCircleElement>(".seal-ring");
  if (!ring) return;
  const f = Math.max(0, Math.min(1, fraction));
  ring.style.strokeDashoffset = String(RING * (1 - f));
}

export function sealReady(host: HTMLElement): void {
  const ring = host.querySelector<SVGCircleElement>(".seal-ring");
  const check = host.querySelector<SVGPathElement>(".seal-check");
  if (ring) {
    ring.style.transition = reducedMotion() ? "none" : "stroke-dashoffset 500ms cubic-bezier(0.2, 0.8, 0.2, 1)";
    ring.style.strokeDashoffset = "0";
  }
  if (check) {
    check.style.transition = reducedMotion() ? "none" : "stroke-dashoffset 420ms cubic-bezier(0.2, 0.8, 0.2, 1) 120ms";
    check.style.strokeDashoffset = "0";
  }
  host.classList.add("is-ready");
}

/**
 * Stamp a verdict: ring closes (fully / three quarters / stays open), the check draws, the mark
 * springs from 1.15 to 1 while one ripple ring expands to nothing, all started on one frame.
 */
export function stampSeal(seal: HTMLElement, verdict: Verdict, animate = true): void {
  seal.dataset.verdict = verdict;
  const ring = seal.querySelector<SVGCircleElement>(".seal-ring");
  const check = seal.querySelector<SVGPathElement>(".seal-check");
  const ripple = seal.querySelector<SVGCircleElement>(".seal-ripple");
  const closed = verdict === "supported" ? 1 : verdict === "partial" ? 0.75 : verdict === "unsupported" ? 0.55 : 1;
  const drawCheck = verdict === "supported" || verdict === "partial";
  const instant = !animate || reducedMotion();
  const svg = seal.querySelector("svg");
  requestAnimationFrame(() => {
    if (ring) {
      ring.style.transition = instant ? "none" : "stroke-dashoffset 480ms cubic-bezier(0.2, 0.8, 0.2, 1)";
      ring.style.strokeDashoffset = String(RING * (1 - closed));
    }
    if (check) {
      check.style.transition = instant ? "none" : "stroke-dashoffset 360ms cubic-bezier(0.2, 0.8, 0.2, 1) 140ms";
      check.style.strokeDashoffset = drawCheck ? "0" : String(CHECK);
    }
    if (instant) {
      if (svg) springTo(svg, { opacity: [0, 1] });
      return;
    }
    if (svg) springTo(svg, { scale: [1.15, 1] }, { stamp: true });
    if (ripple) {
      ripple.style.opacity = "0.9";
      springTo(ripple, { scale: [1, 1.9], opacity: [0.9, 0] }, { stamp: true });
    }
    if (coarsePointer() && typeof navigator.vibrate === "function") {
      try {
        navigator.vibrate(10);
      } catch {
        /* unsupported */
      }
    }
  });
}

export function verdictLabel(verdict: Verdict, n: number, supported?: number, total?: number): { label: string; detail: string } {
  switch (verdict) {
    case "supported":
      return { label: `Verified against ${n} source${n === 1 ? "" : "s"}`, detail: supported !== undefined && total !== undefined ? `${supported} of ${total} sentence${total === 1 ? "" : "s"} supported` : "guards passed · critic accepted" };
    case "partial":
      return { label: "Partially supported", detail: `${supported ?? 0} of ${total ?? 0} sentences supported` };
    case "unsupported":
      return { label: "Not enough evidence", detail: supported !== undefined && total !== undefined ? `${supported} of ${total} sentences supported` : "the documents do not answer this" };
    default:
      return { label: `${n} source${n === 1 ? "" : "s"}`, detail: "not yet verified" };
  }
}
