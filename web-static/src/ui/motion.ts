// Springs and motion preferences. The "motion/mini" animate drives the Web Animations API; the
// spring generator gives it Apple-style parameters: damping 1.0 / response 0.35 for UI movement
// (bounce 0, visualDuration 0.35), damping 0.8 / response 0.3 for momentum gestures (bounce 0.2,
// visualDuration 0.3), and the stamp's damping 0.7 (bounce 0.3). Every animation starts from the
// element's current value and can be interrupted by starting another.

import { animate } from "motion/mini";
import { spring } from "motion";

export const SPRING_UI = { type: spring, bounce: 0, visualDuration: 0.35 } as const;
export const SPRING_GESTURE = { type: spring, bounce: 0.2, visualDuration: 0.3 } as const;
export const SPRING_STAMP = { type: spring, bounce: 0.3, visualDuration: 0.4 } as const;

/** html.reduce-motion mirrors prefers-reduced-motion, so CSS and code read one flag. */
export function reducedMotion(): boolean {
  return document.documentElement.classList.contains("reduce-motion");
}

/** Set html.reduce-motion from the media query now and whenever the preference changes. */
export function syncReducedMotion(): void {
  const query = typeof window.matchMedia === "function" ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  const apply = (): void => {
    document.documentElement.classList.toggle("reduce-motion", Boolean(query?.matches));
  };
  apply();
  query?.addEventListener("change", apply);
}

export function coarsePointer(): boolean {
  return typeof window !== "undefined" && "matchMedia" in window && window.matchMedia("(pointer: coarse)").matches;
}

type Target = Record<string, string | number | (string | number)[]>;
interface Options {
  velocity?: number;
  gesture?: boolean;
  stamp?: boolean;
  onComplete?: () => void;
}

/** Spring an element to a target; under reduced motion a 150 ms opacity-only change. */
export function springTo(el: Element, target: Target, opts: Options = {}): { stop(): void; finished: Promise<unknown> } {
  if (reducedMotion()) {
    const safe: Target = {};
    if ("opacity" in target) safe.opacity = target.opacity as number;
    for (const [k, v] of Object.entries(target)) if (k !== "opacity") (el as HTMLElement).style.setProperty(cssName(k), String(Array.isArray(v) ? v[v.length - 1] : v));
    const a = animate(el, safe, { duration: 0.15, ease: "linear", onComplete: opts.onComplete });
    return { stop: () => a.stop(), finished: a.finished };
  }
  const base = opts.stamp ? SPRING_STAMP : opts.gesture ? SPRING_GESTURE : SPRING_UI;
  const a = animate(el, target, { ...base, ...(opts.velocity !== undefined ? { velocity: opts.velocity } : {}), onComplete: opts.onComplete });
  return { stop: () => a.stop(), finished: a.finished };
}

function cssName(k: string): string {
  return k === "x" || k === "y" ? "transform" : k;
}

/** A short fade (and rise) used for enter/exit when a spring is not warranted. */
export function fade(el: Element, from: number, to: number, ms = 150, onComplete?: () => void): { finished: Promise<unknown> } {
  const a = animate(el, { opacity: [from, to] }, { duration: ms / 1000, ease: "linear", onComplete });
  return { finished: a.finished };
}

/** Apple's momentum projection: where a flick would come to rest. */
export function project(velocityPxPerS: number, decelerationRate = 0.998): number {
  return ((velocityPxPerS / 1000) * decelerationRate) / (1 - decelerationRate);
}

/** Progressive resistance past a boundary. */
export function rubberband(overshoot: number, dimension: number, constant = 0.55): number {
  return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot));
}

/** Velocity (px/s) from the last few pointer samples. */
export class VelocityTracker {
  private samples: { t: number; v: number }[] = [];
  add(v: number): void {
    const t = performance.now();
    this.samples.push({ t, v });
    this.samples = this.samples.filter((s) => t - s.t < 100);
  }
  velocity(): number {
    const s = this.samples;
    if (s.length < 2) return 0;
    const a = s[0] as { t: number; v: number };
    const b = s[s.length - 1] as { t: number; v: number };
    const dt = b.t - a.t;
    return dt > 0 ? ((b.v - a.v) / dt) * 1000 : 0;
  }
  reset(): void {
    this.samples = [];
  }
}

/** The element's current translate (px) from its computed transform. */
export function currentTranslate(el: HTMLElement): { x: number; y: number } {
  const t = getComputedStyle(el).transform;
  if (!t || t === "none") return { x: 0, y: 0 };
  const m = /matrix\(([^)]+)\)/.exec(t);
  if (!m) return { x: 0, y: 0 };
  const parts = (m[1] ?? "").split(",").map((v) => Number.parseFloat(v));
  return { x: parts[4] ?? 0, y: parts[5] ?? 0 };
}

/**
 * Spring one transform function from its current value to a target. The generator runs over a
 * 0..100 range for string keyframes, so a release velocity in px/s is scaled to that range.
 */
export function springTransform(
  el: HTMLElement,
  fn: "translateX" | "translateY" | "scale",
  from: number,
  to: number,
  opts: { velocity?: number; gesture?: boolean; onComplete?: () => void; unit?: string } = {},
): { stop(): void; finished: Promise<unknown> } {
  const unit = opts.unit ?? (fn === "scale" ? "" : "px");
  const delta = to - from;
  const velocity = opts.velocity !== undefined && delta !== 0 ? (opts.velocity * 100) / delta : undefined;
  const v = (n: number): string => `${fn}(${n}${unit})`;
  if (reducedMotion()) {
    el.style.transform = v(to);
    const a = animate(el, { opacity: [getComputedStyle(el).opacity, 1] }, { duration: 0.15, ease: "linear", onComplete: opts.onComplete });
    return { stop: () => a.stop(), finished: a.finished };
  }
  const base = opts.gesture ? SPRING_GESTURE : SPRING_UI;
  const a = animate(el, { transform: [v(from), v(to)] }, { ...base, ...(velocity !== undefined ? { velocity } : {}), onComplete: opts.onComplete });
  return { stop: () => a.stop(), finished: a.finished };
}
