// The evidence panel: a side panel on desktop (the conversation keeps its layout through a FLIP
// transform), a right drawer on tablets and a bottom sheet with two detents on phones. Enter and
// exit share a path; every move starts from the on-screen value and hands over drag velocity.

import { $ } from "./dom.ts";
import { linkLabels } from "./linking.ts";
import { currentTranslate, project, reducedMotion, rubberband, springTransform, VelocityTracker, fade } from "./motion.ts";

type Mode = "panel" | "drawer" | "sheet";
type Detent = "half" | "full";

const DESKTOP = "(min-width: 1100px)";
const MOBILE = "(max-width: 699px)";

export class Evidence {
  private readonly app = $("app");
  private readonly panel = $("evidence");
  private readonly column = $("column");
  private readonly scrim = $("scrim");
  private readonly grip = $("sheet-grip");
  private readonly body = $("evidence-body");
  private opener: HTMLElement | null = null;
  private running: { stop(): void } | null = null;
  private detent: Detent = "half";
  open = false;

  constructor(private readonly onClosed: () => void) {
    this.bindSheet();
    window.addEventListener("resize", () => {
      if (this.open) this.place(false);
    });
  }

  mode(): Mode {
    if (window.matchMedia(DESKTOP).matches) return "panel";
    if (window.matchMedia(MOBILE).matches) return "sheet";
    return "drawer";
  }

  /** Geometry of the sheet: the panel is 90 % of the viewport; translateY for each detent. */
  private sheetY(d: Detent | "closed"): number {
    const h = this.panel.getBoundingClientRect().height || window.innerHeight * 0.9;
    if (d === "closed") return h;
    if (d === "full") return 0;
    return Math.max(0, h - window.innerHeight * 0.4);
  }

  show(opener?: HTMLElement | null, detent: Detent = "half"): void {
    this.opener = opener ?? (document.activeElement as HTMLElement | null);
    const wasOpen = this.open;
    this.open = true;
    this.detent = detent;
    this.panel.hidden = false;
    this.panel.setAttribute("aria-hidden", "false");
    $("evidence-toggle").setAttribute("aria-expanded", "true");
    const mode = this.mode();
    if (mode === "panel" && !wasOpen) this.flipColumn(() => (this.app.dataset.evidence = "open"));
    if (mode !== "panel") {
      this.scrim.hidden = false;
      void fade(this.scrim, Number(getComputedStyle(this.scrim).opacity) || 0, 1, 150);
    }
    this.place(true);
    if (!wasOpen) {
      if (mode === "sheet") this.panel.setAttribute("role", "dialog");
      else this.panel.removeAttribute("role");
      $("evidence-title").focus({ preventScroll: true });
    }
  }

  hide(velocity = 0): void {
    if (!this.open) return;
    this.open = false;
    this.panel.setAttribute("aria-hidden", "true");
    $("evidence-toggle").setAttribute("aria-expanded", "false");
    const mode = this.mode();
    if (mode === "panel") this.flipColumn(() => (this.app.dataset.evidence = "closed"));
    else {
      void fade(this.scrim, Number(getComputedStyle(this.scrim).opacity) || 1, 0, 150, () => {
        if (!this.open) this.scrim.hidden = true;
      });
    }
    this.running?.stop();
    const cur = currentTranslate(this.panel);
    if (mode === "sheet") {
      this.running = springTransform(this.panel, "translateY", cur.y, this.sheetY("closed"), { velocity, gesture: velocity !== 0, onComplete: () => this.afterClose() });
    } else {
      const w = this.panel.getBoundingClientRect().width;
      this.running = springTransform(this.panel, "translateX", cur.x, w, { velocity, gesture: velocity !== 0, onComplete: () => this.afterClose() });
    }
    const opener = this.opener;
    if (opener && document.body.contains(opener)) opener.focus({ preventScroll: true });
    this.onClosed();
  }

  private afterClose(): void {
    if (!this.open) this.panel.hidden = true;
  }

  toggle(opener?: HTMLElement | null): void {
    if (this.open) this.hide();
    else this.show(opener);
  }

  /** Move the panel to its resting place for the current mode (interruptible). */
  private place(animated: boolean, velocity = 0): void {
    this.running?.stop();
    const mode = this.mode();
    const cur = currentTranslate(this.panel);
    if (mode === "sheet") {
      const target = this.sheetY(this.detent);
      const from = this.panel.hidden || !animated ? this.sheetY("closed") : cur.y;
      this.running = springTransform(this.panel, "translateY", animated ? from : target, target, { velocity, gesture: velocity !== 0 });
    } else {
      const w = this.panel.getBoundingClientRect().width;
      const from = animated ? (this.wasHiddenAtStart(cur.x, w) ? w : cur.x) : 0;
      this.running = springTransform(this.panel, "translateX", from, 0, { velocity, gesture: velocity !== 0 });
    }
  }

  private wasHiddenAtStart(x: number, w: number): boolean {
    // a freshly unhidden panel reports its CSS transform (100 %) as a matrix in px
    return Math.abs(x - w) < 1 || x === 0 && getComputedStyle(this.panel).transform === "none";
  }

  /** FLIP: the conversation column keeps its place while the grid gains or loses the panel slot. */
  private flipColumn(change: () => void): void {
    const before = this.column.getBoundingClientRect().left;
    change();
    const after = this.column.getBoundingClientRect().left;
    const dx = before - after;
    if (!dx || reducedMotion()) return;
    springTransform(this.column, "translateX", dx, 0);
  }

  setContent(node: Node): void {
    this.body.textContent = "";
    this.body.appendChild(node);
  }

  /** Highlight the source card of a citation and bring it into view. */
  link(label: string | null, scroll = false): void {
    const card = linkLabels(this.body, label, ".source-card");
    if (card && scroll) card.scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
  }

  openCard(label: string): void {
    const card = this.body.querySelector<HTMLElement>(`.source-card[data-label="${label}"]`);
    if (!card) return;
    const head = card.querySelector<HTMLButtonElement>(".source-head");
    if (head && !card.classList.contains("is-open")) head.click();
    this.link(label, true);
    head?.focus({ preventScroll: true });
  }

  // ---------------------------------------------------------------- bottom sheet gestures

  private bindSheet(): void {
    const tracker = new VelocityTracker();
    let startY = 0;
    let startT = 0;
    let dragging = false;
    let fromBody = false;
    let decided = false;
    // The pointer is captured only once the drag passes a 10 px threshold: capturing on
    // pointer-down would retarget the click of a button inside the sheet (the close button,
    // a source card's header) to the capturing element, so taps would stop working.
    const down = (e: PointerEvent, body: boolean): void => {
      if (this.mode() !== "sheet" || !this.open) return;
      if (body && this.body.scrollTop > 0) return;
      fromBody = body;
      decided = false;
      dragging = true;
      startY = e.clientY;
      startT = currentTranslate(this.panel).y;
      tracker.reset();
      tracker.add(startT);
    };
    const move = (e: PointerEvent): void => {
      if (!dragging) return;
      const dy = e.clientY - startY;
      if (!decided) {
        if (Math.abs(dy) < 10) return;
        if (fromBody && (dy < 0 || this.body.scrollTop > 0)) {
          dragging = false;
          return;
        }
        decided = true;
        this.running?.stop();
        startT = currentTranslate(this.panel).y;
        startY = e.clientY - Math.sign(dy) * 10;
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      }
      e.preventDefault();
      const h = this.sheetY("closed");
      let y = startT + dy;
      if (y < 0) y = rubberband(y, h);
      if (y > h) y = h + rubberband(y - h, h);
      tracker.add(y);
      this.panel.style.transform = `translateY(${y}px)`;
    };
    const up = (): void => {
      if (!dragging) return;
      dragging = false;
      if (!decided) return; // a tap: let the click through untouched
      const v = tracker.velocity();
      const y = currentTranslate(this.panel).y;
      const h = this.sheetY("closed");
      const half = this.sheetY("half");
      const target = y + project(v);
      const stops: [Detent | "closed", number][] = [
        ["full", 0],
        ["half", half],
        ["closed", h],
      ];
      let best = stops[0] as [Detent | "closed", number];
      for (const s of stops) if (Math.abs(s[1] - target) < Math.abs(best[1] - target)) best = s;
      if (best[0] === "closed") this.hide(v);
      else {
        this.detent = best[0];
        this.place(true, v);
      }
    };
    for (const [node, body] of [
      [this.grip, false],
      [$("evidence-title").parentElement as HTMLElement, false],
      [this.body, true],
    ] as [HTMLElement, boolean][]) {
      node.addEventListener("pointerdown", (e) => down(e, body));
      node.addEventListener("pointermove", move);
      node.addEventListener("pointerup", up);
      node.addEventListener("pointercancel", up);
    }
  }
}
