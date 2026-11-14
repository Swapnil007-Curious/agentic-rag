// Toasts with an optional action (Undo stays for 5 s).

import { el } from "./dom.ts";
import { fade } from "./motion.ts";

export function toast(text: string, action?: { label: string; onClick: () => void }, ms = 5000): void {
  const host = document.getElementById("toasts");
  if (!host) return;
  const node = el("div", "toast");
  node.setAttribute("role", "status");
  node.appendChild(el("span", null, text));
  let timer = 0;
  const dismiss = (): void => {
    clearTimeout(timer);
    void fade(node, 1, 0, 150, () => node.remove());
  };
  if (action) {
    const b = el("button", "act", action.label);
    b.type = "button";
    b.addEventListener("click", () => {
      action.onClick();
      dismiss();
    });
    node.appendChild(b);
  }
  host.appendChild(node);
  void fade(node, 0, 1, 150);
  timer = window.setTimeout(dismiss, ms);
}
