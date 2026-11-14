// Command palette (Ctrl/Cmd+K): fuzzy search over collections, new chat, theme, history, How it
// works, keyboard shortcuts and the repository. Loaded on first use.

import { el, icon, ICONS, modKey } from "./dom.ts";
import { reducedMotion } from "./motion.ts";
import { animate } from "motion/mini";
import { spring } from "motion";

export interface PaletteItem {
  id: string;
  group: string;
  label: string;
  hint?: string;
  icon?: readonly string[] | string[];
  kbd?: string;
  run: () => void;
}

let dialog: HTMLDialogElement | null = null;
let opener: HTMLElement | null = null;

/** Subsequence match with a bonus for consecutive and word-start hits; 0 when no match. */
export function fuzzyScore(query: string, text: string): number {
  const q = query.toLowerCase().replace(/\s+/g, "");
  if (!q) return 1;
  const t = text.toLowerCase();
  let score = 0;
  let ti = 0;
  let streak = 0;
  for (const ch of q) {
    const i = t.indexOf(ch, ti);
    if (i < 0) return 0;
    streak = i === ti ? streak + 1 : 0;
    score += 1 + streak + (i === 0 || /[\s&,-]/.test(t[i - 1] ?? "") ? 2 : 0);
    ti = i + 1;
  }
  return score / (1 + t.length / 40);
}

export const SHORTCUTS: [string, string][] = [
  [`${modKey} K`, "Command palette"],
  ["/", "Focus the question box"],
  [`${modKey} ⇧ O`, "New chat"],
  ["Enter", "Send"],
  ["⇧ Enter", "New line"],
  ["Esc", "Stop streaming or close an overlay"],
];

function build(): HTMLDialogElement {
  const d = el("dialog", "dialog palette");
  d.setAttribute("aria-label", "Command palette");
  const input = el("input", "palette-input");
  input.type = "text";
  input.placeholder = "Search collections, chats, actions…";
  input.setAttribute("aria-label", "Search commands");
  input.setAttribute("role", "combobox");
  input.setAttribute("aria-expanded", "true");
  input.setAttribute("aria-controls", "palette-list");
  input.autocomplete = "off";
  const list = el("ul", "palette-list");
  list.id = "palette-list";
  list.setAttribute("role", "listbox");
  d.append(input, list);
  d.addEventListener("close", () => {
    if (opener && document.body.contains(opener)) opener.focus();
  });
  d.addEventListener("click", (e) => {
    if (e.target === d) d.close();
  });
  document.body.appendChild(d);
  return d;
}

export function openPalette(items: PaletteItem[]): void {
  opener = document.activeElement as HTMLElement | null;
  dialog ??= build();
  const d = dialog;
  const input = d.querySelector<HTMLInputElement>(".palette-input") as HTMLInputElement;
  const list = d.querySelector<HTMLUListElement>(".palette-list") as HTMLUListElement;
  let selected = 0;
  let visible: PaletteItem[] = [];
  let showingShortcuts = false;

  const renderShortcuts = (): void => {
    showingShortcuts = true;
    list.textContent = "";
    const li = el("li");
    li.appendChild(el("div", "palette-group", "Keyboard shortcuts"));
    const grid = el("div", "shortcuts");
    for (const [keys, what] of SHORTCUTS) {
      grid.appendChild(el("span", null, what));
      grid.appendChild(el("kbd", "kbd", keys));
    }
    li.appendChild(grid);
    list.appendChild(li);
  };

  const render = (): void => {
    showingShortcuts = false;
    const q = input.value;
    visible = items
      .map((it) => ({ it, s: fuzzyScore(q, `${it.label} ${it.hint ?? ""} ${it.group}`) }))
      .filter((x) => x.s > 0)
      .sort((a, b) => (q ? b.s - a.s : 0))
      .map((x) => x.it);
    selected = Math.min(selected, Math.max(0, visible.length - 1));
    list.textContent = "";
    if (!visible.length) {
      list.appendChild(el("li", "palette-empty", "Nothing matches."));
      return;
    }
    let group = "";
    visible.forEach((it, i) => {
      if (it.group !== group) {
        group = it.group;
        const g = el("li", "palette-group", group);
        g.setAttribute("role", "presentation");
        list.appendChild(g);
      }
      const li = el("li");
      li.setAttribute("role", "presentation");
      const b = el("button", "palette-item");
      b.type = "button";
      b.setAttribute("role", "option");
      b.setAttribute("aria-selected", i === selected ? "true" : "false");
      b.id = `palette-opt-${i}`;
      const ic = el("span", "rail-item-icon");
      ic.appendChild(icon(it.icon ? [...it.icon] : [...ICONS.search], 18));
      b.appendChild(ic);
      const text = el("span", "rail-item-text");
      text.appendChild(el("span", "rail-item-name", it.label));
      if (it.hint) text.appendChild(el("span", "rail-item-meta", it.hint));
      b.appendChild(text);
      if (it.kbd) b.appendChild(el("kbd", "kbd", it.kbd));
      b.addEventListener("click", () => choose(it));
      b.addEventListener("mousemove", () => {
        if (selected !== i) {
          selected = i;
          for (const o of list.querySelectorAll<HTMLElement>(".palette-item")) o.setAttribute("aria-selected", o === b ? "true" : "false");
        }
      });
      li.appendChild(b);
      list.appendChild(li);
    });
    input.setAttribute("aria-activedescendant", `palette-opt-${selected}`);
  };

  const choose = (it: PaletteItem): void => {
    if (it.id === "shortcuts") {
      renderShortcuts();
      return;
    }
    d.close();
    it.run();
  };

  input.oninput = () => {
    selected = 0;
    render();
  };
  input.onkeydown = (e) => {
    if (showingShortcuts && e.key !== "Escape") {
      render();
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!visible.length) return;
      selected = (selected + (e.key === "ArrowDown" ? 1 : visible.length - 1)) % visible.length;
      list.querySelectorAll<HTMLElement>(".palette-item").forEach((o, i) => o.setAttribute("aria-selected", i === selected ? "true" : "false"));
      list.querySelectorAll<HTMLElement>(".palette-item")[selected]?.scrollIntoView({ block: "nearest" });
      input.setAttribute("aria-activedescendant", `palette-opt-${selected}`);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const it = visible[selected];
      if (it) choose(it);
    }
  };
  input.value = "";
  render();
  d.showModal();
  input.focus();
  // materialise: scale 0.96 -> 1 with a blur-to-sharp
  if (reducedMotion()) animate(d, { opacity: [0, 1] }, { duration: 0.15 });
  else animate(d, { transform: ["scale(0.96)", "scale(1)"], opacity: [0, 1], filter: ["blur(6px)", "blur(0px)"] }, { type: spring, bounce: 0, visualDuration: 0.3 });
}

export function closePalette(): boolean {
  if (dialog?.open) {
    dialog.close();
    return true;
  }
  return false;
}
