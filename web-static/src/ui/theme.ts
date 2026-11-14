// Theme: follows prefers-color-scheme until the visitor picks one (html[data-theme]); the switch
// eases colours over 250 ms through html.theming.

export type Theme = "dark" | "light";

const media = typeof window !== "undefined" && "matchMedia" in window ? window.matchMedia("(prefers-color-scheme: light)") : null;

export function systemTheme(): Theme {
  return media?.matches ? "light" : "dark";
}

export function currentTheme(): Theme {
  const chosen = document.documentElement.dataset.theme;
  return chosen === "light" || chosen === "dark" ? chosen : systemTheme();
}

export function applyTheme(theme: Theme | null, animate = true): void {
  const root = document.documentElement;
  if (animate) {
    root.classList.add("theming");
    setTimeout(() => root.classList.remove("theming"), 300);
  }
  if (theme) root.dataset.theme = theme;
  else delete root.dataset.theme;
  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]:not([media])');
  if (meta) meta.content = currentTheme() === "light" ? "#F5F7FC" : "#0B1230";
}

export function themeToggleLabel(): string {
  return currentTheme() === "light" ? "Switch to the dark theme" : "Switch to the light theme";
}
