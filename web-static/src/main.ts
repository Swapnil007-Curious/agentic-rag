// Veridex: the page controller. Retrieval and query embedding run in this browser (engine.ts and
// friends, unchanged); replies stream from the LLM proxy. Model text is rendered by markdown.ts
// (text nodes only); everything else is built with createElement. The UI shows only data the
// pipeline really produces: see DESIGN.md for the map from each element to its source.

import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/layout.css";
import "./styles/components.css";
import { ProxyError } from "./api.ts";
import { CollectionStore, loadManifest } from "./data.ts";
import { Embedder } from "./embedder.ts";
import { Assistant, type ChatOutcome, type GroundedCheck, type StageEvent } from "./engine.ts";
import type { Summary, Turn } from "./history.ts";
import { copyText, render, SAFE_URL } from "./markdown.ts";
import { storage } from "./storage.ts";
import type { CollectionMeta, Manifest, Passage } from "./types.ts";
import { UploadError, UploadStore } from "./uploads.ts";
import type { VerifyResult } from "./verify.ts";
import { $, collectionIcon, el, icon, ICONS, modKey, shortName } from "./ui/dom.ts";
import { describeError, formatCountdown, type ErrorView } from "./ui/errors.ts";
import { Evidence } from "./ui/evidence.ts";
import { excerpt, markTerms, queryTerms } from "./ui/highlight.ts";
import { linkLabels } from "./ui/linking.ts";
import { fade, reducedMotion, springTo, springTransform, syncReducedMotion } from "./ui/motion.ts";
import { sealReady, sealSvg, setSealProgress, stampSeal, verdictLabel, type Verdict } from "./ui/seal.ts";
import { buildShareUrl, parseShareParams } from "./ui/share.ts";
import { suggestionsFor } from "./ui/suggestions.ts";
import { applyTheme, currentTheme, themeToggleLabel, type Theme } from "./ui/theme.ts";
import { toast } from "./ui/toast.ts";
import { applyStage, finishTrace, initialTrace, renderTrace, type TraceStep } from "./ui/trace.ts";
import { ensureHljs, ensureKatex, needsCode, needsMath } from "./ui/vendor.ts";

declare const __WORKER_URL__: string;
declare const __MODEL_DTYPE__: string;

const BASE = import.meta.env.BASE_URL;
const STORE_KEY = "ragsys.static.v1";
const PREF_KEY = "ragsys.static.prefs.v1";
const MAX_CONVERSATIONS = 50;
const REPO = "https://github.com/Swapnil007-Curious/agentic-rag";
const TITLE = "Veridex";

type Status = "streaming" | "done" | "error" | "stopped";

interface StoredError extends ErrorView {
  at: number;
}

interface Meta {
  status: Status;
  collection: string;
  notice: string;
  mode?: "chat" | "grounded_only";
  grounded?: boolean;
  sourceLine?: string;
  citations?: string[];
  passages?: Passage[];
  context?: Passage[];
  outcome?: "answered" | "abstained";
  closest?: { heading_path: string; pages: string }[];
  abstentionMessage?: string;
  error?: StoredError;
  verify?: VerifyResult | { error: string };
  statusText?: string;
  trace?: TraceStep[];
  check?: GroundedCheck;
  verifyMs?: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  meta?: Meta;
  node?: HTMLElement;
  stamped?: boolean;
}

interface Conversation {
  id: string;
  title: string;
  created: number;
  updated: number;
  collection: string;
  groundedOnly: boolean;
  summary: Summary | null;
  messages: Message[];
}

interface Prefs {
  collection?: string;
  theme?: Theme;
  mode?: "chat" | "grounded";
  railCollapsed?: boolean;
}

const state = {
  conversations: [] as Conversation[],
  currentId: null as string | null,
  manifest: null as Manifest | null,
  assistant: null as Assistant | null,
  streaming: null as { controller: AbortController; conv: Conversation; msg: Message; deltas: { t: number; n: number }[] } | null,
  autoScroll: true,
  uploading: false,
  collection: "",
  modelReady: false,
  evidenceFor: null as Message | null,
  popoverOpener: null as HTMLElement | null,
};

let evidence: Evidence;

/** Replace text only when it changed: a rewritten text node is a new paint (and a new LCP candidate). */
function setText(node: HTMLElement, text: string): void {
  if (node.textContent !== text) node.textContent = text;
}

// ------------------------------------------------------------------ persistence

function current(): Conversation | null {
  return state.conversations.find((c) => c.id === state.currentId) ?? null;
}

function saveAll(): void {
  if (!storage.ok) return;
  const data = state.conversations.slice(0, MAX_CONVERSATIONS).map((c) => ({
    ...c,
    messages: c.messages.map((m) => ({ role: m.role, content: m.content, meta: stripMeta(m.meta) })),
  }));
  while (!storage.set(STORE_KEY, { v: 1, conversations: data })) {
    const idx = data.map((c) => c.id !== state.currentId).lastIndexOf(true);
    if (idx < 0) {
      storage.ok = false;
      $("storage-note").hidden = false;
      return;
    }
    data.splice(idx, 1);
  }
}

function stripMeta(meta: Meta | undefined): Meta | undefined {
  if (!meta) return undefined;
  const { context: _c, statusText: _s, ...rest } = meta;
  if (rest.status === "streaming") rest.status = "stopped";
  if (rest.trace) rest.trace = finishTrace(rest.trace);
  return rest;
}

function loadAll(): void {
  if (!storage.probe()) {
    storage.ok = false;
    $("storage-note").hidden = false;
    return;
  }
  const data = storage.get<{ conversations?: Conversation[] }>(STORE_KEY);
  if (data && Array.isArray(data.conversations)) {
    state.conversations = data.conversations.filter((c) => c && typeof c.id === "string" && Array.isArray(c.messages));
  }
}

function prefs(): Prefs {
  return storage.get<Prefs>(PREF_KEY) ?? {};
}

function savePrefs(update: Prefs): void {
  if (storage.ok) storage.set(PREF_KEY, { ...prefs(), ...update });
}

function uid(): string {
  const a = new Uint8Array(8);
  crypto.getRandomValues(a);
  return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
}

// ------------------------------------------------------------------ collections

function collectionById(id: string): CollectionMeta | null {
  return state.manifest?.collections.find((c) => c.id === id) ?? null;
}

function selectedCollection(): string {
  return state.collection;
}

function groundedOnly(): boolean {
  return $<HTMLInputElement>("grounded-only").checked;
}

function collectionItem(c: CollectionMeta | null, extra: { meta?: string; current?: boolean; checked?: boolean }): HTMLButtonElement {
  const b = el("button", "rail-item");
  b.type = "button";
  const ic = el("span", "rail-item-icon");
  ic.appendChild(c ? collectionIcon(c.id) : icon([...ICONS.none], 20));
  b.appendChild(ic);
  const text = el("span", "rail-item-text");
  text.appendChild(el("span", "rail-item-name", c ? shortName(c.id, c.label) : "No collection"));
  text.appendChild(el("span", "rail-item-meta", extra.meta ?? (c ? `${c.sources.length} doc${c.sources.length === 1 ? "" : "s"} · ${c.count.toLocaleString()} passages` : "general chat · PDFs still searched")));
  b.appendChild(text);
  if (extra.current) b.setAttribute("aria-current", "true");
  if (extra.checked !== undefined) b.setAttribute("aria-checked", extra.checked ? "true" : "false");
  return b;
}

function renderRailCollections(): void {
  const list = $("rail-collections");
  list.textContent = "";
  for (const c of state.manifest?.collections ?? []) {
    const li = el("li");
    const b = collectionItem(c, { current: c.id === selectedCollection() });
    b.title = c.label;
    b.addEventListener("click", () => {
      chooseCollection(c.id);
      closeRailDrawer();
    });
    li.appendChild(b);
    list.appendChild(li);
  }
}

function renderTiles(): void {
  const tiles = $("tiles");
  tiles.textContent = "";
  (state.manifest?.collections ?? []).forEach((c, i) => {
    const li = el("li");
    const b = el("button", "tile");
    b.type = "button";
    b.style.setProperty("--i", String(i));
    b.setAttribute("aria-pressed", c.id === selectedCollection() ? "true" : "false");
    const ic = el("span", "tile-icon");
    ic.appendChild(collectionIcon(c.id, 24));
    b.appendChild(ic);
    b.appendChild(el("span", "tile-name", shortName(c.id, c.label)));
    b.appendChild(el("span", "tile-meta", `${c.sources.length} doc${c.sources.length === 1 ? "" : "s"} · ${c.count.toLocaleString()} passages`));
    b.addEventListener("click", () => chooseCollection(c.id));
    li.appendChild(b);
    tiles.appendChild(li);
  });
}

function renderSuggestions(animated: boolean): void {
  const list = $("suggestion-list");
  const key = selectedCollection() || "general";
  // the same questions are already on screen: keep the nodes (a re-render would be a new paint)
  if (list.dataset.for === key && list.childElementCount) return;
  list.dataset.for = key;
  const fill = (): void => {
    list.textContent = "";
    for (const text of suggestionsFor(selectedCollection() || null)) {
      const li = el("li");
      const b = el("button", "chip", text);
      b.type = "button";
      b.addEventListener("click", () => {
        const box = $<HTMLTextAreaElement>("composer-input");
        box.value = text;
        autosize();
        box.focus();
      });
      li.appendChild(b);
      list.appendChild(li);
    }
  };
  if (!animated || reducedMotion()) {
    fill();
    return;
  }
  list.classList.add("is-swapping");
  setTimeout(() => {
    fill();
    list.classList.remove("is-swapping");
  }, 180);
}

function updatePill(): void {
  const c = collectionById(selectedCollection());
  const label = c ? shortName(c.id, c.label) : "No collection";
  setText($("pill-label"), label);
  setText($("topbar-collection-label"), label);
  const ic = $("pill-icon");
  ic.textContent = "";
  ic.appendChild(c ? collectionIcon(c.id, 18) : icon([...ICONS.none], 18));
  setText($("notice"), c?.notice ?? "");
  $("collection-pill").setAttribute("aria-label", `Collection: ${label}. Change collection`);
  $("topbar-collection").setAttribute("aria-label", `Collection: ${label}. Change collection`);
}

function chooseCollection(id: string, persist = true): void {
  if (state.streaming) return;
  const changed = id !== state.collection;
  state.collection = collectionById(id) ? id : "";
  const st = indexStatus.get(state.collection);
  $("collection-pill").dataset.status = !state.collection ? "none" : st && st.pct >= 100 ? "ready" : "loading";
  updatePill();
  renderRailCollections();
  for (const t of $("tiles").querySelectorAll<HTMLElement>(".tile")) {
    const name = t.querySelector(".tile-name")?.textContent;
    const c = collectionById(state.collection);
    t.setAttribute("aria-pressed", c && name === shortName(c.id, c.label) ? "true" : "false");
  }
  renderSuggestions(changed);
  updateTitle();
  const chosen = collectionById(state.collection);
  if (chosen) void preloadCollection(chosen);
  const c = current();
  if (c) {
    c.collection = state.collection;
    saveAll();
  }
  if (persist) savePrefs({ collection: state.collection });
}

const indexStatus = new Map<string, { text: string; pct: number }>();

function setIndexStatus(id: string, text: string, pct: number): void {
  indexStatus.set(id, { text, pct });
  if (id === selectedCollection()) {
    if (state.modelReady) setText($("hero-status"), text);
    $("collection-pill").dataset.status = pct >= 100 ? "ready" : "loading";
    $("collection-pill").title = text;
  }
  renderPopoverStatus();
}

async function preloadCollection(meta: CollectionMeta): Promise<void> {
  const a = state.assistant;
  if (!a) return;
  const mb = (meta.files.passages.gzip_bytes + meta.files.vectors.gzip_bytes) / 1e6;
  if (!a.store.isLoaded(meta.id)) setIndexStatus(meta.id, `Loading ${shortName(meta.id, meta.label)} (about ${mb.toFixed(1)} MB)…`, 0);
  try {
    await a.store.load(meta.id, (loaded, total) => {
      setIndexStatus(meta.id, `Loading ${shortName(meta.id, meta.label)}… ${total ? Math.round((100 * loaded) / total) : 0}%`, total ? (100 * loaded) / total : 0);
    });
    setIndexStatus(meta.id, `${shortName(meta.id, meta.label)}: ${meta.count.toLocaleString()} passages ready.`, 100);
  } catch (err) {
    setIndexStatus(meta.id, `${shortName(meta.id, meta.label)} could not be loaded: ${err instanceof Error ? err.message : String(err)}`, 0);
  }
}

// ------------------------------------------------------------------ collection popover

function openPopover(opener: HTMLElement): void {
  const pop = $("collection-popover");
  state.popoverOpener = opener;
  pop.textContent = "";
  pop.appendChild(el("p", "eyebrow popover-title", "Collection"));
  const options: (CollectionMeta | null)[] = [...(state.manifest?.collections ?? []), null];
  for (const c of options) {
    const b = collectionItem(c, { checked: (c?.id ?? "") === selectedCollection() });
    b.setAttribute("role", "radio");
    if (c) {
      b.dataset.id = c.id;
      const st = indexStatus.get(c.id);
      if (st && st.pct < 100) {
        const p = el("progress");
        p.max = 100;
        p.value = st.pct;
        p.setAttribute("aria-label", st.text);
        b.querySelector(".rail-item-text")?.appendChild(p);
      }
    }
    b.addEventListener("click", () => {
      chooseCollection(c?.id ?? "");
      closePopover();
    });
    pop.appendChild(b);
  }
  pop.setAttribute("role", "dialog");
  pop.hidden = false;
  const r = opener.getBoundingClientRect();
  const w = pop.offsetWidth;
  const h = pop.offsetHeight;
  const below = r.top < window.innerHeight / 2;
  const left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8));
  const top = below ? r.bottom + 8 : r.top - h - 8;
  pop.style.left = `${left}px`;
  pop.style.top = `${Math.max(8, top)}px`;
  pop.style.transformOrigin = `${r.left + r.width / 2 - left}px ${below ? 0 : h}px`;
  opener.setAttribute("aria-expanded", "true");
  if (reducedMotion()) fade(pop, 0, 1, 150);
  else springTo(pop, { transform: ["scale(0.92)", "scale(1)"], opacity: [0, 1] });
  (pop.querySelector<HTMLElement>('[aria-checked="true"]') ?? pop.querySelector<HTMLElement>("button"))?.focus();
}

function renderPopoverStatus(): void {
  const pop = $("collection-popover");
  if (pop.hidden) return;
  for (const b of pop.querySelectorAll<HTMLElement>("[data-id]")) {
    const st = indexStatus.get(b.dataset.id ?? "");
    const p = b.querySelector("progress");
    if (st && st.pct < 100) {
      if (p) p.value = st.pct;
    } else p?.remove();
  }
}

function closePopover(): boolean {
  const pop = $("collection-popover");
  if (pop.hidden) return false;
  const opener = state.popoverOpener;
  opener?.setAttribute("aria-expanded", "false");
  const done = (): void => {
    pop.hidden = true;
  };
  if (reducedMotion()) fade(pop, 1, 0, 150, done);
  else springTo(pop, { transform: ["scale(1)", "scale(0.96)"], opacity: [1, 0] }, { onComplete: done });
  if (opener && document.body.contains(opener)) opener.focus();
  state.popoverOpener = null;
  return true;
}

// ------------------------------------------------------------------ mode control

function placeSegIndicator(animated: boolean): void {
  const control = $("mode-control");
  const checked = control.querySelector<HTMLInputElement>(".seg-input:checked");
  const label = checked ? control.querySelector<HTMLElement>(`label[for="${checked.id}"]`) : null;
  const ind = $("seg-indicator");
  if (!label) return;
  const x = label.offsetLeft;
  ind.style.width = `${label.offsetWidth}px`;
  const m = /translateX\(([-\d.]+)px\)/.exec(ind.style.transform);
  const from = m ? Number.parseFloat(m[1] ?? "0") : x;
  if (!animated) ind.style.transform = `translateX(${x}px)`;
  else springTransform(ind, "translateX", from, x);
}

function updateTitle(): void {
  const c = current();
  $("chat-title").textContent = c?.title || TITLE;
  document.title = c?.title ? `${c.title} · ${TITLE}` : TITLE;
}

// ------------------------------------------------------------------ rail

function openRailDrawer(): void {
  const rail = $("rail");
  const scrim = $("scrim");
  scrim.hidden = false;
  fade(scrim, Number(getComputedStyle(scrim).opacity) || 0, 1, 150);
  rail.classList.add("open");
  $("open-rail").setAttribute("aria-expanded", "true");
  const w = rail.getBoundingClientRect().width;
  const m = /matrix\(([^)]+)\)/.exec(getComputedStyle(rail).transform);
  const cur = m ? Number.parseFloat((m[1] ?? "").split(",")[4] ?? "0") : -w;
  springTransform(rail, "translateX", cur, 0);
  $("new-chat").focus({ preventScroll: true });
}

function closeRailDrawer(): void {
  const rail = $("rail");
  if (!rail.classList.contains("open")) return;
  rail.classList.remove("open");
  $("open-rail").setAttribute("aria-expanded", "false");
  const scrim = $("scrim");
  if (!evidence.open) {
    fade(scrim, 1, 0, 150, () => {
      if (!rail.classList.contains("open") && !evidence.open) scrim.hidden = true;
    });
  }
  const w = rail.getBoundingClientRect().width;
  const m = /matrix\(([^)]+)\)/.exec(getComputedStyle(rail).transform);
  const cur = m ? Number.parseFloat((m[1] ?? "").split(",")[4] ?? "0") : 0;
  springTransform(rail, "translateX", cur, -w);
  if (document.activeElement && rail.contains(document.activeElement)) $("open-rail").focus();
}

function setRailCollapsed(collapsed: boolean, persist = true): void {
  $("app").dataset.rail = collapsed ? "collapsed" : "open";
  const b = $("rail-collapse");
  b.setAttribute("aria-expanded", collapsed ? "false" : "true");
  b.setAttribute("aria-label", collapsed ? "Expand the sidebar" : "Collapse the sidebar");
  if (persist) savePrefs({ railCollapsed: collapsed });
}

// ------------------------------------------------------------------ history

function groupTitle(updated: number): string {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return updated >= d.getTime() ? "Today" : "Earlier";
}

function renderConversationList(): void {
  const host = $("conv-list");
  host.textContent = "";
  const convs = [...state.conversations].sort((a, b) => (b.updated || 0) - (a.updated || 0));
  $("conv-empty").hidden = convs.length > 0;
  let group = "";
  let ul: HTMLUListElement | null = null;
  for (const c of convs) {
    const g = groupTitle(c.updated || 0);
    if (g !== group) {
      group = g;
      host.appendChild(el("p", "conv-group-title", g));
      ul = el("ul", "conv-list");
      host.appendChild(ul);
    }
    const li = el("li", "conv-item");
    if (c.id === state.currentId) li.setAttribute("aria-current", "true");
    const open = el("button", "conv-open", c.title || "New chat");
    open.type = "button";
    open.title = c.title || "New chat";
    open.addEventListener("click", () => {
      openConversation(c.id);
      closeRailDrawer();
    });
    const del = el("button", "conv-del");
    del.type = "button";
    del.setAttribute("aria-label", `Delete chat: ${c.title || "New chat"}`);
    del.appendChild(icon([...ICONS.trash], 16));
    del.addEventListener("click", () => deleteConversation(c.id));
    li.append(open, del);
    ul?.appendChild(li);
  }
}

function newConversation(): void {
  if (state.streaming) stopStreaming();
  state.currentId = null;
  evidence.hide();
  $<HTMLInputElement>("mode-chat").checked = true;
  placeSegIndicator(true);
  renderMessages();
  renderConversationList();
  updateTitle();
  $("composer-input").focus();
}

function ensureConversation(): Conversation {
  let c = current();
  if (!c) {
    c = {
      id: uid(),
      title: "",
      created: Date.now(),
      updated: Date.now(),
      collection: selectedCollection(),
      groundedOnly: groundedOnly(),
      summary: null,
      messages: [],
    };
    state.conversations.unshift(c);
    state.currentId = c.id;
    state.conversations.length = Math.min(state.conversations.length, MAX_CONVERSATIONS);
  }
  return c;
}

function openConversation(id: string): void {
  if (state.streaming) stopStreaming();
  state.currentId = id;
  evidence.hide();
  const c = current();
  if (c) {
    chooseCollection(c.collection || "", false);
    $<HTMLInputElement>(c.groundedOnly ? "grounded-only" : "mode-chat").checked = true;
    placeSegIndicator(true);
  }
  updateTitle();
  renderConversationList();
  renderMessages();
  scrollToBottom(true);
}

function deleteConversation(id: string): void {
  const idx = state.conversations.findIndex((c) => c.id === id);
  const removed = state.conversations[idx];
  if (!removed) return;
  if (state.streaming?.conv.id === id) stopStreaming();
  state.conversations.splice(idx, 1);
  const wasCurrent = state.currentId === id;
  if (wasCurrent) {
    state.currentId = null;
    evidence.hide();
  }
  saveAll();
  renderConversationList();
  renderMessages();
  updateTitle();
  toast("Chat deleted", {
    label: "Undo",
    onClick: () => {
      state.conversations.splice(Math.min(idx, state.conversations.length), 0, removed);
      if (wasCurrent) state.currentId = removed.id;
      saveAll();
      renderConversationList();
      renderMessages();
      updateTitle();
    },
  });
}

// ------------------------------------------------------------------ messages

function announce(text: string): void {
  const live = $("live");
  live.textContent = "";
  setTimeout(() => {
    live.textContent = text;
  }, 30);
}

function renderMessages(): void {
  const list = $("messages");
  list.textContent = "";
  const c = current();
  const msgs = c?.messages ?? [];
  $("hero").hidden = msgs.length > 0;
  $("evidence-toggle").hidden = !msgs.some((m) => m.meta?.passages?.length);
  if (c) msgs.forEach((m, i) => list.appendChild(messageNode(c, m, i)));
  if (state.evidenceFor && !msgs.includes(state.evidenceFor)) {
    state.evidenceFor = null;
    evidence.hide();
  }
}

function lastIndex(c: Conversation, role: Message["role"]): number {
  for (let i = c.messages.length - 1; i >= 0; i--) if (c.messages[i]?.role === role) return i;
  return -1;
}

function actionButton(label: string, aria: string, iconPaths: readonly string[] | null, onClick: (b: HTMLButtonElement) => void): HTMLButtonElement {
  const b = el("button", "act");
  b.type = "button";
  if (iconPaths) b.appendChild(icon([...iconPaths], 14));
  b.appendChild(el("span", null, label));
  b.setAttribute("aria-label", aria);
  b.addEventListener("click", () => onClick(b));
  return b;
}

function messageNode(c: Conversation, m: Message, index: number): HTMLElement {
  const li = el("li", `msg ${m.role}`);
  m.node = li;
  if (m.role === "user") {
    li.appendChild(el("span", "visually-hidden", "You said:"));
    li.appendChild(el("div", "bubble-user", m.content));
    if (index === lastIndex(c, "user") && !state.streaming) {
      const actions = el("div", "msg-actions");
      actions.appendChild(actionButton("Edit", "Edit and resend your last message", null, () => startEdit(c, m, li)));
      actions.appendChild(actionButton("Copy", "Copy your message", ICONS.copy, (b) => copyText(m.content, b.querySelector("span") ?? b)));
      li.appendChild(actions);
    }
    return li;
  }
  li.appendChild(el("span", "visually-hidden", "Veridex answered:"));
  li.appendChild(el("div", "trace-strip"));
  li.appendChild(el("div", "answer"));
  renderAnswer(c, m, li, true);
  return li;
}

function passageMap(m: Message): Map<string, Passage> {
  const map = new Map<string, Passage>();
  for (const p of m.meta?.context ?? []) map.set(p.label, p);
  for (const p of m.meta?.passages ?? []) map.set(p.label, p);
  return map;
}

function citeFactory(m: Message): (label: string) => HTMLElement | null {
  const map = passageMap(m);
  return (label) => {
    const p = map.get(label);
    if (!p) return null;
    const b = el("button", "cite-inline", label);
    b.type = "button";
    b.dataset.label = label;
    b.setAttribute("aria-label", `Source ${label}: ${p.title || p.heading_path}`);
    b.addEventListener("mouseenter", () => linkSource(m, label, true));
    b.addEventListener("mouseleave", () => linkSource(m, null, false));
    b.addEventListener("focus", () => linkSource(m, label, true));
    b.addEventListener("blur", () => linkSource(m, null, false));
    b.addEventListener("click", () => openSource(m, label, b));
    return b;
  };
}

/** Hovering or focusing a citation highlights its source card (and the matching chips). */
function linkSource(m: Message, label: string | null, scroll: boolean): void {
  if (state.evidenceFor === m && evidence.open) evidence.link(label, scroll);
  if (m.node) linkLabels(m.node, label);
}

function openSource(m: Message, label: string, opener: HTMLElement): void {
  showEvidence(m, opener, label);
}

function questionFor(c: Conversation, m: Message): string {
  const idx = c.messages.indexOf(m);
  for (let i = idx - 1; i >= 0; i--) {
    const q = c.messages[i];
    if (q?.role === "user") return q.content;
  }
  return "";
}

function verdictOf(meta: Meta): { verdict: Verdict; supported?: number; total?: number } | null {
  if (meta.mode === "grounded_only") {
    if (meta.outcome === "abstained") return { verdict: "unsupported" };
    if (meta.outcome === "answered") {
      const ok = meta.check?.critic?.verdict === "accept" && meta.check.guards?.passed;
      return { verdict: ok ? "supported" : "none" };
    }
    return null;
  }
  if (meta.verify && !("error" in meta.verify)) {
    const { supported, total } = meta.verify;
    if (!total) return { verdict: "none", supported, total };
    return { verdict: supported === total ? "supported" : supported > 0 ? "partial" : "unsupported", supported, total };
  }
  if (meta.grounded) return { verdict: "none" };
  return null;
}

function traceFor(c: Conversation, m: Message): TraceStep[] {
  const meta = m.meta as Meta;
  const base = meta.trace ?? initialTrace(c.groundedOnly, Boolean(meta.collection) || Boolean(state.assistant?.uploads.docs.length));
  if (meta.mode !== "grounded_only" && meta.grounded) {
    const v = verdictOf(meta);
    const hasVerify = base.some((s) => s.id === "verify");
    const steps = hasVerify ? base : [...base, { id: "verify", label: "Verify", status: "pending" as const, ms: null }];
    return steps.map((s) => (s.id === "verify" && v && v.verdict !== "none" ? { ...s, status: "done" as const, ms: meta.verifyMs ?? s.ms, verdict: v.verdict } : s));
  }
  return base.filter((s) => s.id !== "verify");
}

let verifying: Message | null = null;

function renderAnswer(c: Conversation, m: Message, li: HTMLElement, final: boolean): void {
  const meta = m.meta ?? { status: "done", collection: "", notice: "" };
  m.meta ??= meta;
  const body = li.querySelector<HTMLElement>(".answer");
  const strip = li.querySelector<HTMLElement>(".trace-strip");
  if (!body || !strip) return;
  // the trace strip (compact)
  const showTrace = meta.status === "streaming" || meta.grounded || meta.mode === "grounded_only";
  strip.textContent = "";
  if (showTrace) {
    const t = el("div");
    renderTrace(t, traceFor(c, m), { onVerify: meta.grounded && meta.mode !== "grounded_only" && !meta.verify ? () => void verify(c, m) : undefined, verifying: verifying === m });
    strip.appendChild(t);
  }
  body.textContent = "";
  body.classList.toggle("streaming", meta.status === "streaming");
  if (meta.status === "streaming" && !m.content) {
    const t = el("div", "thinking");
    t.setAttribute("aria-label", "Veridex is working");
    t.appendChild(el("span", "caret"));
    t.appendChild(el("span", null, meta.statusText || "Reading the documents…"));
    body.appendChild(t);
  } else if (m.content) {
    body.appendChild(render(m.content, { cite: citeFactory(m), highlight: final }));
    if (meta.status === "streaming") decorateStream(body);
    else if (final) void loadVendorsFor(c, m, li);
  }
  for (const n of li.querySelectorAll(".source-row, .notice-line, .error-card, .abstain-card, .msg-actions, .verify, .status-line.after")) n.remove();
  if (meta.status === "streaming") {
    if (meta.context?.length && m.content) {
      li.appendChild(el("div", "status-line after", `Using ${meta.context.length} passage${meta.context.length === 1 ? "" : "s"} from the documents…`));
    }
    return;
  }
  if (meta.mode === "grounded_only" && meta.outcome === "abstained") li.appendChild(abstainCard(meta));
  if (meta.sourceLine && !(meta.mode === "grounded_only" && meta.outcome === "abstained")) li.appendChild(sourceRow(c, m));
  if (meta.notice && meta.collection && meta.status !== "error") li.appendChild(el("div", "notice-line", meta.notice));
  if (meta.error) li.appendChild(errorCard(c, meta.error));
  if (meta.verify) li.appendChild(verifyNode(meta.verify));
  if (!meta.error) li.appendChild(answerActions(c, m));
}

/** KaTeX and highlight.js only when the answer needs them; re-render once they arrive. */
async function loadVendorsFor(c: Conversation, m: Message, li: HTMLElement): Promise<void> {
  const wantMath = needsMath(m.content) && !window.katex;
  const wantCode = needsCode(m.content) && !window.hljs;
  if (!wantMath && !wantCode) return;
  try {
    await Promise.all([wantMath ? ensureKatex(BASE) : null, wantCode ? ensureHljs(BASE) : null]);
  } catch {
    return;
  }
  if (m.node === li && document.body.contains(li) && m.meta?.status !== "streaming") renderAnswer(c, m, li, true);
}

/** The tail of the streamed text rises in: one span per delta still younger than 120 ms, plus the caret. */
function decorateStream(body: HTMLElement): void {
  const s = state.streaming;
  const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT);
  let last: Text | null = null;
  while (walker.nextNode()) {
    const t = walker.currentNode as Text;
    if (t.textContent?.trim() && !t.parentElement?.closest("pre")) last = t;
  }
  if (s && last && !reducedMotion()) {
    const now = performance.now();
    const recent = s.deltas.filter((d) => now - d.t < 120);
    let node: Text = last;
    for (let i = recent.length - 1; i >= 0; i--) {
      const d = recent[i] as { t: number; n: number };
      const text = node.textContent ?? "";
      if (!text.length) break;
      const n = Math.min(d.n, text.length);
      const tail = node.splitText(text.length - n);
      const span = el("span", "tok");
      span.style.animationDelay = `-${Math.round(now - d.t)}ms`;
      tail.parentNode?.insertBefore(span, tail);
      span.appendChild(tail);
      if (n < d.n || !node.textContent?.length) break;
    }
  }
  const caret = el("span", "caret");
  caret.setAttribute("aria-hidden", "true");
  const host = last?.parentElement && !last.parentElement.closest("pre") ? last.parentElement : body;
  host.appendChild(caret);
}

function shortHeading(path: string): string {
  const parts = path.split(" > ");
  return parts.length > 2 ? `${parts[0]} › ${parts[parts.length - 1]}` : parts.join(" › ");
}

function docTitle(p: Passage): string {
  if (p.title) return p.title;
  const coll = collectionById(p.source);
  return coll?.sources.find((s) => s.doc_id === p.doc_id)?.title ?? p.heading_path.split(" > ")[0] ?? "Document";
}

function sourceRow(c: Conversation, m: Message): HTMLElement {
  const meta = m.meta as Meta;
  const row = el("div", "source-row");
  const v = verdictOf(meta);
  if (v && meta.passages?.length) {
    const seal = sealRow(m, v, meta.passages.length);
    row.appendChild(seal);
  } else {
    const line = el("div", "source-line general");
    line.appendChild(el("span", "label-text", meta.sourceLine ?? ""));
    row.appendChild(line);
  }
  for (const p of meta.passages ?? []) {
    const b = el("button", "src-chip");
    b.type = "button";
    b.dataset.label = p.label;
    b.appendChild(el("b", null, p.label));
    b.appendChild(document.createTextNode(p.title ? `${p.title} · ${shortHeading(p.heading_path)}` : shortHeading(p.heading_path)));
    b.title = p.heading_path;
    b.setAttribute("aria-label", `Open source ${p.label}: ${p.title ? `${p.title}, ` : ""}${p.heading_path}`);
    b.addEventListener("mouseenter", () => linkSource(m, p.label, true));
    b.addEventListener("mouseleave", () => linkSource(m, null, false));
    b.addEventListener("click", () => openSource(m, p.label, b));
    row.appendChild(b);
  }
  void c;
  return row;
}

function sealRow(m: Message, v: { verdict: Verdict; supported?: number; total?: number }, n: number): HTMLElement {
  const meta = m.meta as Meta;
  const row = el("div", "seal-row");
  row.dataset.verdict = v.verdict;
  row.setAttribute("role", "status");
  const seal = el("span", "seal");
  seal.appendChild(sealSvg({ ripple: true }));
  row.appendChild(seal);
  const text = el("span", "seal-text");
  const { label, detail } = verdictLabel(v.verdict, n, v.supported, v.total);
  text.appendChild(el("span", "seal-label label-text", v.verdict === "none" ? (meta.sourceLine ?? label) : label));
  text.appendChild(el("span", "seal-detail", detail));
  row.appendChild(text);
  const animate = !m.stamped && v.verdict !== "none";
  if (v.verdict !== "none") m.stamped = true;
  stampSeal(seal, v.verdict, animate);
  return row;
}

function abstainCard(meta: Meta): HTMLElement {
  const card = el("section", "abstain-card");
  card.setAttribute("aria-label", "Not enough evidence");
  const coll = collectionById(meta.collection);
  const seal = el("span", "seal");
  seal.appendChild(sealSvg({ ripple: true }));
  const head = el("div", "seal-row");
  head.dataset.verdict = "unsupported";
  head.appendChild(seal);
  const text = el("span", "seal-text");
  text.appendChild(el("span", "abstain-title", `Not enough evidence in ${coll ? shortName(coll.id, coll.label) : "your documents"} to answer this.`));
  text.appendChild(el("span", "seal-detail label-text", meta.abstentionMessage ?? "Not found in these documents"));
  head.appendChild(text);
  card.appendChild(head);
  stampSeal(seal, "unsupported", true);
  if (meta.closest?.length) {
    card.appendChild(el("p", "abstain-text", "Closest sections:"));
    const ul = el("ul", "closest");
    for (const s of meta.closest) ul.appendChild(el("li", null, `${s.heading_path}${s.pages ? ` (p. ${s.pages})` : ""}`));
    card.appendChild(ul);
  }
  const others = (state.manifest?.collections ?? []).filter((c) => c.id !== meta.collection);
  if (others.length) {
    card.appendChild(el("p", "abstain-text", "Try another collection:"));
    const sw = el("div", "abstain-switch");
    for (const c of others) {
      const b = el("button", "pill");
      b.type = "button";
      const ic = el("span", "pill-icon");
      ic.appendChild(collectionIcon(c.id, 16));
      b.appendChild(ic);
      b.appendChild(el("span", null, shortName(c.id, c.label)));
      b.addEventListener("click", () => {
        chooseCollection(c.id);
        $("composer-input").focus();
      });
      sw.appendChild(b);
    }
    card.appendChild(sw);
  }
  return card;
}

const countdowns = new WeakMap<HTMLElement, number>();

function errorCard(c: Conversation, err: StoredError): HTMLElement {
  const card = el("section", "error-card");
  card.setAttribute("role", "alert");
  const title = el("p", "error-box", err.title);
  card.appendChild(title);
  const detail = el("p", "error-detail mono");
  card.appendChild(detail);
  const actions = el("div", "error-actions");
  const retry = el("button", "btn", "Retry");
  retry.type = "button";
  retry.addEventListener("click", () => regenerate(c));
  actions.appendChild(retry);
  card.appendChild(actions);
  const update = (): void => {
    if (err.countdown && err.retryAfter !== null) {
      const left = err.retryAfter - (Date.now() - err.at) / 1000;
      if (left > 0) {
        detail.textContent = `Try again in ${formatCountdown(left)}`;
        return;
      }
      detail.textContent = "You can try again now.";
      const t = countdowns.get(card);
      if (t) clearInterval(t);
      return;
    }
    detail.textContent = err.detail ?? "";
    detail.hidden = !err.detail;
  };
  update();
  if (err.countdown) countdowns.set(card, window.setInterval(update, 1000));
  return card;
}

function answerActions(c: Conversation, m: Message): HTMLElement {
  const actions = el("div", "msg-actions");
  const meta = m.meta;
  actions.appendChild(actionButton("Copy", "Copy the answer", ICONS.copy, (b) => copyText(m.content, b.querySelector("span") ?? b)));
  if (c.messages[c.messages.length - 1] === m && !state.streaming) {
    actions.appendChild(actionButton("Regenerate", "Regenerate the answer", ICONS.refresh, () => regenerate(c)));
  }
  if (meta?.mode !== "grounded_only" && meta?.grounded && meta.passages?.length && !meta.verify) {
    const b = actionButton("Verify", "Verify this answer against its sources", ICONS.verify, () => void verify(c, m));
    b.classList.add("act-verify");
    if (verifying === m) {
      b.disabled = true;
      (b.querySelector("span") as HTMLElement).textContent = "Verifying…";
    }
    actions.appendChild(b);
  }
  actions.appendChild(
    actionButton("Share", "Copy a link that pre-fills this question", ICONS.share, (b) => {
      const url = buildShareUrl(location.origin, BASE, questionFor(c, m), meta?.collection || null);
      copyText(url, b.querySelector("span") ?? b, "Share");
      toast("Link copied. It pre-fills the question; it never sends it.");
    }),
  );
  if (meta?.passages?.length || meta?.mode === "grounded_only") {
    actions.appendChild(actionButton("Show evidence", "Show the evidence panel", ICONS.evidence, (b) => showEvidence(m, b)));
  }
  if (meta?.mode === "grounded_only" && meta.outcome === "answered") actions.appendChild(el("span", "status-line", "Checked by guards and the critic"));
  return actions;
}

function verifyNode(v: VerifyResult | { error: string }): HTMLElement {
  const box = el("section", "verify");
  box.setAttribute("aria-label", "Verification");
  if ("error" in v) {
    box.appendChild(el("p", "error-box", v.error));
    return box;
  }
  box.appendChild(el("h4", null, `Verification: ${v.supported} of ${v.total} sentence${v.total === 1 ? "" : "s"} supported`));
  const ul = el("ul");
  for (const s of v.sentences) {
    const li = el("li", s.status);
    li.appendChild(el("span", "mark", s.status === "supported" ? "Supported" : "Unsupported"));
    const text = el("span", null, s.text);
    if (s.reasons.length) text.appendChild(el("span", "why", s.reasons.join("; ")));
    li.appendChild(text);
    ul.appendChild(li);
  }
  box.appendChild(ul);
  return box;
}

// ------------------------------------------------------------------ evidence panel content

function evidenceContent(c: Conversation, m: Message): DocumentFragment {
  const frag = document.createDocumentFragment();
  const meta = m.meta as Meta;
  frag.appendChild(el("p", "evidence-section-title", "Verification trace"));
  const t = el("div");
  renderTrace(t, traceFor(c, m), { onVerify: meta.grounded && meta.mode !== "grounded_only" && !meta.verify ? () => void verify(c, m) : undefined, verifying: verifying === m });
  frag.appendChild(t);
  const v = verdictOf(meta);
  const passages = meta.passages ?? [];
  if (v) {
    const row = el("div", "seal-row");
    row.dataset.verdict = v.verdict;
    const seal = el("span", "seal");
    seal.appendChild(sealSvg({ ripple: true }));
    row.appendChild(seal);
    const text = el("span", "seal-text");
    const { label, detail } = verdictLabel(v.verdict, passages.length, v.supported, v.total);
    text.appendChild(el("span", "seal-label", v.verdict === "none" ? (meta.sourceLine ?? label) : label));
    text.appendChild(el("span", "seal-detail", meta.check?.detail ? meta.check.detail : detail));
    row.appendChild(text);
    frag.appendChild(row);
    stampSeal(seal, v.verdict, false);
  }
  const title = el("p", "evidence-section-title", passages.length ? `Sources (${passages.length})` : "Sources");
  frag.appendChild(title);
  if (!passages.length) {
    frag.appendChild(el("p", "evidence-empty", meta.mode === "grounded_only" && meta.outcome === "abstained" ? "No passage supported an answer, so none is cited." : "This answer cites no passages."));
    return frag;
  }
  const terms = queryTerms(questionFor(c, m));
  const list = el("div", "source-list");
  for (const p of passages) list.appendChild(sourceCard(m, p, terms));
  frag.appendChild(list);
  return frag;
}

function sourceCard(m: Message, p: Passage, terms: string[]): HTMLElement {
  const card = el("article", "source-card");
  card.dataset.label = p.label;
  const head = el("button", "source-head");
  head.type = "button";
  head.setAttribute("aria-expanded", "false");
  head.appendChild(el("span", "source-label", p.label));
  const meta = el("span", "source-meta");
  meta.appendChild(el("span", "source-title", docTitle(p)));
  const section = p.heading_path.split(" > ").slice(1).join(" › ");
  meta.appendChild(el("span", "source-section", `${section || p.heading_path}${p.pages ? ` · p. ${p.pages}` : ""}`));
  head.appendChild(meta);
  head.appendChild(icon([...ICONS.chevron], 16, { class: "source-chevron" }));
  card.appendChild(head);
  const body = el("div", "source-body");
  const ex = el("div", "source-excerpt");
  ex.appendChild(markTerms(excerpt(p.text, terms), terms));
  body.appendChild(ex);
  card.appendChild(body);
  if (typeof p.score === "number" && Number.isFinite(p.score)) {
    const score = el("div", "source-score");
    const bar = el("div", "score-bar");
    const fill = el("span");
    fill.style.transform = `scaleX(${Math.max(0.02, Math.min(1, p.score))})`;
    bar.appendChild(fill);
    bar.setAttribute("aria-hidden", "true");
    score.appendChild(bar);
    score.appendChild(el("span", "score-num", `cos ${p.score.toFixed(3)}`));
    score.setAttribute("aria-label", `Relevance score ${p.score.toFixed(3)}`);
    card.appendChild(score);
  }
  card.addEventListener("mouseenter", () => linkSource(m, p.label, false));
  card.addEventListener("mouseleave", () => linkSource(m, null, false));
  head.addEventListener("click", () => {
    const open = card.classList.toggle("is-open");
    head.setAttribute("aria-expanded", open ? "true" : "false");
    const before = body.getBoundingClientRect().height;
    ex.textContent = "";
    ex.classList.toggle("source-full", open);
    ex.appendChild(open ? markTerms(p.text, terms) : markTerms(excerpt(p.text, terms), terms));
    const after = body.getBoundingClientRect().height;
    if (!reducedMotion() && before !== after) {
      body.style.height = `${before}px`;
      body.style.overflow = "hidden";
      springTo(body, { height: [`${before}px`, `${after}px`] }, {
        onComplete: () => {
          body.style.height = "";
          body.style.overflow = "";
        },
      });
    }
  });
  return card;
}

function showEvidence(m: Message, opener?: HTMLElement | null, label?: string): void {
  const c = current();
  if (!c) return;
  state.evidenceFor = m;
  evidence.setContent(evidenceContent(c, m));
  evidence.show(opener ?? null, label ? "full" : "half");
  if (label) requestAnimationFrame(() => evidence.openCard(label));
}

function refreshEvidence(): void {
  const c = current();
  const m = state.evidenceFor;
  if (!c || !m || !evidence.open) return;
  evidence.setContent(evidenceContent(c, m));
}

// ------------------------------------------------------------------ scrolling

function nearBottom(): boolean {
  const s = $("scroller");
  return s.scrollHeight - s.scrollTop - s.clientHeight < 80;
}

function scrollToBottom(force: boolean): void {
  const s = $("scroller");
  if (force || state.autoScroll) {
    s.scrollTop = s.scrollHeight;
    state.autoScroll = true;
    $("jump").hidden = true;
  }
}

// ------------------------------------------------------------------ sending

function historyFor(c: Conversation): Turn[] {
  const out: Turn[] = [];
  for (const m of c.messages) {
    const meta = m.meta;
    if (m.role === "assistant" && (!m.content || meta?.error || meta?.status === "streaming")) continue;
    const content = m.role === "assistant" ? m.content.replace(/[ \t]*\[C\d{1,3}\]/g, "") : m.content;
    if (content.trim()) out.push({ role: m.role, content: content.slice(0, 4000) });
  }
  return out.slice(-40);
}

function setBusy(busy: boolean): void {
  $("stop-btn").hidden = !busy;
  $("send-btn").hidden = busy;
  $<HTMLButtonElement>("stop-btn").disabled = false;
  $<HTMLButtonElement>("collection-pill").disabled = busy;
  $<HTMLButtonElement>("topbar-collection").disabled = busy;
  for (const t of $("tiles").querySelectorAll<HTMLButtonElement>(".tile")) t.disabled = busy;
  $("key-hint").textContent = busy ? "Esc to stop" : "Enter to send · Shift+Enter for a new line";
}

let renderPending = false;
function scheduleRender(c: Conversation, m: Message): void {
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(() => {
    renderPending = false;
    if (m.node) renderAnswer(c, m, m.node, false);
    scrollToBottom(false);
  });
}

async function sendMessage(text: string): Promise<void> {
  const value = text.trim();
  if (!value || state.streaming) return;
  const c = ensureConversation();
  c.collection = selectedCollection();
  c.groundedOnly = groundedOnly();
  // a failed exchange is replaced by the resend: the question was kept in the composer
  const last = c.messages[c.messages.length - 1];
  if (last?.role === "assistant" && last.meta?.error) {
    c.messages.pop();
    if (c.messages[c.messages.length - 1]?.role === "user") c.messages.pop();
  }
  if (!c.title) {
    c.title = value.replace(/\s+/g, " ").slice(0, 60);
    updateTitle();
  }
  c.messages.push({ role: "user", content: value });
  c.updated = Date.now();
  await runAssistant(c);
}

async function runAssistant(c: Conversation): Promise<void> {
  const a = state.assistant;
  const coll = collectionById(c.collection);
  const hasDocs = Boolean(coll) || Boolean(a?.uploads.docs.length);
  const m: Message = {
    role: "assistant",
    content: "",
    meta: {
      status: "streaming",
      collection: c.collection,
      notice: coll?.notice ?? "",
      statusText: !state.modelReady ? "Will send when the search index is ready…" : c.groundedOnly ? "Searching and checking the documents…" : hasDocs ? "Reading the documents…" : "Thinking…",
      trace: initialTrace(c.groundedOnly, hasDocs),
    },
  };
  c.messages.push(m);
  saveAll();
  renderConversationList();
  renderMessages();
  scrollToBottom(true);
  const controller = new AbortController();
  state.streaming = { controller, conv: c, msg: m, deltas: [] };
  setBusy(true);
  if (!state.modelReady) setText($("notice"), "Will send when ready");
  announce("Veridex is answering.");
  const meta = m.meta as Meta;
  try {
    if (!a) throw new ProxyError("loading", "The page is still loading.", 0);
    const turns = historyFor(c);
    const out: ChatOutcome = await a.chat(
      { turns, collection: c.collection || null, groundedOnly: c.groundedOnly, summary: c.summary },
      {
        onContext: (grounded, passages) => {
          meta.context = passages;
          if (grounded) meta.statusText = `Found ${passages.length} matching passage${passages.length === 1 ? "" : "s"}…`;
          else if (hasDocs) meta.statusText = "No passage matched; answering from general knowledge…";
          scheduleRender(c, m);
        },
        onToken: (t) => {
          m.content += t;
          state.streaming?.deltas.push({ t: performance.now(), n: t.length });
          if (state.streaming && state.streaming.deltas.length > 40) state.streaming.deltas.splice(0, state.streaming.deltas.length - 40);
          scheduleRender(c, m);
        },
        onStatus: (s) => {
          meta.statusText = s;
          scheduleRender(c, m);
        },
        onIndexProgress: (loaded, total) => {
          meta.statusText = `Loading the documents… ${total ? Math.round((100 * loaded) / total) : 0}%`;
          scheduleRender(c, m);
        },
        onStage: (e: StageEvent) => {
          meta.trace = applyStage(meta.trace ?? [], e);
          scheduleRender(c, m);
        },
      },
      controller.signal,
    );
    m.content = out.answer;
    Object.assign(meta, {
      mode: out.mode,
      grounded: out.grounded,
      sourceLine: out.stopped && !out.answer ? "" : out.sourceLine,
      citations: out.citations,
      passages: out.passages,
      outcome: out.outcome,
      closest: out.closest,
      abstentionMessage: out.abstentionMessage,
      check: out.check,
    });
    if (out.summary) c.summary = out.summary;
    meta.status = out.stopped ? "stopped" : "done";
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") meta.status = "stopped";
    else {
      meta.status = "error";
      meta.error = { ...describeError(err), at: Date.now() };
      const box = $<HTMLTextAreaElement>("composer-input");
      if (!box.value.trim()) {
        box.value = questionFor(c, m);
        autosize();
      }
    }
  }
  meta.trace = finishTrace(meta.trace ?? []);
  finishMessage(c, m);
}

function finishMessage(c: Conversation, m: Message): void {
  const meta = m.meta as Meta;
  delete meta.statusText;
  state.streaming = null;
  setBusy(false);
  setText($("notice"), collectionById(selectedCollection())?.notice ?? "");
  c.updated = Date.now();
  saveAll();
  renderMessages();
  renderConversationList();
  scrollToBottom(false);
  const v = verdictOf(meta);
  const verdictText = v && v.verdict !== "none" ? verdictLabel(v.verdict, meta.passages?.length ?? 0, v.supported, v.total).label : (meta.sourceLine ?? "");
  announce(meta.error ? `${meta.error.title} ${meta.error.detail ?? ""}` : `Answer finished. ${verdictText}`);
  if (meta.passages?.length && evidence.mode() === "panel" && !evidence.open && state.evidenceFor === null && !meta.error) {
    // desktop: the evidence slides in on its own for the first cited answer
    showEvidence(m, null);
  } else if (state.evidenceFor && evidence.open) refreshEvidence();
}

function stopStreaming(): void {
  state.streaming?.controller.abort();
}

function regenerate(c: Conversation): void {
  if (state.streaming) return;
  if (c.messages[c.messages.length - 1]?.role === "assistant") c.messages.pop();
  if (c.messages[c.messages.length - 1]?.role !== "user") return;
  c.collection = selectedCollection();
  c.groundedOnly = groundedOnly();
  void runAssistant(c);
}

function startEdit(c: Conversation, m: Message, li: HTMLElement): void {
  const box = el("div", "edit-box");
  const area = el("textarea");
  area.id = `edit-${uid()}`;
  const label = el("label", "visually-hidden", "Edit your message");
  label.htmlFor = area.id;
  area.value = m.content;
  area.maxLength = 4000;
  const row = el("div", "edit-row");
  const cancel = el("button", "btn", "Cancel");
  cancel.type = "button";
  const save = el("button", "btn btn-primary", "Save and send");
  save.type = "button";
  row.append(cancel, save);
  box.append(label, area, row);
  li.textContent = "";
  li.appendChild(box);
  area.focus();
  cancel.addEventListener("click", () => renderMessages());
  const submit = (): void => {
    const v = area.value.trim();
    if (!v) return;
    c.messages.length = c.messages.indexOf(m);
    c.messages.push({ role: "user", content: v });
    c.collection = selectedCollection();
    c.groundedOnly = groundedOnly();
    void runAssistant(c);
  };
  save.addEventListener("click", submit);
  area.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    } else if (e.key === "Escape") renderMessages();
  });
}

async function verify(c: Conversation, m: Message): Promise<void> {
  const a = state.assistant;
  const meta = m.meta;
  if (!a || !meta || verifying) return;
  verifying = m;
  if (m.node) renderAnswer(c, m, m.node, true);
  refreshEvidence();
  const t0 = performance.now();
  try {
    meta.verify = await a.verify(questionFor(c, m) || "(question)", m.content, meta.passages ?? [], meta.collection || null);
  } catch (err) {
    meta.verify = { error: err instanceof Error ? err.message : "The verification failed. Try again." };
  }
  meta.verifyMs = performance.now() - t0;
  verifying = null;
  m.stamped = false;
  saveAll();
  if (m.node) renderAnswer(c, m, m.node, true);
  refreshEvidence();
  const v = meta.verify;
  announce("error" in v ? v.error : `Verification finished: ${v.supported} of ${v.total} sentences supported.`);
}

// ------------------------------------------------------------------ uploads

function renderUploads(): void {
  const list = $("upload-list");
  list.textContent = "";
  const store = state.assistant?.uploads;
  for (const d of store?.docs ?? []) {
    const li = el("li", "upload-item");
    li.appendChild(el("span", "name", d.name));
    li.appendChild(el("span", "meta", `${d.pages} p.`));
    const del = el("button", "icon-btn");
    del.type = "button";
    del.setAttribute("aria-label", `Remove ${d.name}`);
    del.appendChild(icon([...ICONS.close], 16));
    del.addEventListener("click", () => {
      store?.remove(d.docId);
      uploadStatus(`${d.name} removed.`, null, false);
      renderUploads();
    });
    li.appendChild(del);
    list.appendChild(li);
  }
  const max = state.manifest?.uploads.max_files ?? 3;
  const full = state.uploading || !store || store.docs.length >= max;
  $<HTMLButtonElement>("upload-btn").disabled = full;
  $<HTMLButtonElement>("attach-btn").disabled = full;
}

function uploadStatus(text: string, pct: number | null, isError: boolean): void {
  $("upload-status").classList.toggle("error", isError);
  $("upload-text").textContent = text;
  const bar = $<HTMLProgressElement>("upload-bar");
  bar.hidden = pct === null;
  if (pct !== null) bar.value = Math.max(0, Math.min(100, pct));
}

async function uploadFile(file: File): Promise<void> {
  const a = state.assistant;
  if (!a) return;
  state.uploading = true;
  renderUploads();
  uploadStatus(`Reading ${file.name}…`, 0, false);
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    await a.embedder.ready;
    const doc = await a.uploads.add({ name: file.name, bytes }, a.embedder, (stage, done, total) => {
      const label = stage === "reading" ? "Reading pages" : "Indexing passages";
      uploadStatus(`${label} ${done}/${total}…`, total ? (100 * done) / total : null, false);
    });
    uploadStatus(`${doc.name} is ready (${doc.source.chunks.length} passages). Ask about it.`, null, false);
    toast(`${doc.name} is ready to search.`);
    announce(`${doc.name} is ready.`);
  } catch (err) {
    uploadStatus(err instanceof UploadError ? err.message : `The PDF could not be processed${err instanceof Error ? `: ${err.message}` : ""}.`, null, true);
  }
  state.uploading = false;
  renderUploads();
}

// ------------------------------------------------------------------ about

function renderAbout(): void {
  const root = $("about-sources");
  root.textContent = "";
  root.appendChild(el("h3", null, "Document collections"));
  for (const c of state.manifest?.collections ?? []) {
    const sec = el("div", "about-coll");
    const h = el("h3", null, c.label);
    h.appendChild(el("span", "tag", c.evaluated ? "Evaluated" : "Not evaluated"));
    sec.appendChild(h);
    if (c.notice) sec.appendChild(el("p", "rail-hint", c.notice));
    const ul = el("ul", "about-list");
    for (const s of c.sources) {
      const li = el("li");
      li.appendChild(el("strong", null, s.title || "Document"));
      if (s.authors) li.appendChild(document.createTextNode(` — ${s.authors}`));
      if (s.edition) li.appendChild(document.createTextNode(` (${s.edition})`));
      li.appendChild(document.createTextNode(". Licence: "));
      if (s.licence_url && SAFE_URL.test(s.licence_url)) {
        const link = el("a", null, s.licence || s.licence_url);
        link.href = s.licence_url;
        link.rel = "noopener noreferrer";
        link.target = "_blank";
        li.appendChild(link);
      } else li.appendChild(document.createTextNode(s.licence || "as stated in the source"));
      if (s.source_url && SAFE_URL.test(s.source_url)) {
        li.appendChild(document.createTextNode(". "));
        const link = el("a", null, "Source");
        link.href = s.source_url;
        link.rel = "noopener noreferrer";
        link.target = "_blank";
        li.appendChild(link);
      }
      ul.appendChild(li);
    }
    sec.appendChild(ul);
    root.appendChild(sec);
  }
}

// ------------------------------------------------------------------ theme, palette, shortcuts

function toggleTheme(): void {
  const next: Theme = currentTheme() === "light" ? "dark" : "light";
  applyTheme(next);
  savePrefs({ theme: next });
  syncThemeButtons();
}

function syncThemeButtons(): void {
  const label = themeToggleLabel();
  $("theme-toggle").setAttribute("aria-label", label);
  $("theme-toggle-top").setAttribute("aria-label", label);
  $("theme-label").textContent = currentTheme() === "light" ? "Dark theme" : "Light theme";
}

async function openCommandPalette(): Promise<void> {
  const { openPalette } = await import("./ui/palette.ts");
  const items = [];
  for (const c of state.manifest?.collections ?? []) {
    items.push({ id: `coll:${c.id}`, group: "Collections", label: shortName(c.id, c.label), hint: `${c.sources.length} documents · ${c.count.toLocaleString()} passages`, icon: collectionIconPaths(c.id), run: () => chooseCollection(c.id) });
  }
  items.push({ id: "coll:none", group: "Collections", label: "No collection (general chat)", icon: [...ICONS.none], run: () => chooseCollection("") });
  items.push({ id: "new", group: "Actions", label: "New chat", kbd: `${modKey} ⇧ O`, icon: [...ICONS.plus], run: () => newConversation() });
  items.push({ id: "theme", group: "Actions", label: currentTheme() === "light" ? "Switch to the dark theme" : "Switch to the light theme", icon: currentTheme() === "light" ? [...ICONS.moon] : [...ICONS.sun], run: () => toggleTheme() });
  items.push({ id: "how", group: "Actions", label: "How it works", icon: [...ICONS.flow], run: () => void openHowItWorks() });
  items.push({ id: "about", group: "Actions", label: "About & sources", icon: [...ICONS.book], run: () => openAbout() });
  items.push({ id: "shortcuts", group: "Actions", label: "Keyboard shortcuts", icon: [...ICONS.keyboard], run: () => undefined });
  items.push({ id: "repo", group: "Actions", label: "GitHub repository", hint: REPO.replace("https://", ""), icon: [...ICONS.github], run: () => window.open(REPO, "_blank", "noopener,noreferrer") });
  for (const c of [...state.conversations].sort((a, b) => (b.updated || 0) - (a.updated || 0)).slice(0, 30)) {
    items.push({ id: `chat:${c.id}`, group: "History", label: c.title || "New chat", hint: groupTitle(c.updated || 0), icon: [...ICONS.chat], run: () => openConversation(c.id) });
  }
  openPalette(items);
}

function collectionIconPaths(id: string): string[] {
  const svg = collectionIcon(id);
  return [...svg.querySelectorAll("path")].map((p) => p.getAttribute("d") ?? "");
}

async function openHowItWorks(): Promise<void> {
  const { openHow } = await import("./ui/how.ts");
  openHow(REPO);
}

function openAbout(): void {
  renderAbout();
  $<HTMLDialogElement>("about").showModal();
}

async function closeTopOverlay(): Promise<boolean> {
  if (closePopover()) return true;
  const about = $<HTMLDialogElement>("about");
  if (about.open) {
    about.close();
    return true;
  }
  if (document.querySelector("dialog.palette[open], dialog.how[open]")) {
    const { closePalette } = await import("./ui/palette.ts");
    if (closePalette()) return true;
    const { closeHow } = await import("./ui/how.ts");
    if (closeHow()) return true;
  }
  if (evidence.open && evidence.mode() !== "panel") {
    evidence.hide();
    return true;
  }
  if ($("rail").classList.contains("open")) {
    closeRailDrawer();
    return true;
  }
  return false;
}

// ------------------------------------------------------------------ composer and wiring

function autosize(): void {
  const box = $<HTMLTextAreaElement>("composer-input");
  box.style.height = "auto";
  const line = Number.parseFloat(getComputedStyle(box).lineHeight) || 24;
  box.style.height = `${Math.min(box.scrollHeight, line * 8 + 16)}px`;
  const n = box.value.length;
  const max = state.manifest?.limits.max_message_chars ?? 4000;
  $("char-count").textContent = n > max * 0.85 ? `${n} / ${max}` : "";
}

function wire(): void {
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const box = $<HTMLTextAreaElement>("composer-input");
    const text = box.value;
    if (!text.trim() || state.streaming) return;
    box.value = "";
    autosize();
    void sendMessage(text);
  });
  $("composer-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $<HTMLFormElement>("composer").requestSubmit();
    }
  });
  $("composer-input").addEventListener("input", autosize);
  $("stop-btn").addEventListener("click", stopStreaming);
  $("new-chat").addEventListener("click", () => {
    newConversation();
    closeRailDrawer();
  });
  $("wordmark").addEventListener("click", (e) => {
    e.preventDefault();
    newConversation();
    closeRailDrawer();
  });
  $("collection-pill").addEventListener("click", (e) => {
    if (!closePopover()) openPopover(e.currentTarget as HTMLElement);
  });
  $("topbar-collection").addEventListener("click", (e) => {
    if (!closePopover()) openPopover(e.currentTarget as HTMLElement);
  });
  document.addEventListener("pointerdown", (e) => {
    const pop = $("collection-popover");
    if (pop.hidden) return;
    const t = e.target as Node;
    if (!pop.contains(t) && !state.popoverOpener?.contains(t)) closePopover();
  });
  for (const r of $("mode-control").querySelectorAll<HTMLInputElement>(".seg-input")) {
    r.addEventListener("change", () => {
      placeSegIndicator(true);
      const c = current();
      if (c) {
        c.groundedOnly = groundedOnly();
        saveAll();
      }
      savePrefs({ mode: groundedOnly() ? "grounded" : "chat" });
    });
  }
  $("upload-btn").addEventListener("click", () => $("file-input").click());
  $("attach-btn").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", () => {
    const input = $<HTMLInputElement>("file-input");
    const f = input.files?.[0];
    input.value = "";
    if (f) void uploadFile(f);
  });
  $("open-rail").addEventListener("click", openRailDrawer);
  $("rail-close").addEventListener("click", closeRailDrawer);
  $("rail-collapse").addEventListener("click", () => setRailCollapsed($("app").dataset.rail !== "collapsed"));
  $("scrim").addEventListener("click", () => {
    closeRailDrawer();
    if (evidence.open && evidence.mode() !== "panel") evidence.hide();
  });
  $("evidence-close").addEventListener("click", () => evidence.hide());
  $("evidence-toggle").addEventListener("click", (e) => {
    if (evidence.open) evidence.hide();
    else {
      const c = current();
      const m = [...(c?.messages ?? [])].reverse().find((x) => x.meta?.passages?.length || x.meta?.mode === "grounded_only");
      if (m) showEvidence(m, e.currentTarget as HTMLElement);
    }
  });
  $("about-btn").addEventListener("click", openAbout);
  $("about-close").addEventListener("click", () => {
    $<HTMLDialogElement>("about").close();
    $("about-btn").focus();
  });
  $("how-btn").addEventListener("click", () => void openHowItWorks());
  $("theme-toggle").addEventListener("click", toggleTheme);
  $("theme-toggle-top").addEventListener("click", toggleTheme);
  $("scroller").addEventListener("scroll", () => {
    state.autoScroll = nearBottom();
    $("jump").hidden = state.autoScroll || !current();
  });
  $("jump").addEventListener("click", () => scrollToBottom(true));
  window.addEventListener("resize", () => placeSegIndicator(false));
  document.addEventListener("keydown", (e) => {
    const mod = e.metaKey || e.ctrlKey;
    const inField = (e.target as HTMLElement | null)?.matches?.("input, textarea, [contenteditable]");
    if (mod && e.key.toLowerCase() === "k") {
      e.preventDefault();
      void openCommandPalette();
    } else if (mod && e.shiftKey && e.key.toLowerCase() === "o") {
      e.preventDefault();
      newConversation();
    } else if (e.key === "/" && !inField && !mod) {
      e.preventDefault();
      $("composer-input").focus();
    } else if (e.key === "Escape") {
      void closeTopOverlay().then((closed) => {
        if (!closed && state.streaming) stopStreaming();
      });
    }
  });
}

function trackModel(embedder: Embedder): void {
  const seal = $("hero-seal");
  const status = $("hero-status");
  const text = $("model-text");
  const dot = $("model-dot");
  dot.classList.add("busy");
  embedder.onProgress((loaded, total, cached) => {
    const f = total ? loaded / total : 0;
    setSealProgress(seal, f);
    const mb = (n: number): string => (n / 1e6).toFixed(0);
    const line = cached ? "Preparing the search index from the browser cache…" : `Preparing the search index - ${mb(loaded)}/${mb(total)} MB`;
    setText(status, line);
    setText(text, line);
  });
  embedder.ready.then(
    (ms) => {
      state.modelReady = true;
      sealReady(seal);
      status.classList.add("is-ready");
      status.textContent = `Search index ready in ${(ms / 1000).toFixed(1)} s. Retrieval runs in this browser.`;
      text.textContent = `Search index ready (${(ms / 1000).toFixed(1)} s)`;
      dot.classList.remove("busy");
      dot.classList.add("ok");
      setText($("notice"), collectionById(selectedCollection())?.notice ?? "");
      const st = indexStatus.get(selectedCollection());
      if (st) status.textContent = st.text;
    },
    (err: unknown) => {
      const line = `The search index could not be loaded: ${err instanceof Error ? err.message : String(err)}`;
      status.textContent = line;
      text.textContent = line;
      dot.classList.remove("busy");
      dot.classList.add("bad");
    },
  );
}

async function init(): Promise<void> {
  syncReducedMotion();
  const p = prefs();
  if (p.theme === "light" || p.theme === "dark") applyTheme(p.theme, false);
  syncThemeButtons();
  if (p.railCollapsed) setRailCollapsed(true, false);
  if (p.mode === "grounded") $<HTMLInputElement>("grounded-only").checked = true;
  $("hero-seal").appendChild(sealSvg({ ticks: true }));
  for (const k of document.querySelectorAll<HTMLElement>("#new-chat .kbd")) k.textContent = `${modKey} ⇧ O`;
  loadAll();
  wire();
  evidence = new Evidence(() => {
    if (!evidence.open) state.evidenceFor = null;
  });
  const share = parseShareParams(location.search);
  if (share) history.replaceState(null, "", `${location.pathname}${location.hash}`);
  document.documentElement.classList.add("ready");
  // the hero (seal, wordmark, suggested questions) paints from static data first; the tiles
  // join when the manifest arrives, and the model and index downloads start after that frame
  const wanted0 = share?.c ?? p.collection ?? "cs_ai";
  state.collection = wanted0;
  renderSuggestions(false);
  requestAnimationFrame(() => document.documentElement.classList.add("loaded"));
  try {
    const manifest = await loadManifest(BASE);
    state.manifest = manifest;
    await new Promise<void>((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
    const embedder = new Embedder(BASE, __MODEL_DTYPE__, manifest.model.query_instruction);
    trackModel(embedder);
    const store = new CollectionStore(BASE, manifest);
    const uploads = new UploadStore(BASE, manifest.uploads, manifest.gate.upload_threshold);
    state.assistant = new Assistant(manifest, store, embedder, uploads, __WORKER_URL__);
    const u = manifest.uploads;
    $("upload-privacy").textContent = `Up to ${u.max_files} PDFs, ${u.max_mb} MB each. Indexed in this browser, never uploaded.`;
    $("upload-btn").title = `Up to ${u.max_files} PDFs (${u.max_mb} MB, ${u.max_pages} pages each). Only passages that match a question are sent with it.`;
    renderTiles();
    const wanted = share?.c && collectionById(share.c) ? share.c : (p.collection ?? "cs_ai");
    chooseCollection(wanted, false);
    // hooks for the browser tests (embedding and search parity with the Python index)
    (window as unknown as Record<string, unknown>).__ragsysTest = {
      embedQueries: async (q: string[]) => (await embedder.queries(q)).map((v) => Array.from(v)),
      search: async (id: string, q: string[], k = 5) => {
        const src = await store.load(id);
        const vecs = await embedder.queries(q);
        return vecs.map((v) => src.search([v], k).map((h) => ({ id: h.chunk.chunk_id, score: h.score })));
      },
    };
    document.documentElement.dataset.ready = "true";
  } catch (err) {
    $("hero-status").textContent = `The site data could not be loaded: ${err instanceof Error ? err.message : String(err)}`;
  }
  renderConversationList();
  renderMessages();
  renderUploads();
  placeSegIndicator(false);
  updateTitle();
  if (share?.q) {
    const box = $<HTMLTextAreaElement>("composer-input");
    box.value = share.q;
  }
  autosize();
  $("composer-input").focus();
}

void init();
