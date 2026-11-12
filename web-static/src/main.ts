// The static study assistant's page (behaviour of the Python app's src/ragsys/app/static/chat.js):
// retrieval and query embedding in this browser, replies streamed from the LLM proxy. Model text
// is rendered by markdown.ts (text nodes only); everything else is built with createElement.

import "../../src/ragsys/app/static/chat.css";
import "./static.css";
import { ProxyError } from "./api.ts";
import { CollectionStore, loadManifest } from "./data.ts";
import { Embedder } from "./embedder.ts";
import { Assistant, type ChatOutcome } from "./engine.ts";
import type { Summary, Turn } from "./history.ts";
import { copyText, el, render, SAFE_URL } from "./markdown.ts";
import { storage } from "./storage.ts";
import type { CollectionMeta, Manifest, Passage } from "./types.ts";
import { UploadError, UploadStore } from "./uploads.ts";
import type { VerifyResult } from "./verify.ts";

declare const __WORKER_URL__: string;
declare const __MODEL_DTYPE__: string;

const BASE = import.meta.env.BASE_URL;
const STORE_KEY = "ragsys.static.v1";
const PREF_KEY = "ragsys.static.prefs.v1";
const MAX_CONVERSATIONS = 50;

type Status = "streaming" | "done" | "error" | "stopped";

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
  error?: string;
  verify?: VerifyResult | { error: string };
  statusText?: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  meta?: Meta;
  node?: HTMLElement;
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

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} is missing`);
  return node as T;
};

const state = {
  conversations: [] as Conversation[],
  currentId: null as string | null,
  manifest: null as Manifest | null,
  assistant: null as Assistant | null,
  streaming: null as { controller: AbortController; conv: Conversation; msg: Message } | null,
  autoScroll: true,
  uploading: false,
};

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

function prefs(): { collection?: string } {
  return storage.get<{ collection?: string }>(PREF_KEY) ?? {};
}

function savePrefs(update: { collection?: string }): void {
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
  return $<HTMLSelectElement>("collection").value;
}

function fillCollections(): void {
  const sel = $<HTMLSelectElement>("collection");
  for (const c of state.manifest?.collections ?? []) {
    const opt = el("option", null, c.label);
    opt.value = c.id;
    sel.appendChild(opt);
  }
}

function applyCollection(id: string): void {
  const sel = $<HTMLSelectElement>("collection");
  sel.value = collectionById(id) ? id : "";
  const chosen = collectionById(sel.value);
  $("collection-desc").textContent = chosen
    ? `${chosen.evaluated ? "Evaluated collection. " : "Not evaluated. "}${chosen.description}`
    : "General chat. Uploaded PDFs are still searched.";
  $("notice").textContent = chosen?.notice ?? "";
  renderExamples();
  updateModeBadge();
  if (chosen) void preloadCollection(chosen);
  else setIndexStatus(null);
}

function setIndexStatus(text: string | null, pct?: number): void {
  const bar = $<HTMLProgressElement>("index-bar");
  bar.value = pct ?? (text ? 0 : 100);
  $("index-text").textContent = text ?? "No collection selected: general chat.";
}

async function preloadCollection(meta: CollectionMeta): Promise<void> {
  const a = state.assistant;
  if (!a) return;
  const mb = (meta.files.passages.gzip_bytes + meta.files.vectors.gzip_bytes) / 1e6;
  if (!a.store.isLoaded(meta.id)) setIndexStatus(`Loading ${meta.label} (about ${mb.toFixed(1)} MB)…`, 0);
  try {
    await a.store.load(meta.id, (loaded, total) => {
      if (selectedCollection() === meta.id) setIndexStatus(`Loading ${meta.label}…`, total ? (100 * loaded) / total : 0);
    });
    if (selectedCollection() === meta.id) setIndexStatus(`${meta.label}: ${meta.count} passages ready.`, 100);
  } catch (err) {
    if (selectedCollection() === meta.id) setIndexStatus(`${meta.label} could not be loaded: ${err instanceof Error ? err.message : String(err)}`, 0);
  }
}

function renderExamples(): void {
  const list = $("examples");
  list.textContent = "";
  const general = state.manifest?.general_examples ?? [];
  const coll = collectionById(selectedCollection());
  const seen = new Set<string>();
  for (const text of [...(coll?.examples.slice(0, 4) ?? []), ...general]) {
    if (seen.has(text)) continue;
    seen.add(text);
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
}

function updateModeBadge(): void {
  const badge = $("mode-badge");
  const c = collectionById(selectedCollection());
  if ($<HTMLInputElement>("grounded-only").checked) {
    badge.textContent = "Grounded only";
    badge.hidden = false;
  } else if (c) {
    badge.textContent = c.label;
    badge.hidden = false;
  } else badge.hidden = true;
}

// ------------------------------------------------------------------ sidebar

function svgIcon(path: string, size = 16): SVGSVGElement {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(ns, "path");
  p.setAttribute("d", path);
  p.setAttribute("fill", "currentColor");
  svg.appendChild(p);
  return svg;
}

function renderConversationList(): void {
  const list = $("conv-list");
  list.textContent = "";
  const convs = [...state.conversations].sort((a, b) => (b.updated || 0) - (a.updated || 0));
  $("conv-empty").hidden = convs.length > 0;
  for (const c of convs) {
    const li = el("li", "conv-item");
    if (c.id === state.currentId) li.setAttribute("aria-current", "true");
    const open = el("button", "conv-open", c.title || "New chat");
    open.type = "button";
    open.title = c.title || "New chat";
    open.addEventListener("click", () => {
      openConversation(c.id);
      closeSidebar();
    });
    const del = el("button", "conv-del");
    del.type = "button";
    del.setAttribute("aria-label", `Delete conversation: ${c.title || "New chat"}`);
    del.appendChild(svgIcon("M7 21q-.8 0-1.4-.6T5 19V6H4V4h5V3h6v1h5v2h-1v13q0 .8-.6 1.4T17 21zM17 6H7v13h10zM9 17h2V8H9zm4 0h2V8h-2z"));
    del.addEventListener("click", () => deleteConversation(c.id));
    li.append(open, del);
    list.appendChild(li);
  }
}

function newConversation(): void {
  if (state.streaming) stopStreaming();
  state.currentId = null;
  $<HTMLInputElement>("grounded-only").checked = false;
  updateModeBadge();
  renderMessages();
  renderConversationList();
  $("chat-title").textContent = "Study Assistant";
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
      groundedOnly: $<HTMLInputElement>("grounded-only").checked,
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
  const c = current();
  if (c) {
    applyCollection(c.collection || "");
    $<HTMLInputElement>("grounded-only").checked = Boolean(c.groundedOnly);
    updateModeBadge();
    $("chat-title").textContent = c.title || "Study Assistant";
  }
  renderConversationList();
  renderMessages();
  scrollToBottom(true);
}

function deleteConversation(id: string): void {
  if (state.streaming?.conv.id === id) stopStreaming();
  state.conversations = state.conversations.filter((c) => c.id !== id);
  if (state.currentId === id) state.currentId = null;
  saveAll();
  renderConversationList();
  renderMessages();
  announce("Conversation deleted.");
}

function openSidebar(): void {
  $("sidebar").classList.add("open");
  $("scrim").hidden = false;
  $("open-sidebar").setAttribute("aria-expanded", "true");
  $("new-chat").focus();
}

function closeSidebar(): void {
  if (!$("sidebar").classList.contains("open")) return;
  $("sidebar").classList.remove("open");
  $("scrim").hidden = true;
  $("open-sidebar").setAttribute("aria-expanded", "false");
  $("open-sidebar").focus();
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
  $("welcome").hidden = msgs.length > 0;
  if (c) msgs.forEach((m, i) => list.appendChild(messageNode(c, m, i)));
}

function lastIndex(c: Conversation, role: Message["role"]): number {
  for (let i = c.messages.length - 1; i >= 0; i--) if (c.messages[i]?.role === role) return i;
  return -1;
}

function actionButton(label: string, aria: string, onClick: (b: HTMLButtonElement) => void): HTMLButtonElement {
  const b = el("button", "act", label);
  b.type = "button";
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
      actions.appendChild(actionButton("Edit", "Edit and resend your last message", () => startEdit(c, m, li)));
      actions.appendChild(actionButton("Copy", "Copy your message", (b) => copyText(m.content, b)));
      li.appendChild(actions);
    }
    return li;
  }
  li.appendChild(el("span", "who", "Assistant"));
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
    b.setAttribute("aria-label", `Source ${label}: ${p.title || p.heading_path}`);
    b.addEventListener("click", () => openPassage(p, b));
    return b;
  };
}

function renderAnswer(c: Conversation, m: Message, li: HTMLElement, final: boolean): void {
  const meta = m.meta ?? { status: "done", collection: "", notice: "" };
  const body = li.querySelector<HTMLElement>(".answer");
  if (!body) return;
  body.textContent = "";
  if (meta.status === "streaming" && !m.content) {
    const t = el("div", "typing");
    t.setAttribute("aria-label", "The assistant is writing");
    t.append(el("span"), el("span"), el("span"));
    body.appendChild(t);
    if (meta.statusText) body.appendChild(el("div", "status-line", meta.statusText));
  } else if (m.content) {
    body.appendChild(render(m.content, { cite: citeFactory(m), highlight: final }));
  }
  for (const n of li.querySelectorAll(".source-line, .notice-line, .error-box, .closest, .msg-actions, .verify, .status-line.after")) n.remove();
  if (meta.status === "streaming") {
    if (meta.context?.length && m.content) {
      li.appendChild(el("div", "status-line after", `Using ${meta.context.length} passage${meta.context.length === 1 ? "" : "s"} from the documents…`));
    }
    return;
  }
  if (meta.mode === "grounded_only" && meta.outcome === "abstained") {
    body.appendChild(el("p", null, meta.abstentionMessage ?? "Not found in these documents"));
    if (meta.closest?.length) {
      li.appendChild(el("p", "status-line", "Closest sections:"));
      const ul = el("ul", "closest");
      for (const s of meta.closest) ul.appendChild(el("li", null, `${s.heading_path}${s.pages ? ` (p. ${s.pages})` : ""}`));
      li.appendChild(ul);
    }
  }
  if (meta.sourceLine) li.appendChild(sourceLineNode(m));
  if (meta.notice && meta.collection) li.appendChild(el("div", "notice-line", meta.notice));
  if (meta.error) {
    const box = el("div", "error-box", meta.error);
    box.setAttribute("role", "alert");
    li.appendChild(box);
  }
  if (meta.verify) li.appendChild(verifyNode(meta.verify));
  li.appendChild(answerActions(c, m, li));
}

function shortHeading(path: string): string {
  const parts = path.split(" > ");
  return parts.length > 2 ? `${parts[0]} › ${parts[parts.length - 1]}` : parts.join(" › ");
}

function sourceLineNode(m: Message): HTMLElement {
  const meta = m.meta as Meta;
  const kind = meta.mode === "grounded_only" && meta.outcome === "abstained" ? "abstained" : meta.grounded ? "grounded" : "general";
  const line = el("div", `source-line ${kind}`);
  line.appendChild(el("span", "label-text", meta.sourceLine ?? ""));
  for (const p of meta.passages ?? []) {
    const b = el("button", "src-chip");
    b.type = "button";
    b.appendChild(el("b", null, p.label));
    b.appendChild(document.createTextNode(p.title ? `${p.title} · ${shortHeading(p.heading_path)}` : shortHeading(p.heading_path)));
    b.title = p.heading_path;
    b.setAttribute("aria-label", `Open source ${p.label}: ${p.title ? `${p.title}, ` : ""}${p.heading_path}`);
    b.addEventListener("click", () => openPassage(p, b));
    line.appendChild(b);
  }
  return line;
}

function answerActions(c: Conversation, m: Message, li: HTMLElement): HTMLElement {
  const actions = el("div", "msg-actions");
  const meta = m.meta;
  actions.appendChild(actionButton("Copy", "Copy the answer", (b) => copyText(m.content, b)));
  if (c.messages[c.messages.length - 1] === m && !state.streaming) {
    actions.appendChild(actionButton("Regenerate", "Regenerate the answer", () => regenerate(c)));
  }
  if (meta?.mode !== "grounded_only" && meta?.grounded && meta.passages?.length && !meta.verify) {
    actions.appendChild(actionButton("Verify", "Verify this answer against its sources", (b) => void verify(c, m, li, b)));
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

// ------------------------------------------------------------------ passage panel

let panelOpener: HTMLElement | null = null;

function openPassage(p: Passage, opener: HTMLElement): void {
  panelOpener = opener;
  const coll = collectionById(p.source);
  $("panel-label").textContent = p.source === "upload" ? `Your upload · ${p.label}` : `${coll?.label ?? "Source"} · ${p.label}`;
  $("panel-title").textContent = p.heading_path || p.title || "Passage";
  $("panel-meta").textContent = `${p.title ? `${p.title} · ` : ""}${p.pages ? `Pages ${p.pages}` : ""}`;
  $("panel-text").textContent = p.text || "(text not available)";
  $("passage-panel").hidden = false;
  $("panel-title").focus();
}

function closePassage(): void {
  if ($("passage-panel").hidden) return;
  $("passage-panel").hidden = true;
  if (panelOpener && document.body.contains(panelOpener)) panelOpener.focus();
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
  $<HTMLButtonElement>("stop-btn").disabled = false;
  $<HTMLButtonElement>("send-btn").disabled = busy;
  $<HTMLSelectElement>("collection").disabled = busy;
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
  c.groundedOnly = $<HTMLInputElement>("grounded-only").checked;
  if (!c.title) {
    c.title = value.replace(/\s+/g, " ").slice(0, 60);
    $("chat-title").textContent = c.title;
  }
  c.messages.push({ role: "user", content: value });
  c.updated = Date.now();
  await runAssistant(c);
}

async function runAssistant(c: Conversation): Promise<void> {
  const a = state.assistant;
  const coll = collectionById(c.collection);
  const m: Message = {
    role: "assistant",
    content: "",
    meta: {
      status: "streaming",
      collection: c.collection,
      notice: coll?.notice ?? "",
      statusText: c.groundedOnly ? "Searching and checking the documents…" : "",
    },
  };
  c.messages.push(m);
  saveAll();
  renderConversationList();
  renderMessages();
  scrollToBottom(true);
  const controller = new AbortController();
  state.streaming = { controller, conv: c, msg: m };
  setBusy(true);
  announce("The assistant is replying.");
  const meta = m.meta as Meta;
  try {
    if (!a) throw new ProxyError("loading", "The page is still loading. Please wait a moment.", 0);
    if (!a.embedder) throw new ProxyError("loading", "The search model is not ready.", 0);
    const turns = historyFor(c);
    const out: ChatOutcome = await a.chat(
      { turns, collection: c.collection || null, groundedOnly: c.groundedOnly, summary: c.summary },
      {
        onContext: (grounded, passages) => {
          meta.context = passages;
          if (grounded) meta.statusText = `Found ${passages.length} matching passage${passages.length === 1 ? "" : "s"}…`;
          scheduleRender(c, m);
        },
        onToken: (t) => {
          m.content += t;
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
    });
    if (out.summary) c.summary = out.summary;
    meta.status = out.stopped ? "stopped" : "done";
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") meta.status = "stopped";
    else {
      meta.status = "error";
      meta.error = err instanceof ProxyError || err instanceof Error ? err.message : "Something went wrong.";
    }
  }
  finishMessage(c, m);
}

function finishMessage(c: Conversation, m: Message): void {
  const meta = m.meta as Meta;
  delete meta.statusText;
  state.streaming = null;
  setBusy(false);
  c.updated = Date.now();
  saveAll();
  renderMessages();
  renderConversationList();
  scrollToBottom(false);
  announce(meta.error ? `Error: ${meta.error}` : `Reply finished. ${meta.sourceLine ?? ""}`);
}

function stopStreaming(): void {
  state.streaming?.controller.abort();
}

function regenerate(c: Conversation): void {
  if (state.streaming) return;
  if (c.messages[c.messages.length - 1]?.role === "assistant") c.messages.pop();
  if (c.messages[c.messages.length - 1]?.role !== "user") return;
  c.collection = selectedCollection();
  c.groundedOnly = $<HTMLInputElement>("grounded-only").checked;
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
    c.groundedOnly = $<HTMLInputElement>("grounded-only").checked;
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

async function verify(c: Conversation, m: Message, li: HTMLElement, button: HTMLButtonElement): Promise<void> {
  const a = state.assistant;
  const meta = m.meta;
  if (!a || !meta) return;
  button.disabled = true;
  button.textContent = "Verifying…";
  const idx = c.messages.indexOf(m);
  let question = "";
  for (let i = idx - 1; i >= 0; i--) {
    if (c.messages[i]?.role === "user") {
      question = c.messages[i]?.content ?? "";
      break;
    }
  }
  try {
    meta.verify = await a.verify(question || "(question)", m.content, meta.passages ?? [], meta.collection || null);
  } catch (err) {
    meta.verify = { error: err instanceof Error ? err.message : "The verification failed. Please try again." };
  }
  saveAll();
  renderAnswer(c, m, li, true);
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
    del.appendChild(svgIcon("M6.4 5 5 6.4 10.6 12 5 17.6 6.4 19l5.6-5.6 5.6 5.6 1.4-1.4-5.6-5.6L19 6.4 17.6 5 12 10.6z"));
    del.addEventListener("click", () => {
      store?.remove(d.docId);
      uploadStatus(`${d.name} removed.`, null, false);
      renderUploads();
    });
    li.appendChild(del);
    list.appendChild(li);
  }
  const max = state.manifest?.uploads.max_files ?? 3;
  $<HTMLButtonElement>("upload-btn").disabled = state.uploading || !store || store.docs.length >= max;
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
    const limits = a.uploads;
    void limits;
    const bytes = new Uint8Array(await file.arrayBuffer());
    await a.embedder.ready;
    const doc = await a.uploads.add({ name: file.name, bytes }, a.embedder, (stage, done, total) => {
      const label = stage === "reading" ? "Reading pages" : "Indexing passages";
      uploadStatus(`${label} ${done}/${total}…`, total ? (100 * done) / total : null, false);
    });
    uploadStatus(`${doc.name} is ready (${doc.source.chunks.length} passages). Ask about it.`, null, false);
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
    if (c.notice) sec.appendChild(el("p", "hint", c.notice));
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

// ------------------------------------------------------------------ composer and wiring

function autosize(): void {
  const box = $<HTMLTextAreaElement>("composer-input");
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, window.innerHeight * 0.4)}px`;
  const n = box.value.length;
  const max = state.manifest?.limits.max_message_chars ?? 4000;
  $("char-count").textContent = n > max * 0.85 ? `${n} / ${max}` : "";
}

function loadVendorScripts(): void {
  const css = document.createElement("link");
  css.rel = "stylesheet";
  css.href = `${BASE}vendor/katex/katex.min.css`;
  document.head.appendChild(css);
  for (const src of ["vendor/katex/katex.min.js", "vendor/highlight/highlight.min.js"]) {
    const s = document.createElement("script");
    s.src = `${BASE}${src}`;
    s.defer = true;
    document.head.appendChild(s);
  }
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
    closeSidebar();
  });
  $("collection").addEventListener("change", () => {
    applyCollection(selectedCollection());
    const c = current();
    if (c) {
      c.collection = selectedCollection();
      saveAll();
    }
    savePrefs({ collection: selectedCollection() });
  });
  $("grounded-only").addEventListener("change", () => {
    const c = current();
    if (c) {
      c.groundedOnly = $<HTMLInputElement>("grounded-only").checked;
      saveAll();
    }
    updateModeBadge();
  });
  $("upload-btn").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", () => {
    const input = $<HTMLInputElement>("file-input");
    const f = input.files?.[0];
    input.value = "";
    if (f) void uploadFile(f);
  });
  $("open-sidebar").addEventListener("click", openSidebar);
  $("close-sidebar").addEventListener("click", closeSidebar);
  $("scrim").addEventListener("click", closeSidebar);
  $("panel-close").addEventListener("click", closePassage);
  $("about-btn").addEventListener("click", () => {
    renderAbout();
    $<HTMLDialogElement>("about").showModal();
  });
  $("about-close").addEventListener("click", () => {
    $<HTMLDialogElement>("about").close();
    $("about-btn").focus();
  });
  $("scroller").addEventListener("scroll", () => {
    state.autoScroll = nearBottom();
    $("jump").hidden = state.autoScroll || !current();
  });
  $("jump").addEventListener("click", () => scrollToBottom(true));
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!$("passage-panel").hidden) closePassage();
    else if ($("sidebar").classList.contains("open")) closeSidebar();
    else if (state.streaming) stopStreaming();
  });
}

function trackModel(embedder: Embedder): void {
  const bar = $<HTMLProgressElement>("model-bar");
  const text = $("model-text");
  embedder.onProgress((loaded, total, cached) => {
    bar.value = total ? (100 * loaded) / total : 0;
    text.textContent = cached ? "Loading the search model from this browser's cache…" : `Downloading the search model (once): ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`;
  });
  embedder.ready.then(
    (ms) => {
      bar.value = 100;
      text.textContent = `Search model ready (${(ms / 1000).toFixed(1)} s).`;
      document.querySelector(".load-panel")?.classList.add("done");
    },
    (err: unknown) => {
      text.textContent = `The search model could not be loaded: ${err instanceof Error ? err.message : String(err)}`;
    },
  );
}

async function init(): Promise<void> {
  loadVendorScripts();
  loadAll();
  wire();
  try {
    const manifest = await loadManifest(BASE);
    state.manifest = manifest;
    const embedder = new Embedder(BASE, __MODEL_DTYPE__, manifest.model.query_instruction);
    trackModel(embedder);
    const store = new CollectionStore(BASE, manifest);
    const uploads = new UploadStore(BASE, manifest.uploads, manifest.gate.upload_threshold);
    state.assistant = new Assistant(manifest, store, embedder, uploads, __WORKER_URL__);
    const u = manifest.uploads;
    $("upload-privacy").textContent = `Up to ${u.max_files} PDFs (${u.max_mb} MB, ${u.max_pages} pages each). They are read and indexed in this browser and never uploaded; only passages that match a question are sent with it.`;
    fillCollections();
    const p = prefs();
    applyCollection(p.collection ?? "cs_ai");
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
    $("collection-desc").textContent = `The site data could not be loaded: ${err instanceof Error ? err.message : String(err)}`;
  }
  renderConversationList();
  renderMessages();
  renderUploads();
  autosize();
  $("composer-input").focus();
}

void init();
