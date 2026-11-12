// Chat assistant page. Model text is rendered by ChatMarkdown (text nodes only); this file
// builds the rest of the page with createElement / textContent. Conversations live in this
// browser's localStorage (every access guarded; the page works without storage).
"use strict";

(function () {
  const MD = window.ChatMarkdown;
  const $ = (id) => document.getElementById(id);
  const el = MD.el;
  const STORE_KEY = "ragsys.assistant.v1";
  const PREF_KEY = "ragsys.assistant.prefs.v1";
  const MAX_CONVERSATIONS = 50;
  const MAX_SENT_MESSAGES = 100;

  // ------------------------------------------------------------------ storage (guarded)

  const storage = {
    ok: true,
    get(key) {
      try {
        const raw = window.localStorage.getItem(key);
        return raw ? JSON.parse(raw) : null;
      } catch (err) {
        this.ok = false;
        return null;
      }
    },
    set(key, value) {
      try {
        window.localStorage.setItem(key, JSON.stringify(value));
        return true;
      } catch (err) {
        return false;
      }
    },
    probe() {
      try {
        const k = "ragsys.probe";
        window.localStorage.setItem(k, "1");
        window.localStorage.removeItem(k);
        return true;
      } catch (err) {
        return false;
      }
    },
  };

  // ------------------------------------------------------------------ state

  const state = {
    conversations: [],
    currentId: null,
    collections: [],
    health: null,
    backend: null,
    sessionId: null,
    uploads: [],
    streaming: null, // {controller, streamId, conv, msg}
    autoScroll: true,
    uploading: false,
  };

  function uid() {
    const a = new Uint8Array(8);
    if (window.crypto && window.crypto.getRandomValues) window.crypto.getRandomValues(a);
    else for (let i = 0; i < a.length; i++) a[i] = Math.floor(Math.random() * 256);
    return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
  }

  function current() {
    return state.conversations.find((c) => c.id === state.currentId) || null;
  }

  function saveAll() {
    if (!storage.ok) return;
    const data = state.conversations.slice(0, MAX_CONVERSATIONS).map((c) => ({
      ...c,
      messages: c.messages.map((m) => ({ role: m.role, content: m.content, meta: stripMeta(m.meta) })),
    }));
    while (!storage.set(STORE_KEY, { v: 1, conversations: data })) {
      // storage full: drop the oldest conversation other than the current one
      const idx = data.map((c) => c.id !== state.currentId).lastIndexOf(true);
      if (idx < 0) {
        storage.ok = false;
        showStorageNote();
        return;
      }
      data.splice(idx, 1);
    }
  }

  function stripMeta(meta) {
    if (!meta) return undefined;
    const { context, ...rest } = meta; // passages offered during streaming are not kept
    if (rest.status === "streaming") rest.status = "stopped";
    return rest;
  }

  function loadAll() {
    if (!storage.probe()) {
      storage.ok = false;
      showStorageNote();
      return;
    }
    const data = storage.get(STORE_KEY);
    if (data && Array.isArray(data.conversations)) {
      state.conversations = data.conversations.filter((c) => c && c.id && Array.isArray(c.messages));
    }
  }

  function prefs() {
    return storage.get(PREF_KEY) || {};
  }

  function savePrefs(update) {
    if (storage.ok) storage.set(PREF_KEY, { ...prefs(), ...update });
  }

  function showStorageNote() {
    $("storage-note").hidden = false;
  }

  // ------------------------------------------------------------------ API helpers

  async function getJSON(url) {
    const r = await fetch(url, { headers: { Accept: "application/json" } });
    if (!r.ok) throw new Error((await errorText(r)) || "HTTP " + r.status);
    return r.json();
  }

  async function errorText(r) {
    try {
      const j = await r.json();
      if (typeof j.detail === "string") return j.detail;
      if (Array.isArray(j.detail) && j.detail[0] && j.detail[0].msg) return "Request rejected: " + j.detail[0].msg;
    } catch (err) {
      /* not JSON */
    }
    return r.status === 429 ? "Too many requests. Please wait a minute." : "The server answered with an error (HTTP " + r.status + ").";
  }

  function parseEvent(block) {
    let name = "message";
    const data = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) name = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (!data.length) return null;
    try {
      return { event: name, data: JSON.parse(data.join("\n")) };
    } catch (err) {
      return null;
    }
  }

  // reads Server-Sent Events from a fetch response
  async function readEvents(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const ev = parseEvent(buf.slice(0, i));
        buf = buf.slice(i + 2);
        if (ev) onEvent(ev);
      }
    }
  }

  // ------------------------------------------------------------------ collections, backends

  function collectionById(id) {
    return state.collections.find((c) => c.id === id) || null;
  }

  function selectedCollection() {
    return $("collection").value || "";
  }

  function fillCollections() {
    const sel = $("collection");
    for (const c of state.collections) {
      const opt = el("option", null, c.label + (c.available ? "" : " (not available)"));
      opt.value = c.id;
      opt.disabled = !c.available;
      sel.appendChild(opt);
    }
  }

  function applyCollection(id) {
    const sel = $("collection");
    const c = collectionById(id);
    sel.value = c && c.available ? id : "";
    const chosen = collectionById(sel.value);
    $("collection-desc").textContent = chosen
      ? (chosen.evaluated ? "Evaluated collection. " : "Not evaluated. ") + (chosen.description || "")
      : "General chat. Uploaded PDFs are still searched.";
    $("notice").textContent = chosen && chosen.notice ? chosen.notice : "";
    renderExamples();
    updateModeBadge();
  }

  function renderExamples() {
    const list = $("examples");
    list.textContent = "";
    const general = (state.health && state.health.general_examples) || [];
    const c = collectionById(selectedCollection());
    const fromColl = c ? (c.examples || []).slice(0, 4) : [];
    const seen = new Set();
    for (const text of [...fromColl, ...general]) {
      if (seen.has(text)) continue;
      seen.add(text);
      const li = el("li");
      const b = el("button", "chip", text);
      b.type = "button";
      b.addEventListener("click", () => {
        const box = $("composer-input");
        box.value = text;
        autosize();
        box.focus();
        box.setSelectionRange(text.length, text.length);
      });
      li.appendChild(b);
      list.appendChild(li);
    }
  }

  function fillBackends() {
    const h = state.health;
    if (!h) return;
    const list = h.backends || [];
    const pref = prefs().backend;
    state.backend = list.some((b) => b.name === pref) ? pref : h.default_backend;
    const sel = $("backend");
    sel.textContent = "";
    if (list.length > 1) {
      for (const b of list) {
        const opt = el("option", null, b.label + (b.available ? "" : " (unavailable)"));
        opt.value = b.name;
        sel.appendChild(opt);
      }
      sel.value = state.backend;
      sel.hidden = false;
      $("backend-name").hidden = true;
    } else {
      sel.hidden = true;
      $("backend-name").hidden = false;
    }
    updateBackend();
  }

  function updateBackend() {
    const h = state.health;
    const b = h && (h.backends || []).find((x) => x.name === state.backend);
    $("backend-name").textContent = b ? b.label : "";
    const dot = $("backend-dot");
    dot.className = "dot " + (b ? (b.available ? "ok" : "bad") : "");
    $("backend-desc").textContent = b ? (b.available ? b.privacy || "" : b.reason || "Unavailable") : "";
    $("about-model").textContent = b ? "Model: " + b.label + ". " + (b.privacy || "") : "";
  }

  function updateModeBadge() {
    const badge = $("mode-badge");
    const grounded = $("grounded-only").checked;
    const c = collectionById(selectedCollection());
    if (grounded) {
      badge.textContent = "Grounded only";
      badge.hidden = false;
    } else if (c) {
      badge.textContent = c.label;
      badge.hidden = false;
    } else {
      badge.hidden = true;
    }
  }

  // ------------------------------------------------------------------ sidebar

  function renderConversationList() {
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
      del.setAttribute("aria-label", "Delete conversation: " + (c.title || "New chat"));
      del.appendChild(icon("M7 21q-.8 0-1.4-.6T5 19V6H4V4h5V3h6v1h5v2h-1v13q0 .8-.6 1.4T17 21zM17 6H7v13h10zM9 17h2V8H9zm4 0h2V8h-2z", 16));
      del.addEventListener("click", () => deleteConversation(c.id));
      li.appendChild(open);
      li.appendChild(del);
      list.appendChild(li);
    }
  }

  function icon(path, size) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("width", String(size || 16));
    svg.setAttribute("height", String(size || 16));
    svg.setAttribute("aria-hidden", "true");
    const p = document.createElementNS(ns, "path");
    p.setAttribute("d", path);
    p.setAttribute("fill", "currentColor");
    svg.appendChild(p);
    return svg;
  }

  function newConversation() {
    if (state.streaming) stopStreaming();
    state.currentId = null;
    $("grounded-only").checked = false;
    renderMessages();
    renderConversationList();
    $("chat-title").textContent = "Study Assistant";
    $("composer-input").focus();
  }

  function ensureConversation() {
    let c = current();
    if (!c) {
      c = {
        id: uid(),
        title: "",
        created: Date.now(),
        updated: Date.now(),
        collection: selectedCollection(),
        groundedOnly: $("grounded-only").checked,
        summary: null,
        messages: [],
      };
      state.conversations.unshift(c);
      state.currentId = c.id;
      if (state.conversations.length > MAX_CONVERSATIONS) state.conversations.length = MAX_CONVERSATIONS;
    }
    return c;
  }

  function openConversation(id) {
    if (state.streaming) stopStreaming();
    state.currentId = id;
    const c = current();
    if (c) {
      applyCollection(c.collection || "");
      $("grounded-only").checked = !!c.groundedOnly;
      updateModeBadge();
      $("chat-title").textContent = c.title || "Study Assistant";
    }
    renderConversationList();
    renderMessages();
    scrollToBottom(true);
  }

  function deleteConversation(id) {
    if (state.streaming && state.streaming.conv.id === id) stopStreaming();
    state.conversations = state.conversations.filter((c) => c.id !== id);
    if (state.currentId === id) state.currentId = null;
    saveAll();
    renderConversationList();
    renderMessages();
    announce("Conversation deleted.");
  }

  function openSidebar() {
    $("sidebar").classList.add("open");
    $("scrim").hidden = false;
    $("open-sidebar").setAttribute("aria-expanded", "true");
    $("new-chat").focus();
  }

  function closeSidebar() {
    if (!$("sidebar").classList.contains("open")) return;
    $("sidebar").classList.remove("open");
    $("scrim").hidden = true;
    $("open-sidebar").setAttribute("aria-expanded", "false");
    $("open-sidebar").focus();
  }

  // ------------------------------------------------------------------ messages

  function announce(text) {
    const live = $("live");
    live.textContent = "";
    setTimeout(() => {
      live.textContent = text;
    }, 30);
  }

  function renderMessages() {
    const list = $("messages");
    list.textContent = "";
    const c = current();
    const msgs = c ? c.messages : [];
    $("welcome").hidden = msgs.length > 0;
    msgs.forEach((m, i) => list.appendChild(messageNode(c, m, i)));
  }

  function lastIndex(c, role) {
    for (let i = c.messages.length - 1; i >= 0; i--) if (c.messages[i].role === role) return i;
    return -1;
  }

  function messageNode(c, m, index) {
    const li = el("li", "msg " + m.role);
    m._node = li;
    if (m.role === "user") {
      li.appendChild(el("span", "visually-hidden", "You said:"));
      li.appendChild(el("div", "bubble-user", m.content));
      const isLast = index === lastIndex(c, "user");
      if (isLast && !state.streaming) {
        const actions = el("div", "msg-actions");
        actions.appendChild(actionButton("Edit", "Edit and resend your last message", () => startEdit(c, m, li)));
        actions.appendChild(actionButton("Copy", "Copy your message", (b) => MD.copyText(m.content, b, "Copy")));
        li.appendChild(actions);
      }
      return li;
    }
    li.appendChild(el("span", "who", "Assistant"));
    const body = el("div", "answer");
    li.appendChild(body);
    renderAnswer(c, m, li, true);
    return li;
  }

  function actionButton(label, aria, onClick) {
    const b = el("button", "act", label);
    b.type = "button";
    b.setAttribute("aria-label", aria);
    b.addEventListener("click", () => onClick(b));
    return b;
  }

  function passageMap(m) {
    const map = {};
    const meta = m.meta || {};
    for (const p of meta.context || []) map[p.label] = p;
    for (const p of meta.passages || []) map[p.label] = p;
    return map;
  }

  function citeFactory(m) {
    const map = passageMap(m);
    return (label) => {
      const p = map[label];
      if (!p) return null;
      const b = el("button", "cite-inline", label);
      b.type = "button";
      b.setAttribute("aria-label", "Source " + label + ": " + (p.title || p.heading_path || ""));
      b.addEventListener("click", () => openPassage(p, b));
      return b;
    };
  }

  // (re)builds an assistant message: answer text, source line, notices and actions
  function renderAnswer(c, m, li, final) {
    const meta = m.meta || {};
    const body = li.querySelector(".answer");
    body.textContent = "";
    if (meta.status === "streaming" && !m.content) {
      const t = el("div", "typing");
      t.setAttribute("aria-label", "The assistant is writing");
      t.append(el("span"), el("span"), el("span"));
      body.appendChild(t);
      if (meta.statusText) body.appendChild(el("div", "status-line", meta.statusText));
    } else if (m.content) {
      body.appendChild(MD.render(m.content, { cite: citeFactory(m), highlight: final }));
    }
    for (const n of li.querySelectorAll(".source-line, .notice-line, .error-box, .closest, .msg-actions, .verify, .status-line.after")) n.remove();
    if (meta.status === "streaming") {
      if (meta.context && meta.context.length && m.content) {
        li.appendChild(el("div", "status-line after", "Using " + meta.context.length + " passage" + (meta.context.length === 1 ? "" : "s") + " from the documents…"));
      }
      return;
    }
    if (meta.mode === "grounded_only" && meta.outcome === "abstained") {
      body.appendChild(el("p", null, meta.abstentionMessage || "Not found in these documents"));
      if (meta.closest && meta.closest.length) {
        li.appendChild(el("p", "status-line", "Closest sections:"));
        const ul = el("ul", "closest");
        for (const s of meta.closest) ul.appendChild(el("li", null, s.heading_path + (s.pages ? " (p. " + s.pages + ")" : "")));
        li.appendChild(ul);
      }
    }
    if (meta.sourceLine) li.appendChild(sourceLine(m));
    if (meta.notice && (meta.collection || "") !== "") li.appendChild(el("div", "notice-line", meta.notice));
    if (meta.error) {
      const box = el("div", "error-box", meta.error);
      box.setAttribute("role", "alert");
      li.appendChild(box);
    }
    if (meta.verify) li.appendChild(verifyNode(meta.verify));
    li.appendChild(answerActions(c, m, li));
  }

  function sourceLine(m) {
    const meta = m.meta || {};
    const kind = meta.mode === "grounded_only" && meta.outcome === "abstained" ? "abstained" : meta.grounded ? "grounded" : "general";
    const line = el("div", "source-line " + kind);
    line.appendChild(el("span", "label-text", meta.sourceLine));
    for (const p of meta.passages || []) {
      const b = el("button", "src-chip");
      b.type = "button";
      b.appendChild(el("b", null, p.label));
      b.appendChild(document.createTextNode(p.title ? p.title + " · " + shortHeading(p.heading_path) : shortHeading(p.heading_path)));
      b.title = p.heading_path;
      b.setAttribute("aria-label", "Open source " + p.label + ": " + (p.title ? p.title + ", " : "") + p.heading_path);
      b.addEventListener("click", () => openPassage(p, b));
      line.appendChild(b);
    }
    return line;
  }

  function shortHeading(path) {
    const parts = String(path || "").split(" > ");
    return parts.length > 2 ? parts[0] + " › " + parts[parts.length - 1] : parts.join(" › ");
  }

  function answerActions(c, m, li) {
    const actions = el("div", "msg-actions");
    const meta = m.meta || {};
    actions.appendChild(actionButton("Copy", "Copy the answer", (b) => MD.copyText(m.content || "", b, "Copy")));
    const isLast = c.messages[c.messages.length - 1] === m;
    if (isLast && !state.streaming) actions.appendChild(actionButton("Regenerate", "Regenerate the answer", () => regenerate(c)));
    if (meta.mode !== "grounded_only" && meta.grounded && (meta.passages || []).length && !meta.verify) {
      actions.appendChild(actionButton("Verify", "Verify this answer against its sources", (b) => verify(c, m, li, b)));
    }
    if (meta.mode === "grounded_only" && meta.outcome === "answered") {
      actions.appendChild(el("span", "status-line", "Checked by guards and the critic"));
    }
    if (meta.model) actions.appendChild(el("span", "status-line", "· " + meta.model));
    return actions;
  }

  function verifyNode(v) {
    const box = el("section", "verify");
    box.setAttribute("aria-label", "Verification");
    if (v.error) {
      box.appendChild(el("p", "error-box", v.error));
      return box;
    }
    box.appendChild(el("h4", null, "Verification: " + v.supported + " of " + v.total + " sentence" + (v.total === 1 ? "" : "s") + " supported"));
    const ul = el("ul");
    for (const s of v.sentences || []) {
      const li = el("li", s.status);
      li.appendChild(el("span", "mark", s.status === "supported" ? "Supported" : "Unsupported"));
      const text = el("span", null, s.text);
      if (s.reasons && s.reasons.length) text.appendChild(el("span", "why", s.reasons.join("; ")));
      li.appendChild(text);
      ul.appendChild(li);
    }
    box.appendChild(ul);
    return box;
  }

  // ------------------------------------------------------------------ passage panel

  let panelOpener = null;

  function openPassage(p, opener) {
    panelOpener = opener || null;
    const panel = $("passage-panel");
    const coll = collectionById(p.source);
    $("panel-label").textContent = p.source === "upload" ? "Your upload · " + p.label : (coll ? coll.label : "Source") + " · " + p.label;
    $("panel-title").textContent = p.heading_path || p.title || "Passage";
    $("panel-meta").textContent = (p.title ? p.title + " · " : "") + (p.pages ? "Pages " + p.pages : "");
    $("panel-text").textContent = p.text || "(text not available)";
    panel.hidden = false;
    $("panel-title").focus();
  }

  function closePassage() {
    const panel = $("passage-panel");
    if (panel.hidden) return;
    panel.hidden = true;
    if (panelOpener && document.body.contains(panelOpener)) panelOpener.focus();
  }

  // ------------------------------------------------------------------ scrolling

  function nearBottom() {
    const s = $("scroller");
    return s.scrollHeight - s.scrollTop - s.clientHeight < 80;
  }

  function scrollToBottom(force) {
    const s = $("scroller");
    if (force || state.autoScroll) {
      s.scrollTop = s.scrollHeight;
      state.autoScroll = true;
      $("jump").hidden = true;
    }
  }

  // ------------------------------------------------------------------ sending

  function historyFor(c) {
    const out = [];
    for (const m of c.messages) {
      const meta = m.meta || {};
      if (m.role === "assistant" && (!m.content || meta.error || meta.status === "streaming")) continue;
      const content = m.role === "assistant" ? m.content.replace(/[ \t]*\[C\d{1,3}\]/g, "") : m.content;
      if (!content.trim()) continue;
      out.push({ role: m.role, content: content.slice(0, 20000) });
    }
    // the history must alternate sensibly and end with the user's message
    return out.slice(-MAX_SENT_MESSAGES);
  }

  function setBusy(busy) {
    $("stop-btn").hidden = !busy;
    $("stop-btn").disabled = false;
    $("send-btn").disabled = busy;
    $("collection").disabled = busy;
  }

  let renderPending = false;
  function scheduleRender(c, m) {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      if (m._node) renderAnswer(c, m, m._node, false);
      scrollToBottom(false);
    });
  }

  async function sendMessage(text) {
    const value = text.trim();
    if (!value || state.streaming) return;
    const c = ensureConversation();
    c.collection = selectedCollection();
    c.groundedOnly = $("grounded-only").checked;
    if (!c.title) {
      c.title = value.replace(/\s+/g, " ").slice(0, 60);
      $("chat-title").textContent = c.title;
    }
    c.messages.push({ role: "user", content: value });
    c.updated = Date.now();
    await runAssistant(c);
  }

  async function runAssistant(c) {
    const coll = collectionById(c.collection);
    const m = {
      role: "assistant",
      content: "",
      meta: {
        status: "streaming",
        collection: c.collection || "",
        notice: coll ? coll.notice : "",
        backend: state.backend,
        statusText: c.groundedOnly ? "Searching and checking the documents…" : "",
      },
    };
    c.messages.push(m);
    saveAll();
    renderConversationList();
    renderMessages();
    state.autoScroll = true;
    scrollToBottom(true);
    const controller = new AbortController();
    state.streaming = { controller, streamId: null, conv: c, msg: m };
    setBusy(true);
    announce("The assistant is replying.");
    const body = {
      messages: historyFor(c),
      collection: c.collection || null,
      session_id: state.sessionId,
      grounded_only: !!c.groundedOnly,
      backend: state.backend,
    };
    if (c.summary) body.summary = c.summary;
    try {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!r.ok) {
        m.meta.error = await errorText(r);
      } else {
        await readEvents(r, (ev) => handleEvent(c, m, ev));
      }
    } catch (err) {
      if (err && err.name === "AbortError") m.meta.stopped = true;
      else m.meta.error = "The connection was interrupted. Please try again.";
    }
    finishMessage(c, m);
  }

  function handleEvent(c, m, ev) {
    const d = ev.data || {};
    if (ev.event === "meta") {
      if (state.streaming) state.streaming.streamId = d.stream_id;
      m.meta.model = d.model;
      m.meta.backend = d.backend;
    } else if (ev.event === "context") {
      m.meta.context = d.passages || [];
      if (d.grounded) m.meta.statusText = "Found " + m.meta.context.length + " matching passage" + (m.meta.context.length === 1 ? "" : "s") + "…";
      scheduleRender(c, m);
    } else if (ev.event === "status") {
      m.meta.statusText = d.message || "";
      scheduleRender(c, m);
    } else if (ev.event === "token") {
      m.content += d.t || "";
      scheduleRender(c, m);
    } else if (ev.event === "done") {
      m.content = d.answer || "";
      m.meta.passages = d.passages || [];
      m.meta.citations = d.citations || [];
      m.meta.grounded = !!d.grounded;
      m.meta.sourceLine = d.source_line || "";
      m.meta.stopped = !!d.stopped;
      m.meta.model = d.model || m.meta.model;
      m.meta.mode = d.mode || "chat";
      m.meta.outcome = d.outcome;
      m.meta.closest = d.closest || [];
      m.meta.abstentionMessage = d.abstention_message;
      m.meta.timings = d.timings;
      if (d.summary) c.summary = d.summary;
      m.meta.done = true;
    } else if (ev.event === "error") {
      m.meta.error = d.message || "Something went wrong.";
    }
  }

  function finishMessage(c, m) {
    const meta = m.meta;
    meta.status = meta.error ? "error" : meta.stopped ? "stopped" : "done";
    if (!meta.done && !meta.error && m.content) {
      // stopped before the server's final event: keep the partial text, plain chat line
      meta.sourceLine = meta.sourceLine || "Stopped";
    }
    delete meta.statusText;
    state.streaming = null;
    setBusy(false);
    c.updated = Date.now();
    saveAll();
    renderMessages();
    renderConversationList();
    scrollToBottom(false);
    announce(meta.error ? "Error: " + meta.error : "Reply finished. " + (meta.sourceLine || ""));
  }

  // Stop: ask the server to end the reply (it then sends its final event with the cleaned
  // text); abort the request if that does not arrive quickly.
  function stopStreaming() {
    const s = state.streaming;
    if (!s || s.stopping) return;
    s.stopping = true;
    $("stop-btn").disabled = true;
    if (!s.streamId) {
      s.controller.abort();
      return;
    }
    fetch("/api/chat/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stream_id: s.streamId }),
    }).catch(() => {});
    setTimeout(() => {
      if (state.streaming === s) s.controller.abort();
    }, 1500);
  }

  function regenerate(c) {
    if (state.streaming) return;
    if (c.messages.length && c.messages[c.messages.length - 1].role === "assistant") c.messages.pop();
    if (!c.messages.length || c.messages[c.messages.length - 1].role !== "user") return;
    c.collection = selectedCollection();
    c.groundedOnly = $("grounded-only").checked;
    runAssistant(c);
  }

  function startEdit(c, m, li) {
    const box = el("div", "edit-box");
    const label = el("label", "visually-hidden", "Edit your message");
    const area = el("textarea");
    area.id = "edit-" + uid();
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
    const submit = () => {
      const v = area.value.trim();
      if (!v) return;
      const idx = c.messages.indexOf(m);
      c.messages.length = idx; // drop the message and everything after it
      c.messages.push({ role: "user", content: v });
      c.collection = selectedCollection();
      c.groundedOnly = $("grounded-only").checked;
      runAssistant(c);
    };
    save.addEventListener("click", submit);
    area.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submit();
      } else if (e.key === "Escape") {
        renderMessages();
      }
    });
  }

  async function verify(c, m, li, button) {
    button.disabled = true;
    button.textContent = "Verifying…";
    const idx = c.messages.indexOf(m);
    let question = "";
    for (let i = idx - 1; i >= 0; i--) {
      if (c.messages[i].role === "user") {
        question = c.messages[i].content;
        break;
      }
    }
    try {
      const r = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.slice(0, 4000) || "(question)",
          answer: m.content.slice(0, 20000),
          passages: (m.meta.passages || []).map((p) => ({ label: p.label, chunk_id: p.chunk_id })),
          collection: m.meta.collection || null,
          session_id: state.sessionId,
          backend: m.meta.backend || state.backend,
        }),
      });
      m.meta.verify = r.ok ? await r.json() : { error: await errorText(r) };
    } catch (err) {
      m.meta.verify = { error: "The verification request failed. Please try again." };
    }
    saveAll();
    renderAnswer(c, m, li, true);
    announce(m.meta.verify.error ? m.meta.verify.error : "Verification finished: " + m.meta.verify.supported + " of " + m.meta.verify.total + " sentences supported.");
  }

  // ------------------------------------------------------------------ uploads

  function renderUploads() {
    const list = $("upload-list");
    list.textContent = "";
    for (const d of state.uploads) {
      const li = el("li", "upload-item");
      li.appendChild(el("span", "name", d.name));
      li.appendChild(el("span", "meta", d.pages + " p."));
      const del = el("button", "icon-btn");
      del.type = "button";
      del.setAttribute("aria-label", "Remove " + d.name);
      del.appendChild(icon("M6.4 5 5 6.4 10.6 12 5 17.6 6.4 19l5.6-5.6 5.6 5.6 1.4-1.4-5.6-5.6L19 6.4 17.6 5 12 10.6z", 16));
      del.addEventListener("click", () => removeUpload(d));
      li.appendChild(del);
      list.appendChild(li);
    }
    const max = (state.health && state.health.uploads && state.health.uploads.max_files) || 3;
    $("upload-btn").disabled = state.uploading || state.uploads.length >= max;
  }

  function uploadStatus(text, percent, isError) {
    const box = $("upload-status");
    box.classList.toggle("error", !!isError);
    $("upload-text").textContent = text || "";
    const bar = $("upload-bar");
    if (percent === null || percent === undefined) bar.hidden = true;
    else {
      bar.hidden = false;
      bar.value = Math.max(0, Math.min(100, percent));
    }
  }

  function uploadFile(file) {
    const lim = (state.health && state.health.uploads) || { max_mb: 20, max_pages: 300, max_files: 3 };
    if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
      uploadStatus("Only PDF files can be uploaded.", null, true);
      return;
    }
    if (file.size > lim.max_mb * 1024 * 1024) {
      uploadStatus("The file is larger than " + lim.max_mb + " MB.", null, true);
      return;
    }
    state.uploading = true;
    renderUploads();
    uploadStatus("Uploading " + file.name + "…", 0, false);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.setRequestHeader("Content-Type", "application/pdf");
    xhr.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
    if (state.sessionId) xhr.setRequestHeader("X-Session-Id", state.sessionId);
    let seen = 0;
    let finished = false;
    const stages = { reading: "Reading pages", structuring: "Finding sections", embedding: "Indexing passages" };
    const handle = (ev) => {
      const d = ev.data || {};
      if (ev.event === "session") state.sessionId = d.session_id;
      else if (ev.event === "progress") {
        const pct = d.total ? (100 * d.done) / d.total : null;
        uploadStatus((stages[d.stage] || "Processing") + (d.total > 1 ? " " + d.done + "/" + d.total : "") + "…", pct, false);
      } else if (ev.event === "done") {
        finished = true;
        state.sessionId = d.session_id;
        state.uploads = d.documents || [];
        uploadStatus(d.document.name + " is ready (" + d.document.chunks + " passages). Ask about it.", null, false);
        announce(d.document.name + " is ready.");
      } else if (ev.event === "error") {
        finished = true;
        uploadStatus(d.message || "The upload failed.", null, true);
      }
    };
    const pump = () => {
      const text = xhr.responseText || "";
      let i;
      while ((i = text.indexOf("\n\n", seen)) >= 0) {
        const ev = parseEvent(text.slice(seen, i));
        seen = i + 2;
        if (ev) handle(ev);
      }
    };
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) uploadStatus("Uploading " + file.name + "…", (100 * e.loaded) / e.total, false);
    };
    xhr.onprogress = () => {
      if (xhr.status === 200) pump();
    };
    xhr.onload = () => {
      if (xhr.status === 200) pump();
      else {
        let msg = "The upload failed (HTTP " + xhr.status + ").";
        try {
          msg = JSON.parse(xhr.responseText).detail || msg;
        } catch (err) {
          /* keep */
        }
        uploadStatus(msg, null, true);
        finished = true;
      }
      if (!finished) uploadStatus("The upload did not finish. Please try again.", null, true);
      state.uploading = false;
      renderUploads();
    };
    xhr.onerror = () => {
      uploadStatus("The upload failed (network error).", null, true);
      state.uploading = false;
      renderUploads();
    };
    xhr.send(file);
  }

  async function removeUpload(d) {
    if (!state.sessionId) return;
    try {
      const r = await fetch("/api/upload/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, doc_id: d.doc_id }),
      });
      if (r.ok) state.uploads = (await r.json()).documents || [];
    } catch (err) {
      /* keep the list */
    }
    uploadStatus(d.name + " removed.", null, false);
    renderUploads();
  }

  function endSession() {
    if (!state.sessionId) return;
    const payload = JSON.stringify({ session_id: state.sessionId });
    try {
      if (navigator.sendBeacon) navigator.sendBeacon("/api/session/end", new Blob([payload], { type: "text/plain" }));
    } catch (err) {
      /* the server deletes it after 2 hours anyway */
    }
  }

  // ------------------------------------------------------------------ about

  function renderAbout() {
    const root = $("about-sources");
    root.textContent = "";
    root.appendChild(el("h3", null, "Document collections"));
    for (const c of state.collections) {
      const sec = el("div", "about-coll");
      const h = el("h3", null, c.label);
      h.appendChild(el("span", "tag", c.evaluated ? "Evaluated" : "Not evaluated"));
      sec.appendChild(h);
      if (c.notice) sec.appendChild(el("p", "hint", c.notice));
      const ul = el("ul", "about-list");
      for (const s of c.sources || []) {
        const li = el("li");
        li.appendChild(el("strong", null, s.title || "Document"));
        if (s.authors) li.appendChild(document.createTextNode(" — " + s.authors));
        if (s.edition) li.appendChild(document.createTextNode(" (" + s.edition + ")"));
        li.appendChild(document.createTextNode(". Licence: "));
        if (s.licence_url && MD.SAFE_URL.test(s.licence_url)) {
          const a = el("a", null, s.licence || s.licence_url);
          a.href = s.licence_url;
          a.rel = "noopener noreferrer";
          a.target = "_blank";
          li.appendChild(a);
        } else {
          li.appendChild(document.createTextNode(s.licence || "as stated in the source"));
        }
        if (s.source_url && MD.SAFE_URL.test(s.source_url)) {
          li.appendChild(document.createTextNode(". "));
          const a = el("a", null, "Source");
          a.href = s.source_url;
          a.rel = "noopener noreferrer";
          a.target = "_blank";
          li.appendChild(a);
        }
        ul.appendChild(li);
      }
      sec.appendChild(ul);
      root.appendChild(sec);
    }
  }

  // ------------------------------------------------------------------ composer

  function autosize() {
    const box = $("composer-input");
    box.style.height = "auto";
    box.style.height = Math.min(box.scrollHeight, window.innerHeight * 0.4) + "px";
    const n = box.value.length;
    const max = (state.health && state.health.limits && state.health.limits.max_message_chars) || 4000;
    $("char-count").textContent = n > max * 0.85 ? n + " / " + max : "";
  }

  // ------------------------------------------------------------------ wiring

  function wire() {
    $("composer").addEventListener("submit", (e) => {
      e.preventDefault();
      const box = $("composer-input");
      const text = box.value;
      if (!text.trim() || state.streaming) return;
      box.value = "";
      autosize();
      sendMessage(text);
    });
    $("composer-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        const form = $("composer");
        if (typeof form.requestSubmit === "function") form.requestSubmit();
        else form.dispatchEvent(new Event("submit", { cancelable: true }));
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
        c.groundedOnly = $("grounded-only").checked;
        saveAll();
      }
      updateModeBadge();
    });
    $("backend").addEventListener("change", () => {
      state.backend = $("backend").value;
      savePrefs({ backend: state.backend });
      updateBackend();
    });
    $("upload-btn").addEventListener("click", () => $("file-input").click());
    $("file-input").addEventListener("change", () => {
      const f = $("file-input").files && $("file-input").files[0];
      $("file-input").value = "";
      if (f) uploadFile(f);
    });
    $("open-sidebar").addEventListener("click", openSidebar);
    $("close-sidebar").addEventListener("click", closeSidebar);
    $("scrim").addEventListener("click", closeSidebar);
    $("panel-close").addEventListener("click", closePassage);
    $("about-btn").addEventListener("click", () => {
      renderAbout();
      const d = $("about");
      if (typeof d.showModal === "function") d.showModal();
      else d.setAttribute("open", "");
    });
    $("about-close").addEventListener("click", () => {
      const d = $("about");
      if (typeof d.close === "function") d.close();
      else d.removeAttribute("open");
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
    window.addEventListener("pagehide", endSession);
    window.addEventListener("storage", (e) => {
      if (e.key === STORE_KEY && !state.streaming) {
        loadAll();
        renderConversationList();
        renderMessages();
      }
    });
  }

  async function init() {
    loadAll();
    wire();
    if (!storage.ok) showStorageNote();
    try {
      const [cols, health] = await Promise.all([getJSON("/api/collections"), getJSON("/api/health")]);
      state.collections = cols.collections || [];
      state.health = health;
      if (health.public) $("research-link").hidden = true;
      const u = health.uploads;
      if (u) $("upload-privacy").textContent = "Up to " + u.max_files + " PDFs (" + u.max_mb + " MB, " + u.max_pages + " pages each). They are processed in memory only, never saved to disk or logs, and deleted when you close this page or after " + Math.round(u.ttl_s / 3600) + " hours of inactivity. " + (health.backends || []).filter((b) => b.hosted).map((b) => "With the hosted model, matching passages are sent to " + (b.provider || "the API provider") + ".").join(" ");
      fillCollections();
      fillBackends();
      const p = prefs();
      const firstColl = p.collection !== undefined ? p.collection : cols.default || "";
      applyCollection(firstColl);
    } catch (err) {
      $("collection-desc").textContent = "Could not reach the server: " + err.message;
    }
    renderConversationList();
    renderMessages();
    renderUploads();
    autosize();
    $("composer-input").focus();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
