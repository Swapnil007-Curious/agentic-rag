// Document Answers: vanilla JS for the local UI. All document and model text is inserted with
// textContent (never innerHTML); Markdown arrives as a block tree built by markdown.js.
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);
  const form = $("ask-form");
  const question = $("question");
  const collectionSel = $("collection");
  const collectionNote = $("collection-note");
  const collectionBadge = $("collection-badge");
  const summaryLink = $("summary-link");
  const backendSel = $("backend");
  const backendDesc = $("backend-desc");
  const modeDesc = $("mode-desc");
  const safetyNote = $("safety-note");
  const indexNote = $("index-note");
  const systemSel = $("system");
  const systemDesc = $("system-desc");
  const plain = $("plain");
  const askBtn = $("ask-button");
  const statusEl = $("status");
  const result = $("result");
  const resultHeading = $("result-heading");
  const badge = $("outcome-badge");
  const generatorEl = $("generator");
  const warnings = $("warnings");
  const answerEl = $("answer");
  const abstainedEl = $("abstained");
  const abstainWhy = $("abstain-why");
  const notFound = $("not-found");
  const closestHeading = $("closest-heading");
  const closest = $("closest");
  const generalEl = $("general");
  const traceList = $("trace-list");
  const traceTotal = $("trace-total");
  const fbStatus = $("fb-status");
  const panel = $("panel");
  const layout = document.querySelector(".layout");
  const examplesBlock = $("examples-block");
  const examplesList = $("examples");
  const chatSection = $("chat");
  const chatLog = $("chat-log");
  const chatEmpty = $("chat-empty");
  const chatForm = $("chat-form");
  const chatInput = $("chat-input");
  const chatSend = $("chat-send");
  const useDocs = $("use-docs");
  const codeMode = $("code-mode");
  const MD = window.SafeMarkdown;

  const STOP = new Set(
    ("the a an and or of to in on for by with from that this these those which who whom whose " +
      "is are was were be been being shall may must can will would should under any such its it " +
      "their there than then also into upon other not no nor but if as at so per out within " +
      "has have had having does do did section rule sub clause act rules")
      .split(" ")
  );
  const STEP_NAMES = {
    retrieval: "Retrieval",
    context: "Context packed",
    threshold: "Retrieval-score check",
    synthesis: "Synthesis",
    guards: "Guards",
    critic: "Critic",
    revision_context: "Context extended",
  };
  const MODE_TEXT = {
    grounded: "Answers only from the documents, one citation per sentence; otherwise “Not found”.",
    hybrid: "Grounded first; if the documents do not answer, a clearly labelled general-knowledge answer.",
    chat: "A conversation in Markdown; optional passages from the documents; coding help. Code is never run.",
  };
  const STORE_KEY = "docanswers.collection";
  const MODE_KEY = "docanswers.mode";
  const BACKEND_KEY = "docanswers.backend";

  let systems = [];
  let meta = null; // /api/meta of the current collection
  let collection = null; // current collection id
  let current = null; // last /ask or /chat response
  let lastTrigger = null; // element that opened the panel
  let timer = null;
  let mode = "grounded";
  let backends = [];
  const chats = {}; // collection id -> [{role, content, response}]

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function fmtMs(ms) {
    if (ms === null || ms === undefined) return "";
    return ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms";
  }

  function setStatus(text, opts) {
    statusEl.textContent = "";
    statusEl.classList.toggle("error", Boolean(opts && opts.error));
    if (opts && opts.busy) statusEl.appendChild(el("span", "spinner"));
    statusEl.appendChild(document.createTextNode(text));
  }

  function store(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (err) {
      /* storage unavailable: defaults are used next time */
    }
  }

  function stored(key) {
    try {
      return window.localStorage.getItem(key);
    } catch (err) {
      return null;
    }
  }

  function safeUrl(url) {
    return /^https?:\/\//i.test(url || "") ? url : null;
  }

  function link(text, url) {
    const href = safeUrl(url);
    if (!href) return el("span", null, text);
    const a = el("a", "doc-link", text);
    a.href = href;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    return a;
  }

  async function errorText(res) {
    try {
      const body = await res.json();
      if (body && body.detail) return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (err) {
      /* not JSON */
    }
    return res.statusText || "request failed";
  }

  function backendLabel(name) {
    const b = backends.find((x) => x.name === name);
    return b ? b.label : name;
  }

  // ------------------------------------------------------------------ collections + metadata
  async function loadBackends() {
    try {
      const res = await fetch("/api/backends");
      const data = await res.json();
      backends = data.backends || [];
      backendSel.textContent = "";
      for (const b of backends) {
        const opt = el("option", null, b.label + (b.available ? "" : " (unavailable)"));
        opt.value = b.name;
        backendSel.appendChild(opt);
      }
      const saved = stored(BACKEND_KEY);
      const usable = (n) => backends.some((b) => b.name === n && b.available);
      backendSel.value = saved && usable(saved) ? saved : data.default || (backends[0] && backends[0].name) || "";
      backendSel.closest(".field-backend").hidden = backends.length === 0;
      updateBackendDesc();
    } catch (err) {
      setStatus("Could not load the models: " + err, { error: true });
    }
  }

  function updateBackendDesc() {
    const b = backends.find((x) => x.name === backendSel.value);
    if (!b) {
      backendDesc.textContent = "";
      $("privacy-note").textContent = "";
      return;
    }
    backendDesc.textContent = (b.available ? "" : "Unavailable: " + b.reason + " ") + (b.privacy || "");
    $("privacy-note").textContent = "Model: " + b.label + ". " + (b.privacy || "");
  }

  async function loadCollections() {
    try {
      const res = await fetch("/api/collections");
      const data = await res.json();
      collectionSel.textContent = "";
      for (const c of data.collections) {
        const tag = c.evaluated ? "" : " · not evaluated";
        const opt = el("option", null, c.label + tag + (c.available ? "" : " (not built yet)"));
        opt.value = c.id;
        collectionSel.appendChild(opt);
      }
      const ids = data.collections.map((c) => c.id);
      const saved = stored(STORE_KEY);
      collection = saved && ids.includes(saved) ? saved : data.default;
      collectionSel.value = collection;
      collectionSel.closest(".field-collection").hidden = data.collections.length < 2;
      await loadMeta();
    } catch (err) {
      setStatus("Could not load the collections: " + err, { error: true });
    }
  }

  async function loadMeta() {
    try {
      const res = await fetch("/api/meta?collection=" + encodeURIComponent(collection));
      meta = await res.json();
      $("eyebrow").textContent = meta.eyebrow || "";
      $("heading").textContent = meta.heading || "Ask the documents";
      $("lede").textContent = meta.lede || "";
      question.placeholder = meta.placeholder || "";
      notFound.textContent = meta.abstention_message || "Not found in these documents";
      systems = meta.systems;
      systemSel.textContent = "";
      for (const s of systems) {
        const opt = el("option", null, s.id + " — " + s.description);
        opt.value = s.id;
        if (s.id === meta.default_system) opt.selected = true;
        systemSel.appendChild(opt);
      }
      delete systemDesc.dataset.note;
      if (!meta.threshold_tuned && systems.some((s) => s.id === "G5")) {
        systemDesc.dataset.note = "G5's abstention threshold is not tuned yet; G5 currently runs like G4.";
      }
      updateSystemDesc();
      $("disclaimer").textContent = meta.disclaimer || "";
      collectionBadge.textContent = meta.evaluated ? "Evaluated" : "Not evaluated";
      collectionBadge.classList.toggle("not-evaluated", !meta.evaluated);
      summaryLink.hidden = !meta.summary_url;
      if (meta.summary_url) summaryLink.href = meta.summary_url;
      safetyNote.hidden = !meta.safety_note;
      safetyNote.textContent = meta.safety_note || "";
      indexNote.hidden = !meta.index_note;
      indexNote.textContent = meta.index_note || "";
      codeMode.checked = Boolean(meta.code_assistant);
      renderDocuments(meta.documents || []);
      renderExamples(meta.examples || []);
      collectionNote.textContent = meta.available ? "" : meta.note || "This collection is not built yet.";
      askBtn.disabled = !meta.available;
      chatSend.disabled = !meta.available;
      renderChatLog();
    } catch (err) {
      setStatus("Could not load the collection: " + err, { error: true });
    }
  }

  function renderDocuments(docs) {
    const list = $("documents");
    list.textContent = "";
    for (const d of docs) {
      const li = el("li", "document");
      li.appendChild(link(d.title, d.source_url));
      const by = [d.authors, d.edition].filter(Boolean).join(" · ");
      if (by) li.appendChild(el("span", "doc-by", " — " + by));
      const facts = el("span", "version");
      const parts = [d.file, d.pages_kept ? d.pages_kept + " of " + d.pages + " pages used" : d.pages ? d.pages + " pages" : "", d.version];
      facts.textContent = " (" + parts.filter(Boolean).join(" · ") + ")";
      li.appendChild(facts);
      if (d.licence) {
        const lic = el("span", "doc-licence");
        lic.appendChild(document.createTextNode(" Licence: "));
        lic.appendChild(link(d.licence, d.licence_url));
        if (d.licence_source) lic.title = d.licence_source;
        li.appendChild(lic);
      }
      list.appendChild(li);
    }
  }

  function renderExamples(examples) {
    examplesList.textContent = "";
    examplesBlock.hidden = examples.length === 0;
    for (const q of examples) {
      const li = el("li");
      const btn = el("button", "example-chip", q);
      btn.type = "button";
      btn.addEventListener("click", () => {
        if (mode === "chat") {
          chatInput.value = q;
          chatInput.focus();
        } else {
          question.value = q;
          question.focus();
        }
      });
      li.appendChild(btn);
      examplesList.appendChild(li);
    }
  }

  function updateSystemDesc() {
    const s = systems.find((x) => x.id === systemSel.value);
    let text = s ? s.description : "";
    if (systemSel.value === "G5" && systemDesc.dataset.note) text += " " + systemDesc.dataset.note;
    systemDesc.textContent = text;
  }

  async function switchCollection() {
    collection = collectionSel.value;
    store(STORE_KEY, collection);
    closePanel(false);
    result.hidden = true;
    current = null;
    setStatus("");
    await loadMeta();
  }

  function setMode(value) {
    mode = value;
    store(MODE_KEY, mode);
    for (const r of document.querySelectorAll('input[name="mode"]')) r.checked = r.value === mode;
    modeDesc.textContent = MODE_TEXT[mode] || "";
    const chat = mode === "chat";
    form.hidden = chat;
    chatSection.hidden = !chat;
    examplesBlock.parentNode.removeChild(examplesBlock);
    (chat ? chatForm : form).insertBefore(examplesBlock, chat ? chatForm.firstChild : form.querySelector(".controls"));
    if (!chat) result.hidden = !current || current.mode === "chat";
    else result.hidden = true;
  }

  // ------------------------------------------------------------------ ask (grounded / hybrid)
  function startBusy(btn, label) {
    btn.disabled = true;
    btn.textContent = "Working…";
    const started = Date.now();
    const tick = () =>
      setStatus(label + " " + Math.round((Date.now() - started) / 1000) + " s", { busy: true });
    tick();
    timer = setInterval(tick, 1000);
  }

  function stopBusy(btn, text) {
    clearInterval(timer);
    btn.disabled = Boolean(meta && !meta.available);
    btn.textContent = text;
  }

  async function ask(evt) {
    if (evt) evt.preventDefault();
    const q = question.value.trim();
    if (q.length < 3) {
      setStatus("Please type a question first.", { error: true });
      question.focus();
      return;
    }
    closePanel(false);
    startBusy(askBtn, "Retrieving passages and writing the answer…");
    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          system: systemSel.value,
          plain: plain.checked,
          collection: collection,
          mode: mode === "hybrid" ? "hybrid" : "grounded",
          backend: backendSel.value || null,
        }),
      });
      if (!res.ok) throw new Error(await errorText(res));
      const body = await res.json();
      current = body;
      render(body);
      setStatus("Done in " + fmtMs(body.cost.e2e_ms + ((body.general && body.general.ms) || 0)) + ".");
      resultHeading.focus();
    } catch (err) {
      setStatus("Something went wrong: " + err.message, { error: true });
    } finally {
      stopBusy(askBtn, "Ask");
    }
  }

  // ------------------------------------------------------------------ rendering
  function generatorText(g) {
    if (!g) return "";
    return "Answered by " + backendLabel(g.backend) + (g.model ? " — model " + g.model : "");
  }

  function render(data) {
    result.hidden = false;
    fbStatus.textContent = "";
    for (const b of document.querySelectorAll(".fb-btn")) b.setAttribute("aria-pressed", "false");
    warnings.textContent = "";
    const allWarnings = (data.warnings || []).concat(data.mode_warnings || []);
    warnings.hidden = allWarnings.length === 0;
    for (const w of allWarnings) warnings.appendChild(el("li", null, w));
    generatorEl.textContent = generatorText(data.generator);

    const answered = data.outcome === "answered";
    badge.textContent = answered ? "Answered from the documents · " + data.system : "Not found · " + data.system;
    badge.classList.toggle("abstained", !answered);
    answerEl.textContent = "";
    answerEl.hidden = !answered;
    abstainedEl.hidden = answered;
    generalEl.hidden = true;

    if (answered) {
      const p = el("p");
      data.sentences.forEach((s, i) => {
        const span = el("span", "sentence", s.text);
        span.id = "sentence-" + i;
        p.appendChild(span);
        for (const label of s.citations) p.appendChild(chip(label, s, span, data.passages));
        p.appendChild(document.createTextNode(" "));
      });
      answerEl.appendChild(p);
    } else {
      notFound.textContent = data.abstention_message || (meta && meta.abstention_message) || "Not found in these documents";
      abstainWhy.textContent = reasonText(data.reason, data.detail);
      closestHeading.textContent = data.general ? "Related reading" : "Closest sections";
      closest.textContent = "";
      for (const c of data.closest_sections || []) {
        const li = el("li");
        const btn = el("button", "link-btn");
        btn.type = "button";
        btn.appendChild(el("span", "path", c.heading_path));
        btn.appendChild(el("span", "sub", (c.section_ids.join(", ") || "-") + " · pages " + c.pages));
        btn.setAttribute("aria-label", "Open passage: " + c.heading_path);
        btn.addEventListener("click", () => openClosest(c.chunk_id, btn));
        li.appendChild(btn);
        closest.appendChild(li);
      }
      if (data.general) renderGeneral(data.general);
    }
    renderTrace(data);
  }

  function renderGeneral(g) {
    generalEl.hidden = false;
    $("general-banner").textContent = g.banner;
    const box = $("general-answer");
    box.textContent = "";
    box.appendChild(MD.render(g.blocks, { headingBase: 4 }));
    $("general-model").textContent = generatorText({ backend: g.backend, model: g.model });
  }

  function reasonText(reason, detail) {
    const map = {
      synth_abstain: "The passages retrieved for this question do not contain an answer.",
      guard_fail: "An answer was drafted but failed the citation and number checks, so it is not shown.",
      critic_abstain: "The checking step could not confirm an answer from the passages.",
      low_retrieval_score: "No passage matched the question closely enough.",
    };
    return (map[reason] || "No supported answer was found.") + (detail ? " (" + detail + ")" : "");
  }

  function chip(label, sentence, span, passages) {
    const passage = passages ? passages[label] : null;
    const btn = el("button", "chip", label.replace(/^C/, ""));
    btn.type = "button";
    if (!passage) {
      btn.classList.add("missing");
      btn.setAttribute("aria-label", "Citation " + label + " (not in the retrieved passages)");
      btn.disabled = true;
      return btn;
    }
    btn.setAttribute("aria-label", "Show source " + label + ": " + passage.heading_path);
    btn.setAttribute("aria-controls", "panel");
    btn.setAttribute("aria-expanded", "false");
    btn.addEventListener("click", () => {
      for (const s of document.querySelectorAll(".sentence.active")) s.classList.remove("active");
      for (const c of document.querySelectorAll(".chip[aria-expanded='true']")) c.setAttribute("aria-expanded", "false");
      if (span) span.classList.add("active");
      btn.setAttribute("aria-expanded", "true");
      const terms = sentence ? keyTerms(sentence.text) : [];
      openPanel(passage, label, terms, btn, "Key terms of the cited sentence are highlighted.");
    });
    return btn;
  }

  function renderTrace(data) {
    traceList.textContent = "";
    traceTotal.textContent = "· " + fmtMs(data.cost.e2e_ms) + ", " + data.cost.llm_calls + " LLM call(s)";
    for (const s of data.steps) {
      const li = el("li");
      const bad = ["unsupported", "below", "abstain", "error", "revise"].includes(s.status);
      if (bad) li.classList.add("bad");
      const row = el("div", "trace-row");
      let name = STEP_NAMES[s.name] || s.name;
      if (s.name === "synthesis" && s.round > 1) name = "Revision";
      if (s.round > 1 && s.name !== "synthesis") name += " (round " + s.round + ")";
      row.appendChild(el("span", "trace-name", name));
      row.appendChild(el("span", "trace-status", s.status));
      row.appendChild(el("span", "trace-ms", fmtMs(s.ms)));
      li.appendChild(row);
      const detail = traceDetail(s);
      if (detail) li.appendChild(el("p", "trace-detail", detail));
      traceList.appendChild(li);
    }
  }

  function traceDetail(s) {
    const d = s.detail || {};
    const calls = s.llm_calls || [];
    let usage = "";
    if (calls.length) {
      const pt = calls.reduce((a, c) => a + c.prompt_tokens, 0);
      const ct = calls.reduce((a, c) => a + c.completion_tokens, 0);
      usage = " " + calls.length + " LLM call(s), " + pt + " prompt + " + ct + " completion tokens.";
    }
    switch (s.name) {
      case "retrieval":
        return (d.config_id || "") + (d.abstention_score != null ? " · top score " + d.abstention_score.toFixed(3) : "");
      case "context":
      case "revision_context":
        return d.n_passages + " passages, " + d.used_tokens + " of " + d.budget_tokens + " tokens" +
          (d.added && d.added.length ? " (added " + d.added.join(", ") + ")" : "");
      case "threshold":
        return "score " + d.score + " vs threshold " + d.threshold;
      case "guards":
        return d.reasons && d.reasons.length ? d.reasons.join("; ") : "Every sentence cited; every number found in its sources.";
      case "critic":
        return (d.feedback || (d.verdict === "accept" ? "Accepted." : "")) + usage;
      case "synthesis":
        return (d.error || d.note || "") + usage;
      default:
        return usage.trim();
    }
  }

  // ------------------------------------------------------------------ chat
  function chatHistory() {
    if (!chats[collection]) chats[collection] = [];
    return chats[collection];
  }

  function renderChatLog() {
    const hist = chatHistory();
    chatLog.textContent = "";
    chatEmpty.hidden = hist.length > 0;
    hist.forEach((m, i) => chatLog.appendChild(chatItem(m, i)));
  }

  function chatItem(m, i) {
    const li = el("li", "chat-msg " + (m.role === "user" ? "from-user" : "from-model"));
    const who = el("p", "chat-who", m.role === "user" ? "You" : "Assistant");
    li.appendChild(who);
    const body = el("div", "md chat-body");
    if (m.role === "user") {
      for (const line of m.content.split("\n")) body.appendChild(el("p", null, line));
    } else if (m.response && m.response.chat) {
      const c = m.response.chat;
      body.appendChild(
        MD.render(c.blocks, {
          headingBase: 4,
          cite: (label) => (c.passages && c.passages[label] ? chip(label, null, null, c.passages) : null),
        })
      );
      const info = el("p", "generator", generatorText(m.response.generator) + (c.use_documents ? " · with documents" : "") + (c.code ? " · code assistant" : ""));
      li.appendChild(body);
      li.appendChild(info);
      const notes = (m.response.mode_warnings || []).filter(Boolean);
      if (notes.length) li.appendChild(el("p", "hint", notes.join(" ")));
      li.dataset.index = String(i);
      return li;
    } else {
      body.appendChild(el("p", null, m.content));
    }
    li.appendChild(body);
    return li;
  }

  async function sendChat(evt) {
    if (evt) evt.preventDefault();
    const text = chatInput.value.trim();
    if (!text) {
      setStatus("Please type a message first.", { error: true });
      chatInput.focus();
      return;
    }
    const hist = chatHistory();
    hist.push({ role: "user", content: text });
    chatInput.value = "";
    renderChatLog();
    startBusy(chatSend, "Writing the reply…");
    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: hist.map((m) => ({ role: m.role, content: m.content })),
          collection: collection,
          backend: backendSel.value || null,
          use_documents: useDocs.checked,
          code: codeMode.checked,
        }),
      });
      if (!res.ok) throw new Error(await errorText(res));
      const body = await res.json();
      current = body;
      hist.push({ role: "assistant", content: body.chat.answer, response: body });
      renderChatLog();
      setStatus("Reply in " + fmtMs(body.mode_ms) + ".");
      const last = chatLog.lastElementChild;
      if (last) last.scrollIntoView({ block: "nearest" });
    } catch (err) {
      hist.pop();
      chatInput.value = text;
      renderChatLog();
      setStatus("Something went wrong: " + err.message, { error: true });
    } finally {
      stopBusy(chatSend, "Send");
    }
  }

  function newChat() {
    chats[collection] = [];
    renderChatLog();
    setStatus("Started a new chat.");
    chatInput.focus();
  }

  // ------------------------------------------------------------------ passage panel
  function keyTerms(text) {
    const words = (text.toLowerCase().match(/[a-z][a-z\-']{3,}|\d[\d,]*/g) || []).filter((w) => !STOP.has(w));
    return Array.from(new Set(words)).slice(0, 24);
  }

  function escapeRe(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function highlighted(text, terms) {
    const frag = document.createDocumentFragment();
    if (!terms.length) {
      frag.appendChild(document.createTextNode(text));
      return frag;
    }
    const re = new RegExp("\\b(" + terms.map(escapeRe).sort((a, b) => b.length - a.length).join("|") + ")", "gi");
    let last = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      frag.appendChild(el("mark", null, m[0]));
      last = m.index + m[0].length;
      if (m[0].length === 0) re.lastIndex++;
    }
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
    return frag;
  }

  function openPanel(passage, label, terms, trigger, note) {
    lastTrigger = trigger;
    $("panel-label").textContent = label ? "Source " + label : "Passage";
    $("panel-heading").textContent = passage.heading_path;
    $("panel-section").textContent = (passage.section_ids || []).join(", ") || "-";
    $("panel-pages").textContent = passage.pages;
    $("panel-note").textContent = terms.length ? note : "";
    const textEl = $("panel-text");
    textEl.textContent = "";
    textEl.appendChild(highlighted(passage.text, terms));
    panel.hidden = false;
    layout.classList.add("has-panel");
    $("panel-heading").focus();
  }

  function closePanel(restoreFocus) {
    if (panel.hidden) return;
    panel.hidden = true;
    layout.classList.remove("has-panel");
    for (const c of document.querySelectorAll(".chip[aria-expanded='true']")) c.setAttribute("aria-expanded", "false");
    for (const s of document.querySelectorAll(".sentence.active")) s.classList.remove("active");
    if (restoreFocus !== false && lastTrigger) lastTrigger.focus();
  }

  async function openClosest(chunkId, trigger) {
    try {
      const url = "/passage?chunk_id=" + encodeURIComponent(chunkId) + "&collection=" + encodeURIComponent(collection);
      const res = await fetch(url);
      if (!res.ok) throw new Error(await errorText(res));
      const passage = await res.json();
      const terms = current && current.question ? keyTerms(current.question) : [];
      openPanel(passage, "", terms, trigger, "Words from your question are highlighted.");
    } catch (err) {
      setStatus("Could not open the passage: " + err.message, { error: true });
    }
  }

  // ------------------------------------------------------------------ feedback
  async function sendFeedback(btn) {
    if (!current || current.mode === "chat") return;
    const rating = btn.dataset.rating;
    try {
      const res = await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: current.question,
          system: current.system,
          outcome: current.outcome,
          answer: current.answer || (current.general ? current.general.answer.slice(0, 5000) : ""),
          rating: rating,
          collection: current.collection || collection,
          mode: current.mode || "grounded",
          model: (current.generator && current.generator.model) || "",
        }),
      });
      if (!res.ok) throw new Error(await errorText(res));
      for (const b of document.querySelectorAll(".fb-btn")) b.setAttribute("aria-pressed", String(b === btn));
      fbStatus.textContent = "Thanks — saved on this computer.";
    } catch (err) {
      fbStatus.textContent = "Could not save feedback: " + err.message;
    }
  }

  // ------------------------------------------------------------------ wiring
  form.addEventListener("submit", ask);
  question.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask(e);
  });
  chatForm.addEventListener("submit", sendChat);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) sendChat(e);
  });
  $("new-chat").addEventListener("click", newChat);
  collectionSel.addEventListener("change", switchCollection);
  backendSel.addEventListener("change", () => {
    store(BACKEND_KEY, backendSel.value);
    updateBackendDesc();
  });
  for (const r of document.querySelectorAll('input[name="mode"]')) {
    r.addEventListener("change", () => {
      if (r.checked) setMode(r.value);
    });
  }
  systemSel.addEventListener("change", updateSystemDesc);
  $("panel-close").addEventListener("click", () => closePanel(true));
  panel.addEventListener("click", (e) => {
    if (e.target === panel) closePanel(true); // backdrop on small screens
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !panel.hidden) closePanel(true);
  });
  for (const b of document.querySelectorAll(".fb-btn")) b.addEventListener("click", () => sendFeedback(b));
  const savedMode = stored(MODE_KEY);
  setMode(["grounded", "hybrid", "chat"].includes(savedMode) ? savedMode : "grounded");
  loadBackends().then(loadCollections);
})();
