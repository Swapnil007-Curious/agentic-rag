// Evaluation summary page: SUMMARY.md of an evaluated collection, rendered from the server's
// block tree with text nodes only (see markdown.js).
"use strict";

(function () {
  const params = new URLSearchParams(window.location.search);
  const collection = params.get("collection") || "";
  const box = document.getElementById("summary");
  const status = document.getElementById("summary-status");

  async function load() {
    try {
      const res = await fetch("/api/summary?collection=" + encodeURIComponent(collection));
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || res.statusText);
      document.getElementById("summary-title").textContent = body.label + ": evaluation summary";
      document.title = body.label + " — evaluation summary";
      box.textContent = "";
      box.appendChild(window.SafeMarkdown.render(body.blocks, { headingBase: 2 }));
    } catch (err) {
      status.textContent = "Could not load the summary: " + err.message;
      status.classList.add("error");
    }
  }

  load();
})();
