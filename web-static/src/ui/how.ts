// "How it works": the real stages of the pipeline (web-static/src/gate.ts, engine.ts,
// grounded.ts, verify.ts), drawn in as they scroll into view. Loaded on first use.

import { el, svgEl } from "./dom.ts";
import { reducedMotion } from "./motion.ts";

interface Stage {
  title: string;
  text: string;
  where: string;
}

const STAGES: Stage[] = [
  { title: "Embed the question in your browser", text: "bge-small-en-v1.5 runs on ONNX Runtime inside a Web Worker on this page. The question is turned into a vector here; no server reads it to understand it.", where: "embed.worker.ts" },
  { title: "Search the collection", text: "Cosine similarity over the int8 index of every passage in the selected collection (and any PDF you attached). A per-collection threshold decides whether the documents really cover the question; below it, Veridex answers from general knowledge and says so.", where: "gate.ts · vectors.ts" },
  { title: "Pack the passages", text: "The best passages are labelled C1, C2, … and packed within the context budget. Only these passages, your question and the recent conversation go to the LLM proxy.", where: "gate.ts" },
  { title: "Draft the answer", text: "The model writes with a citation after each claim and the answer streams back. In Grounded only mode the synthesizer must return JSON that can be checked, and may abstain outright.", where: "engine.ts · deploy/worker" },
  { title: "Check against the sources", text: "Guards run here: every sentence must cite a given passage and every number must occur in the passages it cites. Then the critic model reviews the whole answer. Grounded only allows one revision round; in Chat you run Verify on demand.", where: "guards.ts · verify.ts · grounded.ts" },
  { title: "Answer or abstain", text: "A checked answer gets the seal with its real verdict. When the evidence is not there, Veridex says so and shows the closest sections instead of guessing.", where: "grounded.ts" },
];

let dialog: HTMLDialogElement | null = null;
let opener: HTMLElement | null = null;

function build(repo: string): HTMLDialogElement {
  const d = el("dialog", "dialog how");
  d.setAttribute("aria-labelledby", "how-title");
  const head = el("div", "dialog-head");
  head.appendChild(el("h2", "dialog-title", "How it works"));
  (head.firstChild as HTMLElement).id = "how-title";
  const close = el("button", "icon-btn");
  close.type = "button";
  close.setAttribute("aria-label", "Close How it works");
  close.appendChild(svgEl("svg", { viewBox: "0 0 24 24", width: 20, height: 20, "aria-hidden": "true" }));
  (close.firstChild as SVGElement).appendChild(svgEl("path", { d: "M6 6l12 12M18 6L6 18", fill: "none", stroke: "currentColor", "stroke-width": 1.8, "stroke-linecap": "round" }));
  close.addEventListener("click", () => d.close());
  head.appendChild(close);
  d.appendChild(head);

  const body = el("div", "how-body");
  body.appendChild(el("p", "how-lede", "Every answer walks the same path. Retrieval and the guards run in your browser; only the drafting and the critic call the model through the proxy."));
  const pipeline = el("div", "pipeline");
  const rail = el("div", "pipeline-rail");
  const svg = svgEl("svg", { "aria-hidden": "true" });
  const base = svgEl("path", { class: "pipeline-line", d: "M11 14 V 14" });
  const lit = svgEl("path", { class: "pipeline-line-lit", d: "M11 14 V 14" });
  svg.append(base, lit);
  rail.appendChild(svg);
  const steps = el("ol", "pipeline-steps");
  STAGES.forEach((s, i) => {
    const li = el("li", "pipeline-step");
    li.appendChild(el("span", "pipeline-node", String(i + 1)));
    li.appendChild(el("h3", null, s.title));
    li.appendChild(el("p", null, s.text));
    li.appendChild(el("span", "mono", s.where));
    steps.appendChild(li);
  });
  pipeline.append(rail, steps);
  body.appendChild(pipeline);
  const privacy = el("div", "how-privacy");
  privacy.appendChild(el("strong", null, "Privacy. "));
  privacy.appendChild(document.createTextNode("Retrieval runs in the browser. Only the question, the recent conversation and the retrieved passages go to the LLM proxy, which holds the provider key and keeps no message contents. "));
  const a = el("a", null, "Source code on GitHub");
  a.href = repo;
  a.rel = "noopener noreferrer";
  a.target = "_blank";
  privacy.appendChild(a);
  body.appendChild(privacy);
  d.appendChild(body);
  d.addEventListener("close", () => {
    if (opener && document.body.contains(opener)) opener.focus();
  });
  document.body.appendChild(d);

  // draw the connector as the steps enter the viewport
  const nodes = [...steps.querySelectorAll<HTMLElement>(".pipeline-node")];
  const layout = (): void => {
    const top = rail.getBoundingClientRect().top;
    const ys = nodes.map((n) => {
      const r = n.getBoundingClientRect();
      return r.top - top + r.height / 2;
    });
    const path = `M28 ${ys[0] ?? 0} V ${ys[ys.length - 1] ?? 0}`;
    base.setAttribute("d", path);
    lit.setAttribute("d", path);
    const len = (ys[ys.length - 1] ?? 0) - (ys[0] ?? 0);
    lit.style.strokeDasharray = String(len);
    lit.style.strokeDashoffset = String(len);
    lit.style.transition = reducedMotion() ? "none" : "stroke-dashoffset 600ms cubic-bezier(0.2, 0.8, 0.2, 1)";
    return;
  };
  let litTo = 0;
  const update = (): void => {
    const ys = nodes.map((n) => n.getBoundingClientRect().top);
    const first = ys[0] ?? 0;
    const last = ys[ys.length - 1] ?? 0;
    const total = last - first || 1;
    const y = ys[litTo] ?? first;
    lit.style.strokeDashoffset = String(Math.max(0, (1 - (y - first) / total) * total));
  };
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const li = e.target as HTMLElement;
        const i = [...steps.children].indexOf(li);
        if (reducedMotion()) li.classList.add("is-in");
        else setTimeout(() => li.classList.add("is-in"), (i % 3) * 80);
        litTo = Math.max(litTo, i);
        update();
        io.unobserve(li);
      }
    },
    { root: d, threshold: 0.4 },
  );
  d.addEventListener("close", () => {
    io.disconnect();
  });
  (d as HTMLDialogElement & { __layout?: () => void; __observe?: () => void }).__layout = layout;
  (d as HTMLDialogElement & { __observe?: () => void }).__observe = () => {
    litTo = 0;
    for (const li of steps.children) {
      li.classList.remove("is-in");
      io.observe(li);
    }
  };
  return d;
}

export function openHow(repo: string): void {
  opener = document.activeElement as HTMLElement | null;
  dialog ??= build(repo);
  const d = dialog as HTMLDialogElement & { __layout?: () => void; __observe?: () => void };
  d.scrollTop = 0;
  d.showModal();
  requestAnimationFrame(() => {
    d.__layout?.();
    d.__observe?.();
  });
}

export function closeHow(): boolean {
  if (dialog?.open) {
    dialog.close();
    return true;
  }
  return false;
}
