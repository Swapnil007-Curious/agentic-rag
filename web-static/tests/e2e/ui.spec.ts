// The static app with the LLM proxy mocked (page.route): streaming, automatic grounding with
// citation chips, invalid labels removed, Verify, Grounded only, Stop, friendly errors, uploads,
// stored conversations, 360 px and the Content-Security-Policy.

import { expect, test, type ConsoleMessage, type Page } from "@playwright/test";
import { ask, makePdf, mockProxy, waitReady } from "./helpers.ts";

function watchConsole(page: Page): string[] {
  const problems: string[] = [];
  page.on("console", (m: ConsoleMessage) => {
    if (m.type() === "error") problems.push(m.text());
  });
  page.on("pageerror", (e) => problems.push(e.message));
  return problems;
}

async function selectCollection(page: Page, id: string): Promise<void> {
  await page.selectOption("#collection", id);
  if (id) await expect(page.locator("#index-text")).toContainText("passages ready", { timeout: 120_000 });
}

test("general question streams a reply with the general-knowledge source line", async ({ page }) => {
  const problems = watchConsole(page);
  const calls = await mockProxy(page, () => "**Big-O** describes growth. For example `for x in xs` is $O(n)$.\n\n```python\nprint(1)\n```");
  await page.goto("./");
  await waitReady(page);
  await selectCollection(page, "");
  await ask(page, "write a poem about the sea and the stars");
  const reply = page.locator(".msg.assistant").last();
  await expect(reply.locator(".source-line")).toHaveText("General knowledge — no matching documents");
  await expect(reply.locator(".answer strong")).toHaveText("Big-O");
  await expect(reply.locator(".md-pre .md-lang")).toHaveText("python");
  await expect(reply.locator(".katex").first()).toBeVisible();
  expect(calls).toHaveLength(1);
  expect(calls[0]?.mode).toBe("chat");
  expect(calls[0]?.body.passages).toBeUndefined();
  expect(calls[0]?.body.messages).toEqual([{ role: "user", content: "write a poem about the sea and the stars" }]);
  await expect(page.locator("#model-text")).toContainText("Search model ready", { timeout: 120_000 });
  expect(problems).toEqual([]);
});

test("grounded question cites passages; invalid labels are removed; chips open the passage; Verify", async ({ page }) => {
  const problems = watchConsole(page);
  const calls = await mockProxy(page, (body) => {
    if (body.mode === "critic") return JSON.stringify({ verdict: "accept", unsupported_sentences: [], feedback: "" });
    return "Rebasing replays your commits on top of another branch [C1]. It rewrites history [C9].";
  });
  await page.goto("./");
  await waitReady(page);
  await selectCollection(page, "cs_ai");
  await ask(page, "What does git rebase do?");
  const reply = page.locator(".msg.assistant").last();
  await expect(reply.locator(".source-line .label-text")).toHaveText("Answered with documents (1 source)");
  await expect(reply.locator(".answer")).not.toContainText("[C9]");
  await expect(reply.locator(".answer .cite-inline")).toHaveText("C1");
  const sent = calls[0]?.body as { passages: { label: string; heading: string; text: string }[]; collection: { label: string } };
  expect(sent.passages.length).toBeGreaterThan(0);
  expect(sent.passages[0]?.label).toBe("C1");
  expect(sent.passages[0]?.heading).toContain("Rebas");
  expect(sent.collection.label).toBe("Computer Science, AI & Coding");
  await reply.locator(".src-chip").first().click();
  await expect(page.locator("#passage-panel")).toBeVisible();
  await expect(page.locator("#panel-text")).not.toBeEmpty();
  await page.click("#panel-close");
  await reply.getByRole("button", { name: "Verify this answer against its sources" }).click();
  await expect(reply.locator(".verify h4")).toContainText("of 2 sentences supported");
  await expect(reply.locator(".verify li.unsupported")).toContainText("no citation");
  expect(calls.map((c) => c.mode)).toEqual(["chat", "critic"]);
  expect(problems).toEqual([]);
});

test("grounded only runs synthesizer and critic through the proxy, or abstains", async ({ page }) => {
  let answer = true;
  const calls = await mockProxy(page, (body) => {
    if (body.mode === "critic") return JSON.stringify({ verdict: "accept", unsupported_sentences: [], feedback: "" });
    const passages = body.passages as { label: string; text: string }[];
    return answer
      ? JSON.stringify({ abstain: false, answer: `${passages[0]?.text.split(".")[0] ?? "x"} [C1].`, citations: ["C1"] })
      : JSON.stringify({ abstain: true, answer: "", citations: [] });
  });
  await page.goto("./");
  await waitReady(page);
  await selectCollection(page, "consumer_law");
  await page.check("#grounded-only");
  await ask(page, "Who may file a consumer complaint?");
  const reply = page.locator(".msg.assistant").last();
  await expect(reply.locator(".msg-actions")).toContainText("Checked by guards and the critic");
  await expect(reply.locator(".source-line .label-text")).toHaveText("Answered with documents (1 source)");
  expect(calls.map((c) => c.mode)).toEqual(["synth", "critic"]);
  expect(calls[0]?.body.profile).toBe("consumer_law");
  answer = false;
  await ask(page, "What is the capital of France?");
  const second = page.locator(".msg.assistant").last();
  await expect(second.locator(".source-line .label-text")).toHaveText("Not found in these documents");
  await expect(second.locator(".closest li").first()).toBeVisible();
});

test("Stop ends a reply; a busy proxy shows a friendly message", async ({ page }) => {
  let slow = true;
  await mockProxy(page, async () => {
    if (slow) {
      await new Promise((r) => setTimeout(r, 4000));
      return "late reply";
    }
    return { status: 429, json: { error: "busy", message: "The assistant is busy right now. Please try again in a minute.", retry_after: 60 } } as never;
  });
  await page.goto("./");
  await waitReady(page);
  await selectCollection(page, "");
  await ask(page, "tell me a long story");
  await expect(page.locator("#stop-btn")).toBeVisible();
  await page.click("#stop-btn");
  await expect(page.locator("#stop-btn")).toBeHidden();
  await expect(page.locator(".msg.assistant .error-box")).toHaveCount(0);
  slow = false;
  await ask(page, "another question");
  await expect(page.locator(".msg.assistant").last().locator(".error-box")).toHaveText("The assistant is busy right now. Please try again in a minute.");
});

test("an uploaded PDF is indexed in the browser and searched; non-PDFs are refused", async ({ page }) => {
  const calls = await mockProxy(page, () => "The station recorded 4.2 centimetres per day [C1].");
  await page.goto("./");
  await waitReady(page);
  await selectCollection(page, "");
  await page.setInputFiles("#file-input", { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("plain text") });
  await expect(page.locator("#upload-text")).toHaveText("Only PDF files can be uploaded.");
  const pdf = makePdf([
    ["Zorblat Field Station Report", "1 Introduction", "The Zorblat field station was founded in 2019 on the northern ridge.", "It monitors glacier melt with twelve automatic sensors."],
    ["2 Findings", "In July 2023 the Zorblat station recorded an average melt rate of 4.2 centimetres per day.", "The highest daily melt was 7.9 centimetres."],
  ]);
  await page.setInputFiles("#file-input", { name: "zorblat report.pdf", mimeType: "application/pdf", buffer: pdf });
  await expect(page.locator("#upload-text")).toContainText("is ready", { timeout: 120_000 });
  await expect(page.locator("#upload-list .upload-item")).toContainText("zorblat report.pdf");
  await ask(page, "What melt rate did the Zorblat station record in July 2023?");
  const reply = page.locator(".msg.assistant").last();
  await expect(reply.locator(".source-line .label-text")).toHaveText("Answered with documents (1 source)");
  await expect(reply.locator(".src-chip")).toContainText("zorblat report.pdf");
  const passages = calls[0]?.body.passages as { heading: string; text: string }[];
  expect(passages.some((p) => p.text.includes("4.2 centimetres"))).toBe(true);
  expect(passages[0]?.heading).toContain("zorblat report");
});

test("conversations persist in localStorage; the page works at 360 px", async ({ page }) => {
  await mockProxy(page, () => "A short answer.");
  await page.goto("./");
  await waitReady(page);
  await selectCollection(page, "");
  await ask(page, "Remember this conversation");
  await expect(page.locator(".msg.assistant .source-line")).toBeVisible();
  await page.reload();
  await waitReady(page);
  await expect(page.locator("#conv-list .conv-open")).toHaveText(["Remember this conversation"]);
  await page.setViewportSize({ width: 360, height: 740 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.click("#open-sidebar");
  await expect(page.locator("#sidebar")).toHaveClass(/open/);
  const csp = await page.locator('meta[http-equiv="Content-Security-Policy"]').getAttribute("content");
  expect(csp).toContain("connect-src 'self' https://proxy.test.invalid");
  expect(csp).toContain("script-src 'self' 'wasm-unsafe-eval'");
});
