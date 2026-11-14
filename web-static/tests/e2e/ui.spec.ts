// Veridex with the LLM proxy mocked (page.route): streaming, automatic grounding with citation
// chips, invalid labels removed, the evidence panel, Verify and the seal, Grounded only and
// abstention, Stop, limit errors with their countdown, share links, uploads, stored chats, the
// command palette, reduced motion, 360 px and the Content-Security-Policy.

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

/** Choose a collection through the composer pill's popover ("" = no collection). */
async function selectCollection(page: Page, id: string): Promise<void> {
  await page.click("#collection-pill");
  const pop = page.locator("#collection-popover");
  await expect(pop).toBeVisible();
  if (id) await pop.locator(`button[data-id="${id}"]`).click();
  else await pop.getByRole("radio", { name: /No collection/ }).click();
  await expect(pop).toBeHidden();
  if (id) await expect(page.locator("#collection-pill")).toHaveAttribute("data-status", "ready", { timeout: 120_000 });
}

test("general question streams a reply with the general-knowledge source line", async ({ page }) => {
  const problems = watchConsole(page);
  const calls = await mockProxy(page, () => "**Big-O** describes growth. For example `for x in xs` is $O(n)$.\n\n```python\nprint(1)\n```");
  await page.goto("./");
  await waitReady(page);
  await selectCollection(page, "");
  await ask(page, "write a poem about the sea and the stars");
  const reply = page.locator(".msg.assistant").last();
  await expect(reply.locator(".source-line .label-text")).toHaveText("General knowledge — no matching documents");
  await expect(reply.locator(".answer strong")).toHaveText("Big-O");
  await expect(reply.locator(".md-pre .md-lang")).toHaveText("python");
  await expect(reply.locator(".katex").first()).toBeVisible(); // KaTeX was loaded on demand
  expect(calls).toHaveLength(1);
  expect(calls[0]?.mode).toBe("chat");
  expect(calls[0]?.body.passages).toBeUndefined();
  expect(calls[0]?.body.messages).toEqual([{ role: "user", content: "write a poem about the sea and the stars" }]);
  await expect(page.locator("#model-text")).toContainText("Search index ready", { timeout: 120_000 });
  expect(problems).toEqual([]);
});

test("grounded question cites passages; invalid labels are removed; chips link to source cards; Verify stamps the seal", async ({ page }) => {
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
  await expect(reply.locator(".source-row .label-text")).toHaveText("Answered with documents (1 source)");
  await expect(reply.locator(".answer")).not.toContainText("[C9]");
  await expect(reply.locator(".answer .cite-inline")).toHaveText("C1");
  await expect(reply.locator(".trace-strip .trace-step")).toHaveCount(4); // Embed, Search, Draft, Verify
  await expect(reply.locator('.trace-step[data-status="done"]')).toHaveCount(3);
  const sent = calls[0]?.body as { passages: { label: string; heading: string; text: string }[]; collection: { label: string } };
  expect(sent.passages.length).toBeGreaterThan(0);
  expect(sent.passages[0]?.label).toBe("C1");
  expect(sent.passages[0]?.heading).toContain("Rebas");
  expect(sent.collection.label).toBe("Computer Science, AI & Coding");
  // the evidence panel opened by itself on desktop; the chip links to its card
  const evidence = page.locator("#evidence");
  await expect(evidence).toBeVisible();
  await expect(evidence.locator('.source-card[data-label="C1"]')).toHaveCount(1);
  await reply.locator(".answer .cite-inline").hover();
  await expect(evidence.locator('.source-card[data-label="C1"]')).toHaveClass(/is-linked/);
  await reply.locator(".src-chip").first().click();
  await expect(evidence.locator(".source-card.is-open .source-excerpt")).not.toBeEmpty();
  await expect(evidence.locator(".score-num")).toContainText("cos 0.");
  await page.click("#evidence-close");
  await expect(evidence).toBeHidden();
  await reply.getByRole("button", { name: "Verify this answer against its sources" }).first().click();
  await expect(reply.locator(".verify h4")).toContainText("of 2 sentences supported");
  await expect(reply.locator(".verify li.unsupported")).toContainText("no citation");
  await expect(reply.locator(".seal-row")).toHaveAttribute("data-verdict", "partial");
  await expect(reply.locator(".seal-label")).toHaveText("Partially supported");
  expect(calls.map((c) => c.mode)).toEqual(["chat", "critic"]);
  expect(problems).toEqual([]);
});

test("grounded only runs synthesizer and critic through the proxy and seals the answer, or abstains calmly", async ({ page }) => {
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
  await page.click('label[for="grounded-only"]');
  await expect(page.locator("#grounded-only")).toBeChecked();
  await ask(page, "Who may file a consumer complaint?");
  const reply = page.locator(".msg.assistant").last();
  await expect(reply.locator(".msg-actions")).toContainText("Checked by guards and the critic");
  await expect(reply.locator(".seal-row")).toHaveAttribute("data-verdict", "supported");
  await expect(reply.locator(".seal-label")).toHaveText("Verified against 1 source");
  await expect(reply.locator(".trace-strip .trace-step")).toHaveCount(5); // Embed, Search, Synthesize, Guards, Critic
  expect(calls.map((c) => c.mode)).toEqual(["synth", "critic"]);
  expect(calls[0]?.body.profile).toBe("consumer_law");
  answer = false;
  await ask(page, "What is the capital of France?");
  const second = page.locator(".msg.assistant").last();
  await expect(second.locator(".abstain-card .label-text")).toHaveText("Not found in these documents");
  await expect(second.locator(".abstain-title")).toContainText("Not enough evidence in Consumer Law");
  await expect(second.locator(".closest li").first()).toBeVisible();
  await expect(second.locator(".error-card")).toHaveCount(0);
});

test("Stop ends a reply; a rate limit shows a countdown and keeps the question in the composer", async ({ page }) => {
  let mode: "slow" | "busy" | "rate" = "slow";
  await mockProxy(page, async () => {
    if (mode === "slow") {
      await new Promise((r) => setTimeout(r, 4000));
      return "late reply";
    }
    if (mode === "busy") return { status: 429, json: { error: "busy", message: "The assistant is busy right now. Please try again in a minute.", retry_after: 60 } } as never;
    return { status: 429, json: { error: "rate_limited", message: "You have sent many messages in a short time.", retry_after: 125 } } as never;
  });
  await page.goto("./");
  await waitReady(page);
  await selectCollection(page, "");
  await ask(page, "tell me a long story");
  await expect(page.locator("#stop-btn")).toBeVisible();
  await page.click("#stop-btn");
  await expect(page.locator("#stop-btn")).toBeHidden();
  await expect(page.locator(".msg.assistant .error-card")).toHaveCount(0);
  mode = "busy";
  await ask(page, "another question");
  await expect(page.locator(".msg.assistant").last().locator(".error-box")).toHaveText("The answer service is busy.");
  await expect(page.locator(".msg.assistant").last().locator(".error-detail")).toContainText(/Try again in 00:5\d/);
  await expect(page.locator("#composer-input")).toHaveValue("another question");
  mode = "rate";
  await page.fill("#composer-input", "");
  await ask(page, "and one more");
  const last = page.locator(".msg.assistant").last();
  await expect(last.locator(".error-box")).toHaveText("Limit reached: 20 questions per 10 minutes.");
  await expect(last.locator(".error-detail")).toContainText(/Try again in 02:0\d/);
  await expect(page.locator("#composer-input")).toHaveValue("and one more");
  // the failed exchange is replaced by the resend, never duplicated
  expect(await page.locator(".msg.user").count()).toBe(2);
  await expect(page.locator(".msg.user .bubble-user").last()).toHaveText("and one more");
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
  await expect(reply.locator(".source-row .label-text")).toHaveText("Answered with documents (1 source)");
  await expect(reply.locator(".src-chip")).toContainText("zorblat report.pdf");
  const passages = calls[0]?.body.passages as { heading: string; text: string }[];
  expect(passages.some((p) => p.text.includes("4.2 centimetres"))).toBe(true);
  expect(passages[0]?.heading).toContain("zorblat report");
});

test("chats persist in localStorage and can be deleted with Undo; the page works at 360 px", async ({ page }) => {
  await mockProxy(page, () => "A short answer.");
  await page.goto("./");
  await waitReady(page);
  await selectCollection(page, "");
  await ask(page, "Remember this conversation");
  await expect(page.locator(".msg.assistant .source-line")).toBeVisible();
  await page.reload();
  await waitReady(page);
  await expect(page.locator("#conv-list .conv-open")).toHaveText(["Remember this conversation"]);
  await page.locator("#conv-list .conv-del").first().click({ force: true });
  await expect(page.locator("#conv-list .conv-open")).toHaveCount(0);
  await page.getByRole("button", { name: "Undo" }).click();
  await expect(page.locator("#conv-list .conv-open")).toHaveText(["Remember this conversation"]);
  await page.setViewportSize({ width: 360, height: 740 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.click("#open-rail");
  await expect(page.locator("#rail")).toHaveClass(/open/);
  const csp = await page.locator('meta[http-equiv="Content-Security-Policy"]').getAttribute("content");
  expect(csp).toContain("connect-src 'self' https://proxy.test.invalid");
  expect(csp).toContain("script-src 'self' 'wasm-unsafe-eval'");
});

test("a share link pre-fills the composer and the collection without sending", async ({ page }) => {
  const calls = await mockProxy(page, () => "never called");
  await page.goto("./?q=What%20does%20Article%2021%20protect%3F&c=indian_law");
  await waitReady(page);
  await expect(page.locator("#composer-input")).toHaveValue("What does Article 21 protect?");
  await expect(page.locator("#pill-label")).toHaveText("Indian Law");
  expect(new URL(page.url()).search).toBe("");
  await page.waitForTimeout(500);
  expect(calls).toHaveLength(0);
  expect(await page.locator(".msg").count()).toBe(0);
});

test("the command palette opens with Ctrl+K and switches collections; reduced motion sets the class", async ({ page }) => {
  await page.goto("./");
  await waitReady(page);
  await page.keyboard.press("Control+k");
  const palette = page.locator("dialog.palette");
  await expect(palette).toBeVisible();
  await page.keyboard.type("human bio");
  await page.keyboard.press("Enter");
  await expect(palette).toBeHidden();
  await expect(page.locator("#pill-label")).toHaveText("Human Biology");
  await expect(page.locator("html")).not.toHaveClass(/reduce-motion/);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(page.locator("html")).toHaveClass(/reduce-motion/);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await expect(page.locator("html")).not.toHaveClass(/reduce-motion/);
  await page.click("#theme-toggle");
  await expect(page.locator("html")).toHaveAttribute("data-theme", /light|dark/);
});
