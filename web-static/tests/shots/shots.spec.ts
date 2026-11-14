// Screenshots of the real app states for docs/screenshots/. The LLM proxy is a local mock
// (mock-worker.ts) that streams with real pauses; nothing calls the real Worker. Run: npm run shots

import { test, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { ask, waitReady } from "../e2e/helpers.ts";
import { MOCK_WORKER_PORT } from "../../playwright.shots.config.ts";
import { startMockWorker, type MockWorker } from "./mock-worker.ts";

const OUT = join(process.cwd(), "..", "docs", "screenshots");
mkdirSync(OUT, { recursive: true });

let worker: MockWorker;
test.beforeAll(async () => {
  worker = await startMockWorker(MOCK_WORKER_PORT);
});
test.afterAll(async () => {
  await worker.close();
});

async function selectCollection(page: Page, id: string): Promise<void> {
  const mobile = (page.viewportSize()?.width ?? 1440) < 700;
  await page.click(mobile ? "#topbar-collection" : "#collection-pill");
  await page.locator(`#collection-popover button[data-id="${id}"]`).click();
  await page.waitForSelector(`#collection-pill[data-status="ready"]`, { state: "attached", timeout: 120_000 });
}

async function newChat(page: Page): Promise<void> {
  const mobile = (page.viewportSize()?.width ?? 1440) < 700;
  if (mobile) {
    await page.click("#open-rail");
    await page.waitForTimeout(400);
  }
  await page.click("#new-chat");
  await page.waitForTimeout(400);
}

for (const scheme of ["dark", "light"] as const) {
  test.describe(scheme, () => {
    test.use({ colorScheme: scheme });

    test(`states (${scheme})`, async ({ page }, info) => {
      const device = info.project.name;
      const mobile = device !== "desktop";
      const shot = (name: string): Promise<Buffer> => page.screenshot({ path: join(OUT, `${name}-${device}-${scheme}.png`) });
      worker.setMode("answer");
      await page.goto("./");
      await waitReady(page);
      await page.waitForSelector("#hero-seal.is-ready", { timeout: 180_000 });
      await selectCollection(page, "cs_ai");
      await page.waitForTimeout(900);
      await shot("hero");

      // streaming mid-answer: the mock pauses after the first words
      worker.setMode("stall");
      await ask(page, "What does git rebase do?");
      await page.waitForSelector(".msg.assistant .answer .caret", { timeout: 60_000 });
      await page.waitForTimeout(700);
      await shot("streaming");
      worker.setMode("answer");
      await page.waitForSelector(".msg.assistant .msg-actions", { timeout: 60_000 });

      // finished, verified answer with the evidence open
      await newChat(page);
      await ask(page, "What does git rebase do, and when is it dangerous?");
      await page.waitForSelector(".msg.assistant .msg-actions", { timeout: 60_000 });
      await page.getByRole("button", { name: "Verify this answer against its sources" }).first().click();
      await page.waitForSelector('.seal-row[data-verdict="supported"]', { timeout: 60_000 });
      if (mobile) await page.getByRole("button", { name: "Show the evidence panel" }).first().click();
      await page.waitForTimeout(1000);
      await shot("verified-evidence");
      await page.click("#evidence-close");
      await page.waitForTimeout(500);

      // abstention (Grounded only)
      worker.setMode("abstain");
      await newChat(page);
      await selectCollection(page, "consumer_law");
      await page.click('label[for="grounded-only"]');
      await ask(page, "What is the capital of France?");
      await page.waitForSelector(".abstain-card", { timeout: 60_000 });
      await page.waitForTimeout(900);
      await shot("abstention");

      // rate limit
      worker.setMode("rate");
      await page.click('label[for="mode-chat"]');
      await ask(page, "Who may file a consumer complaint?");
      await page.waitForSelector(".error-card", { timeout: 60_000 });
      await page.waitForTimeout(400);
      await shot("rate-limit");

      // command palette
      await page.keyboard.press("Control+k");
      await page.waitForSelector("dialog.palette[open]");
      await page.keyboard.type("law");
      await page.waitForTimeout(500);
      await shot("palette");
      await page.keyboard.press("Escape");

      // how it works
      if (mobile) {
        await page.click("#open-rail");
        await page.waitForTimeout(500);
      }
      await page.click("#how-btn");
      await page.waitForSelector("dialog.how[open]");
      await page.waitForTimeout(1200);
      await shot("how-it-works");
    });
  });
}
