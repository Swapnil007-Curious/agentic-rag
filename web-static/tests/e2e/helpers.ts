import type { Page, Route } from "@playwright/test";

export const PROXY = "https://proxy.test.invalid";

/** A streamed chat completion in the provider's SSE format, as the proxy passes it through. */
export function sse(text: string, pieces = 6): string {
  const words = text.match(/\S+\s*/g) ?? [text];
  const size = Math.max(1, Math.ceil(words.length / pieces));
  let out = "";
  for (let i = 0; i < words.length; i += size) {
    const delta = words.slice(i, i + size).join("");
    out += `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: delta }, finish_reason: null }] })}\n\n`;
  }
  out += `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\n`;
  return `${out}data: [DONE]\n\n`;
}

export interface ProxyCall {
  mode: string;
  body: Record<string, unknown>;
}

/** Route the proxy: ``reply(body)`` returns the text to stream (or a Response-like object). */
export async function mockProxy(
  page: Page,
  reply: (body: Record<string, unknown>) => string | { status: number; json: Record<string, unknown> } | Promise<string>,
): Promise<ProxyCall[]> {
  const calls: ProxyCall[] = [];
  await page.route(`${PROXY}/**`, async (route: Route) => {
    const req = route.request();
    const headers = { "access-control-allow-origin": "*", "access-control-allow-headers": "content-type" };
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers });
    const body = JSON.parse(req.postData() ?? "{}") as Record<string, unknown>;
    calls.push({ mode: String(body.mode), body });
    const r = await reply(body);
    if (typeof r === "string") return route.fulfill({ status: 200, headers: { ...headers, "content-type": "text/event-stream" }, body: sse(r) });
    return route.fulfill({ status: r.status, headers: { ...headers, "content-type": "application/json" }, body: JSON.stringify(r.json) });
  });
  return calls;
}

export async function waitReady(page: Page): Promise<void> {
  await page.waitForFunction(() => document.documentElement.dataset.ready === "true", undefined, { timeout: 120_000 });
}

export async function ask(page: Page, text: string): Promise<void> {
  await page.fill("#composer-input", text);
  await page.press("#composer-input", "Enter");
}

/** A minimal valid PDF (Helvetica text, one line per entry, one page per list). */
export function makePdf(pages: string[][]): Buffer {
  const objects: string[] = [];
  const add = (s: string): number => objects.push(s);
  add("<< /Type /Catalog /Pages 2 0 R >>");
  add(""); // pages, filled below
  add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
  const kids: number[] = [];
  for (const lines of pages) {
    const ops = lines.map((l, i) => `BT /F1 ${i === 0 ? 14 : 11} Tf 72 ${760 - i * 22} Td (${l.replace(/[()\\]/g, (c) => `\\${c}`)}) Tj ET`).join("\n");
    const content = add(`<< /Length ${ops.length} >>\nstream\n${ops}\nendstream`);
    kids.push(add(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents ${content} 0 R >>`));
  }
  objects[1] = `<< /Type /Pages /Kids [${kids.map((k) => `${k} 0 R`).join(" ")}] /Count ${kids.length} >>`;
  let pdf = "%PDF-1.4\n";
  const offsets: number[] = [];
  objects.forEach((o, i) => {
    offsets.push(pdf.length);
    pdf += `${i + 1} 0 obj\n${o}\nendobj\n`;
  });
  const xref = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n${offsets.map((o) => `${String(o).padStart(10, "0")} 00000 n \n`).join("")}`;
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Buffer.from(pdf, "latin1");
}
