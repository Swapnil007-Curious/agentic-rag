// A local stand-in for the Cloudflare Worker, used only by the screenshot run: it streams Server-
// Sent Events with real pauses (so a mid-answer state can be photographed) and answers the
// synthesizer, critic and rate-limit cases. Nothing here contacts the real proxy.

import { createServer, type Server } from "node:http";

export type MockMode = "answer" | "abstain" | "rate" | "stall";

export interface MockWorker {
  server: Server;
  url: string;
  setMode(mode: MockMode): void;
  close(): Promise<void>;
}

const ANSWER =
  "Rebasing takes the commits of your branch and replays them on top of another branch, so the history reads as a straight line [C1]. It rewrites commits, which is why you should never rebase commits that exist outside your repository [C2].";

function sseChunk(text: string): string {
  return `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: text }, finish_reason: null }] })}\n\n`;
}

export function startMockWorker(port: number): Promise<MockWorker> {
  let mode: MockMode = "answer";
  const server = createServer((req, res) => {
    const cors = { "access-control-allow-origin": req.headers.origin ?? "*", "access-control-allow-headers": "content-type", "access-control-allow-methods": "POST, OPTIONS" };
    if (req.method === "OPTIONS") {
      res.writeHead(204, cors);
      res.end();
      return;
    }
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString()));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as { mode?: string };
      if (mode === "rate") {
        res.writeHead(429, { ...cors, "content-type": "application/json" });
        res.end(JSON.stringify({ error: "rate_limited", message: "Too many.", retry_after: 342 }));
        return;
      }
      res.writeHead(200, { ...cors, "content-type": "text/event-stream", "cache-control": "no-store" });
      const finish = (): void => {
        res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
      };
      if (body.mode === "critic") {
        res.write(sseChunk(JSON.stringify({ verdict: "accept", unsupported_sentences: [], feedback: "" })));
        finish();
        return;
      }
      if (body.mode === "synth") {
        res.write(
          sseChunk(
            mode === "abstain"
              ? JSON.stringify({ abstain: true, answer: "", citations: [] })
              : JSON.stringify({ abstain: false, answer: "A complaint may be filed by the consumer to whom the goods were sold [C1].", citations: ["C1"] }),
          ),
        );
        finish();
        return;
      }
      const words = ANSWER.split(" ");
      let i = 0;
      const step = (): void => {
        if (i >= words.length) return finish();
        const n = Math.min(words.length - i, 2 + Math.floor(Math.random() * 3));
        res.write(sseChunk(`${words.slice(i, i + n).join(" ")} `));
        i += n;
        // in "stall" mode the stream pauses after the first few words so the caret is visible
        setTimeout(step, mode === "stall" && i >= 10 && i < 14 ? 6000 : 60);
      };
      step();
    });
  });
  return new Promise((resolve) => {
    server.listen(port, "127.0.0.1", () => {
      resolve({
        server,
        url: `http://localhost:${port}`,
        setMode: (m) => {
          mode = m;
        },
        close: () => new Promise((r) => server.close(() => r())),
      });
    });
  });
}
