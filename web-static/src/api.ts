// Client of the LLM proxy (deploy/worker): POST {url}/chat with a structured request; the reply
// is the provider's streamed chat completion (Server-Sent Events, OpenAI chunk format), passed
// through unchanged. The proxy builds the prompts, so only the user's messages, the passages
// selected for the question and (for checks) the answer are sent.

export interface ProxyPassage {
  label: string;
  heading: string;
  sections: string[];
  text: string;
}

export type ProxyRequest =
  | {
      mode: "chat";
      messages: { role: "user" | "assistant"; content: string }[];
      passages?: ProxyPassage[];
      summary?: string;
      collection?: { label: string; notice: string };
    }
  | { mode: "summary"; previous_summary?: string; transcript: string }
  | {
      mode: "synth";
      profile: string;
      question: string;
      passages: ProxyPassage[];
      plain?: boolean;
      revision?: { previous: string; problems: string[]; new_labels: string[] };
    }
  | { mode: "critic"; profile: string; question: string; passages: ProxyPassage[]; answer: string };

export class ProxyError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status: number,
    readonly retryAfter: number | null = null,
  ) {
    super(message);
  }
}

export interface ProxyResult {
  text: string;
  usage: Record<string, number> | null;
  finishReason: string | null;
}

const FRIENDLY: Record<number, string> = {
  429: "The assistant is busy right now. Please try again in a minute.",
  503: "The model could not be reached. Please try again in a moment.",
};

/** Parse SSE ``data:`` lines; returns the unconsumed tail. */
export function parseSse(buffer: string, onData: (data: string) => void): string {
  let rest = buffer;
  let idx: number;
  while ((idx = rest.indexOf("\n")) >= 0) {
    const line = rest.slice(0, idx).replace(/\r$/, "");
    rest = rest.slice(idx + 1);
    if (line.startsWith("data:")) onData(line.slice(5).trimStart());
  }
  return rest;
}

export async function callProxy(
  url: string,
  body: ProxyRequest,
  opts: { signal?: AbortSignal; onToken?: (t: string) => void } = {},
): Promise<ProxyResult> {
  if (!url) throw new ProxyError("not_configured", "The LLM proxy is not set up for this site yet (see README: Public demo).", 0);
  let res: Response;
  try {
    res = await fetch(`${url}/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: opts.signal ?? null,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ProxyError("network", "The assistant could not be reached. Check the connection and try again.", 0);
  }
  if (!res.ok) {
    let message = FRIENDLY[res.status] ?? `The assistant answered with an error (HTTP ${res.status}).`;
    let code = "error";
    let retry: number | null = null;
    try {
      const j = (await res.json()) as { error?: string; message?: string; retry_after?: number };
      if (j.message) message = j.message;
      if (j.error) code = j.error;
      if (typeof j.retry_after === "number") retry = j.retry_after;
    } catch {
      /* not JSON */
    }
    throw new ProxyError(code, message, res.status, retry);
  }
  if (!res.body) throw new ProxyError("error", "The reply was empty.", res.status);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let text = "";
  let usage: Record<string, number> | null = null;
  let finishReason: string | null = null;
  let failure: ProxyError | null = null;
  const onData = (data: string): void => {
    if (!data || data === "[DONE]") return;
    let chunk: {
      choices?: { delta?: { content?: string | null }; finish_reason?: string | null }[];
      usage?: Record<string, number>;
      x_groq?: { usage?: Record<string, number> };
      error?: { message?: string } | string;
    };
    try {
      chunk = JSON.parse(data) as typeof chunk;
    } catch {
      return;
    }
    if (chunk.error) {
      failure = new ProxyError("upstream", "The model stopped with an error. Please try again.", 502);
      return;
    }
    for (const choice of chunk.choices ?? []) {
      const t = choice.delta?.content;
      if (t) {
        text += t;
        opts.onToken?.(t);
      }
      if (choice.finish_reason) finishReason = choice.finish_reason;
    }
    usage = chunk.usage ?? chunk.x_groq?.usage ?? usage;
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf = parseSse(buf + decoder.decode(value, { stream: true }), onData);
    if (failure) {
      await reader.cancel().catch(() => undefined);
      throw failure;
    }
  }
  parseSse(`${buf}\n`, onData);
  if (failure) throw failure;
  return { text, usage, finishReason };
}
