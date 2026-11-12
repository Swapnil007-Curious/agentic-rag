// Uploaded PDFs, entirely in the browser: pdf.js (self-hosted, loaded on first use) extracts the
// text lines, chunker.ts cuts them at numbered headings, the embedding worker embeds the chunks
// with progress, and the result is searched next to the selected collection. Nothing is sent
// anywhere; only passages that a question selects go to the LLM proxy with that question.
// Documents live in memory and disappear with the page.

import { chunkLines, type Line } from "./chunker.ts";
import type { Embedder } from "./embedder.ts";
import type { Chunk } from "./types.ts";
import { Float32Index, SearchSource } from "./vectors.ts";

export class UploadError extends Error {}

export interface UploadLimits {
  max_files: number;
  max_mb: number;
  max_pages: number;
  max_chunks_per_file: number;
}

export interface UploadedDoc {
  docId: string;
  name: string;
  pages: number;
  source: SearchSource;
}

export type UploadProgress = (stage: "reading" | "embedding", done: number, total: number) => void;

/** A display name without path or control characters. */
export function cleanName(raw: string): string {
  const base = raw.replace(/\\/g, "/").split("/").pop() ?? "";
  const name = base.replace(/[\x00-\x1f\x7f<>\\/:"|?*]+/g, " ").trim() || "document.pdf";
  return name.slice(0, 120);
}

export function checkFile(name: string, bytes: Uint8Array, limits: UploadLimits): void {
  if (bytes.byteLength > limits.max_mb * 1024 * 1024) throw new UploadError(`The file is larger than ${limits.max_mb} MB.`);
  const head = new TextDecoder("latin1").decode(bytes.subarray(0, 1024));
  if (!head.includes("%PDF-")) throw new UploadError("Only PDF files can be uploaded.");
  void name;
}

interface TextItemLike {
  str: string;
  transform: number[];
  hasEOL?: boolean;
}

function randomId(): string {
  const a = new Uint8Array(5);
  crypto.getRandomValues(a);
  return `up_${Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("")}`;
}

/** Lines of text with their page numbers (items on the same baseline are joined). */
async function extractLines(bytes: Uint8Array, base: string, limits: UploadLimits, onProgress: UploadProgress): Promise<{ lines: Line[]; pages: number }> {
  const pdfjs = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = `${base}pdfjs/pdf.worker.min.mjs`;
  // disableFontFace: no font programs are compiled; the text layer is all that is read
  const task = pdfjs.getDocument({ data: bytes, disableFontFace: true, useWorkerFetch: false });
  let doc: Awaited<typeof task.promise>;
  try {
    doc = await task.promise;
  } catch (err) {
    const name = (err as { name?: string }).name ?? "";
    if (name === "PasswordException") throw new UploadError("The PDF is encrypted or password-protected. Please upload an unprotected copy.");
    throw new UploadError("The file could not be read as a PDF.");
  }
  try {
    const meta = await doc.getMetadata().catch(() => null);
    const info = (meta?.info ?? {}) as Record<string, unknown>;
    if (info.EncryptFilterName) throw new UploadError("The PDF is encrypted or password-protected. Please upload an unprotected copy.");
    if (doc.numPages > limits.max_pages) throw new UploadError(`The PDF has ${doc.numPages} pages; at most ${limits.max_pages} are allowed.`);
    const lines: Line[] = [];
    for (let p = 1; p <= doc.numPages; p++) {
      const page = await doc.getPage(p);
      const content = await page.getTextContent();
      let cur = "";
      let y: number | null = null;
      for (const raw of content.items) {
        const item = raw as TextItemLike;
        if (typeof item.str !== "string") continue;
        const iy = Math.round(item.transform[5] ?? 0);
        if (y !== null && Math.abs(iy - y) > 2 && cur.trim()) {
          lines.push({ text: cur.replace(/\s+/g, " ").trim(), page: p });
          cur = "";
        }
        y = iy;
        cur += item.str;
        if (item.hasEOL) {
          if (cur.trim()) lines.push({ text: cur.replace(/\s+/g, " ").trim(), page: p });
          cur = "";
          y = null;
        }
      }
      if (cur.trim()) lines.push({ text: cur.replace(/\s+/g, " ").trim(), page: p });
      page.cleanup();
      if (p % 5 === 0 || p === doc.numPages) onProgress("reading", p, doc.numPages);
    }
    return { lines, pages: doc.numPages };
  } finally {
    await task.destroy();
  }
}

export class UploadStore {
  readonly docs: UploadedDoc[] = [];

  constructor(
    private readonly base: string,
    private readonly limits: UploadLimits,
    private readonly threshold: number,
  ) {}

  titles(): Map<string, string> {
    return new Map(this.docs.map((d) => [d.docId, d.name]));
  }

  sources(): SearchSource[] {
    return this.docs.map((d) => d.source);
  }

  find(chunkId: string): Chunk | undefined {
    for (const d of this.docs) {
      const c = d.source.get(chunkId);
      if (c) return c;
    }
    return undefined;
  }

  remove(docId: string): void {
    const i = this.docs.findIndex((d) => d.docId === docId);
    if (i >= 0) this.docs.splice(i, 1);
  }

  async add(file: { name: string; bytes: Uint8Array }, embedder: Embedder, onProgress: UploadProgress): Promise<UploadedDoc> {
    if (this.docs.length >= this.limits.max_files) {
      throw new UploadError(`At most ${this.limits.max_files} PDFs at a time. Remove one to add another.`);
    }
    const name = cleanName(file.name);
    checkFile(name, file.bytes, this.limits);
    const { lines, pages } = await extractLines(file.bytes, this.base, this.limits, onProgress);
    const text = lines.reduce((s, l) => s + l.text.length, 0);
    // the Python ingest rule for a text layer: 20 characters per page, or 200 in total
    if (text < 20 * pages && text < 200) throw new UploadError("No text was found in this PDF (scanned pages are not supported; OCR is off).");
    const docId = randomId();
    const title = name.replace(/\.pdf$/i, "") || "Uploaded document";
    const chunks = chunkLines(lines, title, docId);
    if (!chunks.length) throw new UploadError("No text could be extracted from this PDF.");
    if (chunks.length > this.limits.max_chunks_per_file) throw new UploadError("The PDF has too much text for this demo. Please upload a shorter one.");
    const vectors = await embedder.passages(
      chunks.map((c) => c.embedText),
      (done, total) => onProgress("embedding", done, total),
    );
    const plain: Chunk[] = chunks.map(({ embedText: _e, ...c }) => c);
    const doc: UploadedDoc = { docId, name, pages, source: new SearchSource("upload", plain, new Float32Index(vectors), this.threshold) };
    this.docs.push(doc);
    return doc;
  }
}
