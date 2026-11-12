// Collection data: manifest.json, then each collection's passages.json and vectors.bin only when it
// is selected. Downloads report progress, are checked against the manifest's SHA-256 and are kept
// in the Cache API under the manifest version, so later visits load them without the network.

import type { Chunk, CollectionMeta, FileInfo, Manifest } from "./types.ts";
import { Int8Index, SearchSource } from "./vectors.ts";

const CACHE_PREFIX = "agentic-rag-data-";

export type Progress = (loaded: number, total: number) => void;

export async function loadManifest(base: string): Promise<Manifest> {
  const res = await fetch(`${base}data/manifest.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`manifest.json: HTTP ${res.status}`);
  return (await res.json()) as Manifest;
}

async function openCache(version: string): Promise<Cache | null> {
  try {
    if (typeof caches === "undefined") return null;
    const name = CACHE_PREFIX + version;
    for (const key of await caches.keys()) {
      if (key.startsWith(CACHE_PREFIX) && key !== name) await caches.delete(key);
    }
    return await caches.open(name);
  } catch {
    return null; // private windows and some browsers refuse storage: download every time
  }
}

async function sha256Hex(buf: ArrayBuffer): Promise<string | null> {
  try {
    const digest = await crypto.subtle.digest("SHA-256", buf);
    return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
  } catch {
    return null;
  }
}

/** Download with byte progress (totals from the manifest: the response may be compressed). */
async function download(url: string, total: number, onProgress: Progress): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.byteLength;
    onProgress(Math.min(loaded, total), total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.byteLength;
  }
  return out.buffer;
}

export async function fetchFile(base: string, version: string, file: FileInfo, onProgress: Progress): Promise<ArrayBuffer> {
  const url = `${base}data/${file.path}`;
  const cache = await openCache(version);
  const hit = cache ? await cache.match(url).catch(() => undefined) : undefined;
  if (hit) {
    const buf = await hit.arrayBuffer();
    onProgress(file.bytes, file.bytes);
    return buf;
  }
  const buf = await download(url, file.bytes, onProgress);
  const digest = await sha256Hex(buf);
  if (digest !== null && digest !== file.sha256) throw new Error(`${file.path}: the download is damaged (checksum)`);
  if (cache) await cache.put(url, new Response(buf.slice(0))).catch(() => undefined);
  return buf;
}

interface PassageFile {
  fields: string[];
  rows: [string, string, string, string[], string, number, string][];
}

export function parsePassages(text: string): Chunk[] {
  const data = JSON.parse(text) as PassageFile;
  return data.rows.map(([chunk_id, doc_id, heading_path, section_ids, pages, n_tokens, body]) => ({
    chunk_id,
    doc_id,
    heading_path,
    section_ids,
    pages,
    n_tokens,
    text: body,
  }));
}

export class CollectionStore {
  private readonly loaded = new Map<string, Promise<SearchSource>>();

  constructor(
    private readonly base: string,
    readonly manifest: Manifest,
  ) {}

  meta(id: string): CollectionMeta | undefined {
    return this.manifest.collections.find((c) => c.id === id);
  }

  isLoaded(id: string): boolean {
    return this.loaded.has(id);
  }

  load(id: string, onProgress: Progress = () => {}): Promise<SearchSource> {
    const meta = this.meta(id);
    if (!meta) return Promise.reject(new Error(`unknown collection ${id}`));
    let p = this.loaded.get(id);
    if (!p) {
      p = this.fetchCollection(meta, onProgress);
      p.catch(() => this.loaded.delete(id));
      this.loaded.set(id, p);
    }
    return p;
  }

  private async fetchCollection(meta: CollectionMeta, onProgress: Progress): Promise<SearchSource> {
    const { passages, vectors } = meta.files;
    const total = passages.bytes + vectors.bytes;
    let a = 0;
    let b = 0;
    const report = (): void => onProgress(a + b, total);
    const [pbuf, vbuf] = await Promise.all([
      fetchFile(this.base, this.manifest.version, passages, (l) => {
        a = l;
        report();
      }),
      fetchFile(this.base, this.manifest.version, vectors, (l) => {
        b = l;
        report();
      }),
    ]);
    const chunks = parsePassages(new TextDecoder().decode(pbuf));
    return new SearchSource(meta.id, chunks, new Int8Index(vbuf), meta.threshold);
  }
}
