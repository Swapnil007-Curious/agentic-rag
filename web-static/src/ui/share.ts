// Share links: ?q=<question>&c=<collection> pre-fills the composer and selects the collection on
// load. They never send the question.

export interface ShareParams {
  q: string;
  c: string | null;
}

export function parseShareParams(search: string): ShareParams | null {
  const params = new URLSearchParams(search);
  const q = (params.get("q") ?? "").trim().slice(0, 4000);
  const c = (params.get("c") ?? "").trim();
  if (!q && !c) return null;
  return { q, c: /^[a-z0-9_]{1,40}$/.test(c) ? c : null };
}

export function buildShareUrl(origin: string, base: string, question: string, collection: string | null): string {
  const url = new URL(base, origin);
  url.searchParams.set("q", question.trim().slice(0, 4000));
  if (collection) url.searchParams.set("c", collection);
  return url.toString();
}
