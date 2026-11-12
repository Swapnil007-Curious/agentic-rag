// Build-time settings of the static site: web-static/site.config.json, each value overridable by an
// environment variable. vite.config.ts compiles them into the bundle (define) and into the
// Content-Security-Policy meta tag. Nothing reads site.config.json at run time and nothing puts it
// in the Cache API: a browser can only hold an older build, whose entry file name carries a
// content hash, so a new URL always arrives with a new index.html and bundle.

export interface SiteFile {
  workerUrl?: unknown;
  modelDtype?: unknown;
}

export interface Setting {
  value: string;
  source: "environment" | "site.config.json" | "unset";
}

/**
 * The override when it is non-blank, else the file's value, else "". A blank override never wins:
 * GitHub Actions expands an unset repository variable (`${{ vars.NAME }}`) to an empty string.
 */
export function pick(override: string | undefined, fileValue: unknown): Setting {
  const env = override?.trim() ?? "";
  if (env) return { value: env, source: "environment" };
  const file = typeof fileValue === "string" ? fileValue.trim() : "";
  if (file) return { value: file, source: "site.config.json" };
  return { value: "", source: "unset" };
}

/** The LLM proxy's URL (VITE_WORKER_URL, else site.config.json) without trailing slashes. */
export function resolveWorkerUrl(override: string | undefined, fileValue: unknown): Setting {
  const s = pick(override, fileValue);
  const value = s.value.replace(/\/+$/, "");
  if (value) workerOrigin(value); // rejects a malformed or non-https URL
  return { value, source: value ? s.source : "unset" };
}

/** The origin the Content-Security-Policy must allow for the proxy ("" when there is none). */
export function workerOrigin(url: string): string {
  if (!url) return "";
  const u = new URL(url);
  if (u.protocol !== "https:" && u.hostname !== "localhost" && u.hostname !== "127.0.0.1") {
    throw new Error(`the worker URL must use https: ${url}`);
  }
  return u.origin;
}

/** One line for the build log. */
export function describeWorkerUrl(s: Setting): string {
  if (!s.value) return "worker URL: (empty) - sending a message will report that the LLM proxy is not set up";
  return `worker URL: ${s.value} (from ${s.source === "environment" ? "VITE_WORKER_URL" : "site.config.json"})`;
}

/** Fails the build when no URL was configured (the Pages workflow sets REQUIRE_WORKER_URL=true). */
export function assertWorkerUrl(s: Setting): void {
  if (s.value) return;
  throw new Error(
    'The LLM proxy URL is empty: set "workerUrl" in web-static/site.config.json or the repository ' +
      "variable WORKER_URL (Settings -> Secrets and variables -> Actions -> Variables). " +
      "REQUIRE_WORKER_URL=true makes an empty URL fail the build instead of deploying a site that cannot answer.",
  );
}
