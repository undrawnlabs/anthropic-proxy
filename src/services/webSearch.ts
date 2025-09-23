// src/services/webSearch.ts
import { TavilyClient } from "tavily";

export type WebSearchOpts = {
  query: string;
  max_results?: number;         // default 5
  include_domains?: string[];   // optional allowlist
  exclude_domains?: string[];   // optional blocklist
  search_depth?: "basic" | "advanced"; // tavily supports this
};

export type WebSearchResult = {
  url: string;
  title: string;
  content: string;
};

export async function webSearch(apiKey: string, opts: WebSearchOpts): Promise<WebSearchResult[]> {
  const client = new TavilyClient({ apiKey });

  const res = await client.search({
    query: opts.query,
    max_results: opts.max_results ?? 5,
    include_domains: opts.include_domains,
    exclude_domains: opts.exclude_domains,
    search_depth: opts.search_depth ?? "basic",
    // You can toggle these as needed:
    // include_images: false,
    // include_answer: false,
  });

  // Normalize to a compact shape
  const items = (res?.results ?? []).map((r: any) => ({
    url: r.url,
    title: r.title,
    content: r.content || r.snippet || "",
  }));

  return items;
}

/**
 * Turn results into a compact, model-friendly summary.
 * Keeps it short to avoid blowing your token budget.
 */
export function formatWebContext(results: WebSearchResult[], query: string, limit = 5): string {
  const directives = [
    "You have web search context below.",
    "Never say you can’t browse; synthesize from the provided context.",
    "Cite succinctly from the context.",
  ].join("\n");

  const slice = results.slice(0, limit);
  if (!slice.length) {
    return [directives, `Query: ${query}`, "No results found."].join("\n\n");
  }

  const refs = slice
    .map((r, i) => {
      const snippet = (r.content || "").replace(/\s+/g, " ").trim().slice(0, 500);
      return `[${i + 1}] ${r.title} — ${r.url}\n   ${snippet}`;
    })
    .join("\n");

  return [directives, `Query: ${query}`, refs].join("\n\n");
}
