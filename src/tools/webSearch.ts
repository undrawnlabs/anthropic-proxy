import { TavilyClient } from "tavily";

export type WebSearchArgs = {
  query: string;
  maxResults?: number;
};

export async function webSearch({ query, maxResults = 5 }: WebSearchArgs) {
  if (process.env.WEB_SEARCH_ENABLED !== "true") {
    throw new Error("Web search disabled by config");
  }

  const tavily = new TavilyClient({ apiKey: process.env.TAVILY_API_KEY! });
  const res = await tavily.search({
    query,
    max_results: Math.min(maxResults, 10),
  });

  return res.results.map(r => ({
    title: r.title,
    url: r.url,
    snippet: r.content,
  }));
}
