import { webSearch } from "./webSearch";

export async function executeTool(name: string, args: unknown) {
  switch (name) {
    case "web_search":
      return await webSearch(args as any);
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}
