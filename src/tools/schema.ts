export const OpenAITools = [
  {
    type: "function",
    function: {
      name: "web_search",
      description: "Search the live web for fresh information",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string" },
          maxResults: { type: "integer", minimum: 1, maximum: 10 },
        },
        required: ["query"],
      },
    },
  },
];

export const AnthropicTools = [
  {
    name: "web_search",
    description: "Search the live web for fresh information",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string" },
        maxResults: { type: "integer", minimum: 1, maximum: 10 },
      },
      required: ["query"],
    },
  },
] as const;
