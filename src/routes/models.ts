import type { FastifyInstance } from "fastify"
import type { Env } from "../config/env"

export default async function modelsRoutes(app: FastifyInstance, env: Env) {
  app.get("/v1/models", async (_req, reply) => {
  return reply.send({
    object: "list",
    data: [
      { id: "hanna-core", object: "model", owned_by: "alias", description: `alias → ${env.ANTHROPIC_MODEL}` },
      { id: env.ANTHROPIC_MODEL, object: "model", owned_by: "anthropic" }
    ]
  });
});
}
