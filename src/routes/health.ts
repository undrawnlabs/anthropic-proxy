import type { FastifyInstance } from "fastify"
import { metrics } from "../config/metrics"

export default async function healthRoutes(app: FastifyInstance) {
  app.get("/health", async () => ({ ok: true }))
  app.get("/version", async () => ({ version: "core-fast-1.1.0" }))
  app.get("/metrics", async () => metrics)
}
