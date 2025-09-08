import Fastify from "fastify"
import cors from "@fastify/cors"
import { loadEnv } from "./config/env"
import healthRoutes from "./routes/health"
import modelsRoutes from "./routes/models"
import completeRoutes from "./routes/complete"
import chatCompletionsRoutes from "./routes/chatCompletions"

export function createServer() {
  const env = loadEnv()
  const app = Fastify({ logger: true, bodyLimit: env.BODY_LIMIT_BYTES })
  app.register(cors, { origin: true })

  app.addHook("onRequest", (req, _res, done) => { ;(req as any).id ||= crypto.randomUUID(); done() })

  app.register(healthRoutes)
  app.register(async (f)=> modelsRoutes(f, env))
  app.register(completeRoutes)
  app.register(chatCompletionsRoutes)

  return { app, env }
}
