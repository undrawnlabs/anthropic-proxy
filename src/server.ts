import { createServer } from "./app"

const { app, env } = createServer()

app.listen({ port: env.PORT, host: "0.0.0.0" }, (err, address) => {
  if (err) { app.log.error(err); process.exit(1) }
  app.log.info(`undrawn Core (FAST) at ${address}`)
})
