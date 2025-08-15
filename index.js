import Fastify from 'fastify'
import cors from '@fastify/cors'
import fetch from 'node-fetch'
import { Redis } from '@upstash/redis';

const redis = (process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN)
  ? new Redis({
      url: process.env.UPSTASH_REDIS_REST_URL,
      token: process.env.UPSTASH_REDIS_REST_TOKEN
    })
  : null;

// Фолбек на RAM, якщо Redis не налаштований
const RAM_STATE = new Map();
const TTL_SECONDS = 60 * 60 * 24 * 30; // 30 днів

const fastify = Fastify({ logger: true })

await fastify.register(cors, { origin: '*' })

fastify.post('/v1/complete', async (request, reply) => {
  const { model, prompt, max_tokens } = request.body

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': process.env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model,
      max_tokens,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
    }),
  })

  const data = await response.json()
  reply.send(data)
})

const start = async () => {
  try {
    await fastify.listen({ port: process.env.PORT || 3000, host: '0.0.0.0' })
  } catch (err) {
    fastify.log.error(err)
    process.exit(1)
  }
}

start()
