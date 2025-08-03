import Fastify from 'fastify'
import fetch from 'node-fetch'

const fastify = Fastify({ logger: true })

fastify.post('/v1/complete', async (request, reply) => {
  const { model, prompt, max_tokens } = request.body

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': process.env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model,
      max_tokens,
      messages: [
        {
          role: 'user',
          content: prompt
        }
      ]
    })
  })

  const data = await response.json()
  reply.send(data)
})

const start = async () => {
  try {
    await fastify.listen({ port: process.env.PORT || 3000, host: '0.0.0.0' })
  } catch (err) {
    process.exit(1)
  }
}

start()
