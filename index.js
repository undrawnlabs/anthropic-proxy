#!/usr/bin/env node
import Fastify from 'fastify'
import { TextDecoder } from 'util'

const key = process.env.OPENROUTER_API_KEY
const models = {
  reasoning: 'deepseek/deepseek-r1-zero:free',
  completion: 'google/gemma-3-27b-it:free',
}

const fastify = Fastify({
  logger: true
})

// Helper function to send SSE events and flush immediately.
const sendSSE = (reply, event, data) => {
  const sseMessage = `event: ${event}\n` +
                     `data: ${JSON.stringify(data)}\n\n`
  reply.raw.write(sseMessage)
  // Flush if the flush method is available.
  if (typeof reply.raw.flush === 'function') {
    reply.raw.flush()
  }
}

fastify.post('/v1/messages', async (request, reply) => {
  try {
    const payload = request.body

    // Helper to normalize a message's content.
    // If content is a string, return it directly.
    // If it's an array (of objects with text property), join them.
    const normalizeContent = (content) => {
      if (typeof content === 'string') {
        return content
      }
      if (Array.isArray(content)) {
        return content.map(item => item.text).join(' ')
      }
      return ''
    }

    // Build messages array for the OpenAI payload.
    // Start with system messages if provided.
    const messages = []
    if (payload.system && Array.isArray(payload.system)) {
      payload.system.forEach(sysMsg => {
        const normalized = normalizeContent(sysMsg.text || sysMsg.content)
        messages.push({
          role: 'system',
          content: normalized
        })
      })
    }
    // Then add user (or other) messages.
    if (payload.messages && Array.isArray(payload.messages)) {
      payload.messages.forEach(msg => {
        const normalized = normalizeContent(msg.content)
        messages.push({
          role: msg.role,
          content: normalized
        })
      })
    }

    console.log('REQUEST:', JSON.stringify(messages, null, 2));


    // Prepare the OpenAI payload.
    const openaiPayload = {
      model: payload.thinking ? models.reasoning : models.completion,
      messages,
      max_tokens: payload.max_tokens,
      temperature: payload.temperature !== undefined ? payload.temperature : 1,
      stream: payload.stream === true
    }

    const openaiResponse = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${key}`
      },
      body: JSON.stringify(openaiPayload)
    })

    if (!openaiResponse.ok) {
      const errorDetails = await openaiResponse.text()
      reply.code(openaiResponse.status)
      return { error: errorDetails }
    }

    // If stream is not enabled, process the complete response.
    if (!openaiPayload.stream) {
      const data = await openaiResponse.json()

      const choice = data.choices[0]
      const openaiMessage = choice.message

      // Map finish_reason to anthropic stop_reason.
      const stopReason = choice.finish_reason === 'stop' ? 'end_turn' : choice.finish_reason

      // Create a message id; if available, replace prefix, otherwise generate one.
      const messageId = data.id
        ? data.id.replace('chatcmpl', 'msg')
        : 'msg_' + Math.random().toString(36).substr(2, 24)



      const anthropicResponse = {
        content: [
          {
            text: openaiMessage.content,
            type: 'text'
          }
        ],
        id: messageId,
        model: openaiPayload.model,
        role: openaiMessage.role,
        stop_reason: stopReason,
        stop_sequence: null,
        type: 'message',
        usage: {
          input_tokens: data.usage
            ? data.usage.prompt_tokens
            : messages.reduce((acc, msg) => acc + msg.content.split(' ').length, 0),
          output_tokens: data.usage
            ? data.usage.completion_tokens
            : openaiMessage.content.split(' ').length,
        }
      }

      return anthropicResponse
    }

    // Streaming response using Server-Sent Events.
    reply.raw.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive'
    })

    // Create a unique message id.
    const messageId = 'msg_' + Math.random().toString(36).substr(2, 24)

    // Send initial SSE events.
    sendSSE(reply, 'message_start', {
      type: 'message_start',
      message: {
        id: messageId,
        type: 'message',
        role: 'assistant',
        content: [],
        model: openaiPayload.model,
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 }
      }
    })

    sendSSE(reply, 'content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: {
        type: 'text',
        text: ''
      }
    })

    sendSSE(reply, 'ping', { type: 'ping' })

    // Prepare for reading streamed data.
    let accumulatedReasoning = ''
    let accumulatedContent = ''
    const decoder = new TextDecoder('utf-8')
    const reader = openaiResponse.body.getReader()
    let done = false

    while (!done) {
      const { value, done: doneReading } = await reader.read()
      done = doneReading
      if (value) {
        const chunk = decoder.decode(value)
        // OpenAI streaming responses are typically sent as lines prefixed with "data: "
        const lines = chunk.split('\n')

        for (const line of lines) {
          const trimmed = line.trim()
          if (trimmed === '' || !trimmed.startsWith('data:')) continue
          const dataStr = trimmed.replace(/^data:\s*/, '')
          if (dataStr === '[DONE]') {
            console.log('RESPONSE:', JSON.stringify({ content: accumulatedContent, reasoning: accumulatedReasoning }, null, 2));


            // Finalize the stream with stop events.
            sendSSE(reply, 'content_block_stop', {
              type: 'content_block_stop',
              index: 0
            })
            // For demonstration, calculate output_tokens as word count.
            sendSSE(reply, 'message_delta', {
              type: 'message_delta',
              delta: {
                stop_reason: 'end_turn',
                stop_sequence: null
              },
              usage: {
                output_tokens: accumulatedContent.split(' ').length + accumulatedReasoning.split(' ').length
              }
            })
            sendSSE(reply, 'message_stop', {
              type: 'message_stop'
            })
            reply.raw.end()
            return
          }
          try {
            const parsed = JSON.parse(dataStr)
            const delta = parsed.choices[0].delta
            if (delta && delta.content) {
              accumulatedContent += delta.content
              sendSSE(reply, 'content_block_delta', {
                type: 'content_block_delta',
                index: 0,
                delta: {
                  type: 'text_delta',
                  text: delta.content
                }
              })
            } else if (delta && delta.reasoning) {
              accumulatedReasoning += delta.reasoning
              sendSSE(reply, 'content_block_delta', {
                type: 'content_block_delta',
                index: 0,
                delta: {
                  type: 'thinking_delta',
                  thinking: delta.reasoning
                }
              })
            }
          } catch (err) {
            continue
          }
        }
      }
    }

    reply.raw.end()
  } catch (err) {
    reply.code(500)
    return { error: err.message }
  }
})

const start = async () => {
  try {
    await fastify.listen({ port: process.env.PORT || 3000 })
  } catch (err) {
    process.exit(1)
  }
}

start()
