// server.js — OpenRouter Free‑Tier Proxy for Janitor.AI
// Deploy on Render.com

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 10000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// ---------- OpenRouter configuration ----------
const OPENROUTER_API_BASE =
  process.env.OPENROUTER_API_BASE || 'https://openrouter.ai/api/v1';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

// Timeouts
const SOCKET_TIMEOUT = 180000;         // 3 minutes
const REQUEST_TIMEOUT = 180000;        // 3 minutes per attempt
const GLOBAL_TIMEOUT = 900000;         // 15 minutes overall

const httpAgent = new http.Agent({ keepAlive: true, timeout: SOCKET_TIMEOUT });
const httpsAgent = new https.Agent({ keepAlive: true, timeout: SOCKET_TIMEOUT });

// ---------- Model mapping (OpenAI aliases → OpenRouter free models) ----------
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek/deepseek-v4-pro:free',
  'gpt-4': 'deepseek/deepseek-v4-pro:free',
  'gpt-4-turbo': 'deepseek/deepseek-v4-pro:free',
};

// ---------- Rate‑limit pacing (OpenRouter free: 20 req/min, 200/day) ----------
const MIN_REQUEST_INTERVAL_MS = 3000;   // 3 seconds between starts
let lastRequestTime = 0;
let processingQueue = Promise.resolve();

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function backoff(attempt) {
  const base = Math.min(5000 * Math.pow(2, attempt - 1), 60000);
  const jitter = Math.random() * 2000;
  return Math.floor(base + jitter);
}

// ---------- Routes ----------
app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(model => ({
      id: model,
      object: 'model',
      created: Date.now(),
      owned_by: 'openrouter-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  req.setTimeout(0);
  res.setTimeout(0);

  let responseClosed = false;
  let currentController = null;

  res.on('close', () => {
    if (!res.writableEnded) {
      responseClosed = true;
      if (currentController) {
        currentController.abort();
      }
      console.log('Client disconnected → aborting OpenRouter request');
    }
  });

  // Serial queue + pacing
  processingQueue = processingQueue.then(async () => {
    const now = Date.now();
    const wait = Math.max(0, lastRequestTime + MIN_REQUEST_INTERVAL_MS - now);
    if (wait > 0) {
      console.log(`Pacing: waiting ${wait}ms`);
      await sleep(wait);
    }
    lastRequestTime = Date.now();

    try {
      const { model, messages, temperature, max_tokens, stream } = req.body;
      const orModel = MODEL_MAPPING[model] || 'deepseek/deepseek-v4-pro:free';

      // ---------- Streaming path ----------
      if (stream) {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.flushHeaders();

        if (!responseClosed) res.write(': warmup\n\n');

        const heartbeat = setInterval(() => {
          if (!responseClosed) {
            res.write(': waiting-for-model\n\n');
          }
        }, 15000);

        try {
          const orStream = await fetchOpenRouterStream(orModel, messages, temperature, max_tokens, responseClosed);
          clearInterval(heartbeat);

          orStream.on('data', chunk => {
            if (!responseClosed) res.write(chunk);
          });
          orStream.on('end', () => {
            if (!responseClosed) res.end();
          });
          orStream.on('error', err => {
            console.error('Stream error:', err.message);
            if (!responseClosed) res.end();
          });
        } catch (err) {
          clearInterval(heartbeat);
          console.error('OpenRouter streaming error:', err.message);
          if (!responseClosed) {
            res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
            res.end();
          }
        }
        return;
      }

      // ---------- Non‑streaming path ----------
      const orRequest = {
        model: orModel,
        messages,
        temperature: temperature ?? 0.7,
        max_tokens: max_tokens ?? 16384,
        stream: false
      };

      const MAX_ATTEMPTS = 3;
      let response = null;
      const startTime = Date.now();

      for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        if (responseClosed) {
          console.log('Client gone → stop retries');
          break;
        }
        if (Date.now() - startTime > GLOBAL_TIMEOUT) {
          throw new Error('Global timeout reached');
        }

        try {
          currentController = new AbortController();
          console.log(`Attempt ${attempt}/${MAX_ATTEMPTS} → ${orModel}`);

          response = await axios.post(
            `${OPENROUTER_API_BASE}/chat/completions`,
            orRequest,
            {
              headers: {
                Authorization: `Bearer ${OPENROUTER_API_KEY}`,
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://janitor.ai',     // OpenRouter wants a referer
                'X-Title': 'Janitor AI Proxy'              // and a title
              },
              timeout: REQUEST_TIMEOUT,
              responseType: 'json',
              httpAgent,
              httpsAgent,
              signal: currentController.signal
            }
          );
          console.log(`Success on attempt ${attempt}`);
          break;
        } catch (error) {
          if (responseClosed) {
            console.log('Client already gone');
            break;
          }

          const status = error.response?.status;
          const code = error.code;
          let errorBody = '';
          if (error.response?.data) {
            try { errorBody = JSON.stringify(error.response.data).substring(0, 500); } catch {}
          }

          console.log(`Attempt ${attempt} failed: HTTP ${status} / code ${code}`);
          if (errorBody) console.log(`OpenRouter error body: ${errorBody}`);

          const retryable =
            code === 'ECONNABORTED' ||
            code === 'ERR_CANCELED' ||
            status === 429 ||
            (status && status >= 500);

          if (retryable && attempt < MAX_ATTEMPTS) {
            const delay = backoff(attempt);
            console.log(`Retrying in ${delay}ms`);
            await sleep(delay);
            continue;
          }

          throw new Error(`OpenRouter API error: ${status || code} – ${errorBody}`);
        }
      }

      if (!response) throw new Error('Model did not respond after retries');

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices,
        usage: response.data.usage || {}
      });

    } catch (error) {
      console.error('Proxy error:', error.message);
      if (!res.headersSent) {
        res.status(error.response?.status || 502).json({
          error: {
            message: error.message,
            type: 'proxy_error',
            code: error.response?.status || 502
          }
        });
      }
    }
  }).catch(err => console.error('Queue error:', err));

  return new Promise(() => {});
});

// ---------- Helper: fetch a streaming response from OpenRouter with retries ----------
async function fetchOpenRouterStream(model, messages, temperature, max_tokens, responseClosed) {
  const orRequest = {
    model,
    messages,
    temperature: temperature ?? 0.7,
    max_tokens: max_tokens ?? 16384,
    stream: true
  };

  const MAX_ATTEMPTS = 3;
  const startTime = Date.now();

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    if (responseClosed) throw new Error('Client disconnected');
    if (Date.now() - startTime > GLOBAL_TIMEOUT) throw new Error('Global timeout reached');

    try {
      console.log(`Stream attempt ${attempt}/${MAX_ATTEMPTS} → ${model}`);
      const response = await axios.post(
        `${OPENROUTER_API_BASE}/chat/completions`,
        orRequest,
        {
          headers: {
            Authorization: `Bearer ${OPENROUTER_API_KEY}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://janitor.ai',
            'X-Title': 'Janitor AI Proxy'
          },
          timeout: REQUEST_TIMEOUT,
          responseType: 'stream',
          httpAgent,
          httpsAgent
        }
      );
      console.log(`Stream success on attempt ${attempt}`);
      return response.data;
    } catch (error) {
      const status = error.response?.status;
      const code = error.code;
      console.log(`Stream attempt ${attempt} failed: HTTP ${status} / code ${code}`);

      if (
        code === 'ECONNABORTED' ||
        code === 'ERR_CANCELED' ||
        status === 429 ||
        (status && status >= 500)
      ) {
        if (attempt < MAX_ATTEMPTS) {
          const delay = backoff(attempt);
          console.log(`Backing off for ${delay}ms`);
          await sleep(delay);
          continue;
        }
      }
      throw new Error(`OpenRouter stream error: ${status || code}`);
    }
  }
  throw new Error('Stream did not respond after retries');
}

// ---------- 404 fallback ----------
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`Proxy running on port ${PORT}`);
});
