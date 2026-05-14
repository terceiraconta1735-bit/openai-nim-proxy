const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// ----- Nvidia NIM configuration -----
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// ----- Timeouts (generous for large models) -----
const SOCKET_TIMEOUT = 180000;         // 3 minutes (socket keep‑alive)
const REQUEST_TIMEOUT = 180000;        // 3 minutes per HTTP attempt
const GLOBAL_TIMEOUT = 900000;         // 15 minutes (all attempts + backoff)

const httpAgent = new http.Agent({ keepAlive: true, timeout: SOCKET_TIMEOUT });
const httpsAgent = new https.Agent({ keepAlive: true, timeout: SOCKET_TIMEOUT });

// ----- Model mapping (OpenAI → Nvidia) -----
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v4-pro'
};

// ----- Rate‑limit guard: serial queue + minimum interval -----
const MIN_REQUEST_INTERVAL_MS = 2000;   // 2 seconds between requests (safe for 40 RPM)
let lastRequestTime = 0;
let processingQueue = Promise.resolve();

// ----- Helpers -----
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function backoff(attempt) {
  const base = Math.min(5000 * Math.pow(2, attempt - 1), 60000);
  const jitter = Math.random() * 2000;
  return Math.floor(base + jitter);
}

/** Safely extract an error body from Nvidia, even if it’s a stream or circular */
function safeErrorBody(data) {
  if (!data) return '';
  if (typeof data === 'string') {
    const snippet = data.substring(0, 500);
    // Try to extract 'detail' for nicer logging
    try {
      const parsed = JSON.parse(data);
      if (parsed?.detail) return JSON.stringify(parsed.detail).substring(0, 500);
    } catch {}
    return snippet;
  }
  if (typeof data === 'object') {
    try {
      return JSON.stringify(data).substring(0, 500);
    } catch {
      return `[Unstringifiable ${typeof data}]`;
    }
  }
  return String(data).substring(0, 500);
}

// ----- Routes -----
app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(model => ({
      id: model,
      object: 'model',
      created: Date.now(),
      owned_by: 'nvidia-nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  // Disable Express timeouts – we manage them ourselves
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
      console.log('Client disconnected → aborting Nvidia request');
    }
  });

  // ----- Serial queue + minimum pacing -----
  processingQueue = processingQueue.then(async () => {
    const now = Date.now();
    const wait = Math.max(0, lastRequestTime + MIN_REQUEST_INTERVAL_MS - now);
    if (wait > 0) {
      console.log(`Cool-down: waiting ${wait}ms before next request`);
      await sleep(wait);
    }
    lastRequestTime = Date.now();

    try {
      const { model, messages, temperature, max_tokens, stream } = req.body;
      const nimModel = MODEL_MAPPING[model] || 'deepseek-ai/deepseek-v4-pro';

      // ───────────────────────────────────────────────────
      //  STREAMING PATH – keep the client alive from the start
      // ───────────────────────────────────────────────────
      if (stream) {
        // Open SSE immediately and start a heartbeat
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.flushHeaders();

        if (!responseClosed) res.write(': warmup\n\n');

        const heartbeat = setInterval(() => {
          if (!responseClosed) {
            res.write(': waiting-for-model\n\n');
          }
        }, 15000);   // every 15 seconds, keeps the connection alive

        try {
          const nvidiaStream = await fetchNvidiaStream(
            nimModel, messages, temperature, max_tokens, responseClosed
          );
          // Stop the heartbeat – real data is coming
          clearInterval(heartbeat);
          // Pipe the real stream
          nvidiaStream.on('data', chunk => {
            if (!responseClosed) res.write(chunk);
          });
          nvidiaStream.on('end', () => {
            if (!responseClosed) res.end();
          });
          nvidiaStream.on('error', err => {
            console.error('Stream error:', err.message);
            if (!responseClosed) res.end();
          });
        } catch (err) {
          clearInterval(heartbeat);
          console.error('Nvidia streaming error:', err.message);
          if (!responseClosed) {
            // Send an error event inside the SSE stream
            res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
            res.end();
          }
        }
        return;   // streaming handled, exit the serial queue function
      }

      // ───────────────────────────────────────────────────
      //  NON‑STREAMING PATH
      // ───────────────────────────────────────────────────
      const nimRequest = {
        model: nimModel,
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
          console.log(`Attempt ${attempt}/${MAX_ATTEMPTS} → ${nimModel}`);

          response = await axios.post(
            `${NIM_API_BASE}/chat/completions`,
            nimRequest,
            {
              headers: {
                Authorization: `Bearer ${NIM_API_KEY}`,
                'Content-Type': 'application/json'
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
          const nvidiaErrorBody = safeErrorBody(error.response?.data);
          let retryAfterSec = null;

          const retryAfterHeader = error.response?.headers?.['retry-after'];
          if (retryAfterHeader) {
            retryAfterSec = parseInt(retryAfterHeader, 10);
            if (isNaN(retryAfterSec)) retryAfterSec = null;
          }

          console.log(`Attempt ${attempt} failed: HTTP ${status} / code ${code}`);
          if (nvidiaErrorBody) console.log(`Nvidia error body: ${nvidiaErrorBody}`);

          // 429 (rate limit) – retry with Retry-After or backoff
          if (status === 429) {
            if (attempt === MAX_ATTEMPTS) break;
            const delay = (retryAfterSec && retryAfterSec > 0 && retryAfterSec <= 120)
              ? retryAfterSec * 1000
              : backoff(attempt);
            console.log(`Rate limited – waiting ${delay}ms`);
            await sleep(delay);
            continue;
          }

          // Network / server errors – retry
          if (
            code === 'ECONNABORTED' ||
            code === 'ERR_CANCELED' ||
            status === 502 || status === 503 || status === 504
          ) {
            if (attempt === MAX_ATTEMPTS) break;
            const delay = backoff(attempt);
            console.log(`Network/server error, backing off for ${delay}ms`);
            await sleep(delay);
            continue;
          }

          // Anything else (4xx except 429) → hard error
          throw new Error(`Nvidia API error: ${status || code} – ${nvidiaErrorBody || error.message}`);
        }
      }

      if (!response) throw new Error('Model did not respond after retries');

      // Non‑streaming success
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
  }).catch(err => {
    console.error('Queue processing error:', err);
  });

  // Express requires we don’t leave the handler hanging; we always respond inside the chain.
  return new Promise(() => {});
});

// ─────────────────────────────────────────────────────────
//  Helper: fetch a STREAMING response from Nvidia with retries
// ─────────────────────────────────────────────────────────
async function fetchNvidiaStream(model, messages, temperature, max_tokens, responseClosed) {
  const nimRequest = {
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

    if (Date.now() - startTime > GLOBAL_TIMEOUT) {
      throw new Error('Global timeout reached');
    }

    try {
      console.log(`Stream attempt ${attempt}/${MAX_ATTEMPTS} → ${model}`);
      const response = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        nimRequest,
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          timeout: REQUEST_TIMEOUT,
          responseType: 'stream',
          httpAgent,
          httpsAgent
        }
      );
      console.log(`Stream success on attempt ${attempt}`);
      return response.data;   // readable stream
    } catch (error) {
      const status = error.response?.status;
      const code = error.code;
      const nvidiaErrorBody = safeErrorBody(error.response?.data);
      console.log(`Stream attempt ${attempt} failed: HTTP ${status} / code ${code}`);
      if (nvidiaErrorBody) console.log(`Nvidia stream error body: ${nvidiaErrorBody}`);

      // Retry only on temporary failures
      if (
        code === 'ECONNABORTED' ||
        code === 'ERR_CANCELED' ||
        status === 429 ||
        status === 502 || status === 503 || status === 504
      ) {
        if (attempt === MAX_ATTEMPTS) break;
        const delay = backoff(attempt);
        console.log(`Backing off for ${delay}ms`);
        await sleep(delay);
        continue;
      }
      // Non‑retryable error → throw immediately
      throw new Error(`Nvidia stream error: ${status || code} – ${nvidiaErrorBody}`);
    }
  }
  throw new Error('Stream did not respond after retries');
}

// ─────────────────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`Proxy running on port ${PORT}`);
});
