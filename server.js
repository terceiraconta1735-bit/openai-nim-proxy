const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// ----- Timeouts -----
const SOCKET_TIMEOUT = 120000;
const REQUEST_TIMEOUT = 120000;
const GLOBAL_TIMEOUT = 600000;

const httpAgent = new http.Agent({ keepAlive: true, timeout: SOCKET_TIMEOUT });
const httpsAgent = new https.Agent({ keepAlive: true, timeout: SOCKET_TIMEOUT });

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v4-pro'
};

// ----- Rate-limit controls (40 RPM – safe with 2s between requests) -----
const MIN_REQUEST_INTERVAL_MS = 2000;
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

      const nimRequest = {
        model: nimModel,
        messages,
        temperature: temperature ?? 0.7,
        max_tokens: max_tokens ?? 16384,
        stream: !!stream
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
              responseType: stream ? 'stream' : 'text',
              httpAgent,
              httpsAgent,
              signal: currentController.signal,
              transformResponse: [(data) => data]
            }
          );

          if (!stream) {
            try {
              response.data = JSON.parse(response.data);
            } catch (e) {
              console.error('Failed to parse Nvidia JSON:', response.data);
              throw new Error('Invalid JSON from Nvidia');
            }
          }

          console.log(`Success on attempt ${attempt}`);
          break;
        } catch (error) {
          if (responseClosed) {
            console.log('Client already gone');
            break;
          }

          const status = error.response?.status;
          const code = error.code;
          let nvidiaErrorBody = '';
          let retryAfterSec = null;

          // ── EXTRACT ERROR BODY SAFELY (FIXED) ────────
          if (error.response?.data) {
            if (typeof error.response.data === 'string') {
              nvidiaErrorBody = error.response.data.substring(0, 500);
              try {
                const parsed = JSON.parse(error.response.data);
                if (parsed?.detail) {
                  nvidiaErrorBody = JSON.stringify(parsed.detail);
                }
              } catch {}
            } else if (typeof error.response.data === 'object') {
              try {
                nvidiaErrorBody = JSON.stringify(error.response.data).substring(0, 500);
              } catch {
                // Stream or circular object → safe fallback
                nvidiaErrorBody = `[Unstringifiable object: ${typeof error.response.data}]`;
              }
            } else {
              nvidiaErrorBody = String(error.response.data).substring(0, 500);
            }
          }

          const retryAfterHeader = error.response?.headers?.['retry-after'];
          if (retryAfterHeader) {
            retryAfterSec = parseInt(retryAfterHeader, 10);
            if (isNaN(retryAfterSec)) retryAfterSec = null;
          }

          console.log(`Attempt ${attempt} failed: HTTP ${status} / code ${code}`);
          if (nvidiaErrorBody) {
            console.log(`Nvidia error body: ${nvidiaErrorBody}`);
          }

          if (status === 429) {
            if (attempt === MAX_ATTEMPTS) break;

            let delay;
            if (retryAfterSec && retryAfterSec > 0 && retryAfterSec <= 120) {
              delay = retryAfterSec * 1000;
              console.log(`Using Retry-After header: ${delay}ms`);
            } else {
              delay = backoff(attempt);
              console.log(`No/Invalid Retry-After, using backoff: ${delay}ms`);
            }
            await sleep(delay);
            continue;
          }

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

          throw new Error(`Nvidia API error: ${status || code} – ${nvidiaErrorBody || error.message}`);
        }
      }

      if (!response) {
        throw new Error('Model did not respond after retries');
      }

      if (stream) {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.flushHeaders();

        if (!responseClosed) res.write(': warmup\n\n');

        const heartbeat = setInterval(() => {
          if (!responseClosed) res.write(': keep-alive\n\n');
        }, 15000);

        response.data.on('data', chunk => {
          if (!responseClosed) res.write(chunk);
        });

        response.data.on('end', () => {
          clearInterval(heartbeat);
          if (!responseClosed) res.end();
        });

        response.data.on('error', err => {
          clearInterval(heartbeat);
          console.error('Stream error:', err.message);
          if (!responseClosed) res.end();
        });

        return;
      }

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

  return new Promise(() => {});
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`Proxy running on port ${PORT}`);
});
