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

// ----- Concurrency limiter (prevents flooding Nvidia) -----
let activeRequests = 0;
const MAX_CONCURRENT = 2;   // send at most 2 requests at a time

function canProceed() {
  return activeRequests < MAX_CONCURRENT;
}

// ----- Helper functions -----
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function backoff(attempt) {
  const base = Math.min(2000 * Math.pow(2, attempt - 1), 60000);
  const jitter = Math.random() * 1000;
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

  // ----- Concurrency gate: wait until a slot opens -----
  while (!canProceed() && !responseClosed) {
    await sleep(500);
  }
  if (responseClosed) return;
  activeRequests++;

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    const nimModel = MODEL_MAPPING[model] || 'deepseek-ai/deepseek-v4-pro';

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 16384,    // keep your long replies
      stream: !!stream
    };

    const MAX_ATTEMPTS = 5;   // slightly reduced to not hammer for 10 min on obvious errors
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
            responseType: stream ? 'stream' : 'text',   // ← get as text for full error visibility
            httpAgent,
            httpsAgent,
            signal: currentController.signal,
            transformResponse: [(data) => data]          // keep raw string, we'll parse manually
          }
        );

        // If non-streaming, parse the response text into JSON
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

        // ─── Extract error details robustly ────────────────
        const status = error.response?.status;
        const code = error.code;
        let nvidiaErrorMessage = '';

        // Try to read the error body as text, even if parsing failed
        if (error.response?.data) {
          if (typeof error.response.data === 'string') {
            nvidiaErrorMessage = error.response.data.substring(0, 500);
          } else {
            try {
              nvidiaErrorMessage = JSON.stringify(error.response.data).substring(0, 500);
            } catch {}
          }
        }

        console.log(`Attempt ${attempt} failed: HTTP ${status} / code ${code}`);
        if (nvidiaErrorMessage) {
          console.log(`Nvidia error body: ${nvidiaErrorMessage}`);
        }

        // ─── Decide whether to retry ──────────────────────
        // Always retry on network/timeout errors and on 429 rate limit
        const isNetworkError = code === 'ECONNABORTED' || code === 'ERR_CANCELED';
        const isRateLimit = status === 429;
        // CRITICAL: some rate limits come back as 400 with a specific message
        const isPotentialRateLimit400 = status === 400 && nvidiaErrorMessage.toLowerCase().includes('rate');

        const shouldRetry = isNetworkError || isRateLimit || isPotentialRateLimit400 ||
                            status === 500 || status === 502 || status === 503 || status === 504;

        if (!shouldRetry) {
          // Non-retryable error → stop and throw immediately
          throw new Error(`Nvidia API error: ${status || code} – ${nvidiaErrorMessage || error.message}`);
        }

        // Retry with backoff
        if (attempt === MAX_ATTEMPTS) break;

        const delay = backoff(attempt);
        console.log(`Backing off for ${delay}ms before next attempt`);
        await sleep(delay);
      }
    }

    if (!response) {
      throw new Error('Model did not respond after retries');
    }

    // ----- Streaming path -----
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders();

      // Warmup comment to keep Render alive
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

    // ----- Non‑streaming path -----
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
          message: error.response?.data?.error?.message || error.message,
          type: 'proxy_error',
          code: error.response?.status || 502
        }
      });
    }
  } finally {
    activeRequests = Math.max(0, activeRequests - 1);
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`Proxy running on port ${PORT}`);
});
