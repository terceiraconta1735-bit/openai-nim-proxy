const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 Keep-alive agents (HUGE stability boost)
const httpAgent = new http.Agent({ keepAlive: true });
const httpsAgent = new https.Agent({ keepAlive: true });

// ✅ LOCKED MODELS
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v3.1',
  'gpt-4': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1'
};

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(model => ({
      id: model,
      object: 'model',
      created: Date.now(),
      owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let clientDisconnected = false;

  req.on('close', () => {
    clientDisconnected = true;
    console.log('Client disconnected');
  });

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const nimModel = MODEL_MAPPING[model] || 'deepseek-ai/deepseek-v3.1-terminus';

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 16384,
      stream: !!stream
    };

    let response = null;

    const maxAttempts = 12;
    let delay = 1500;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      if (clientDisconnected) return;

      try {
        console.log(`Attempt ${attempt}/${maxAttempts} → ${nimModel}`);

        response = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              Authorization: `Bearer ${NIM_API_KEY}`,
              'Content-Type': 'application/json'
            },
            timeout: 60000,
            responseType: stream ? 'stream' : 'json',
            httpAgent,
            httpsAgent,
            decompress: true,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
          }
        );

        console.log(`✅ Success on attempt ${attempt}`);
        break;

      } catch (error) {
        const status = error.response?.status;
        const code = error.code;

        console.log(`❌ Attempt ${attempt} failed:`, code || status);

        // 🔥 Only retry safe cases
        if (
          code === 'ECONNABORTED' ||
          status === 429 ||
          status === 500 ||
          status === 502 ||
          status === 503 ||
          status === 504
        ) {
          if (attempt === maxAttempts) break;

          // 🔥 Exponential backoff (critical)
          await new Promise(r => setTimeout(r, delay));
          delay = Math.min(delay * 1.5, 10000);
          continue;
        }

        // ❌ Non-retryable
        throw error;
      }
    }

    if (!response) {
      throw new Error('Model did not respond after retries');
    }

    // =========================
    // 🔥 STREAMING MODE
    // =========================
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      res.flushHeaders();

      // 🔥 heartbeat every 15s (prevents CF kill)
      const heartbeat = setInterval(() => {
        if (!res.writableEnded) {
          res.write(': keep-alive\n\n');
        }
      }, 15000);

      response.data.on('data', chunk => {
        if (!clientDisconnected) res.write(chunk);
      });

      response.data.on('end', () => {
        clearInterval(heartbeat);
        res.end();
      });

      response.data.on('error', err => {
        clearInterval(heartbeat);
        console.error('Stream error:', err);
        res.end();
      });

      return;
    }

    // =========================
    // 🔥 NORMAL MODE
    // =========================
    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: response.data.choices.map(choice => ({
        index: choice.index,
        message: choice.message,
        finish_reason: choice.finish_reason
      })),
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    });

  } catch (error) {
    console.error('🔥 Proxy error:', error.message);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message:
            error.response?.data?.error?.message ||
            error.message ||
            'Unknown proxy error',
          type: 'proxy_error',
          code: error.response?.status || 500
        }
      });
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Proxy running on port ${PORT}`);
});
