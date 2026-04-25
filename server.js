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

const httpAgent = new http.Agent({ keepAlive: true });
const httpsAgent = new https.Agent({ keepAlive: true });

// 🔒 locked models
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v3.1',
  'gpt-4': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1'
};

app.post('/v1/chat/completions', async (req, res) => {
  req.setTimeout(0);
  res.setTimeout(0);

  let clientAlive = true;
  let currentController = null;

  req.on('close', () => {
    clientAlive = false;
    if (currentController) currentController.abort();
    console.log('❌ Client disconnected → aborting everything');
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

    const GLOBAL_TIMEOUT = 180000; // 3 min
    const REQUEST_TIMEOUT = 30000; // 30s
    const MAX_ATTEMPTS = 8;

    const startTime = Date.now();
    let response = null;

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {

      if (!clientAlive) {
        console.log('🛑 Stopping retries (client gone)');
        return;
      }

      if (Date.now() - startTime > GLOBAL_TIMEOUT) {
        throw new Error('Global timeout reached');
      }

      currentController = new AbortController();

      try {
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
            responseType: stream ? 'stream' : 'json',
            httpAgent,
            httpsAgent,
            signal: currentController.signal
          }
        );

        console.log(`✅ Success on attempt ${attempt}`);
        break;

      } catch (error) {
        if (!clientAlive) {
          console.log('🛑 Aborted due to client disconnect');
          return;
        }

        const status = error.response?.status;
        const code = error.code;

        console.log(`❌ Attempt ${attempt} failed:`, code || status);

        if (
          code === 'ECONNABORTED' ||
          code === 'ERR_CANCELED' ||
          status === 429 ||
          status === 500 ||
          status === 502 ||
          status === 503 ||
          status === 504
        ) {
          if (attempt === MAX_ATTEMPTS) break;

          await new Promise(r => setTimeout(r, 1500));
          continue;
        }

        throw error;
      }
    }

    if (!response) {
      throw new Error('Model did not respond after retries');
    }

    // ================= STREAM =================
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      res.flushHeaders();

      const heartbeat = setInterval(() => {
        if (clientAlive) {
          res.write(': keep-alive\n\n');
        }
      }, 15000);

      response.data.on('data', chunk => {
        if (clientAlive) res.write(chunk);
      });

      response.data.on('end', () => {
        clearInterval(heartbeat);
        if (clientAlive) res.end();
      });

      response.data.on('error', err => {
        clearInterval(heartbeat);
        console.error('Stream error:', err);
        if (clientAlive) res.end();
      });

      return;
    }

    // ================= NORMAL =================
    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: response.data.choices,
      usage: response.data.usage || {}
    });

  } catch (error) {
    console.error('🔥 Proxy error:', error.message);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message,
          type: 'proxy_error',
          code: error.response?.status || 500
        }
      });
    }
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Running on port ${PORT}`);
});
