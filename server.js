const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// ENV
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v3.2',
  'gpt-4': 'deepseek-ai/deepseek-v3.2',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.2'
};

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to NVIDIA NIM Proxy' });
});

// Models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// 🔥 MAIN ROUTE (FIXED STREAMING + RETRY)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens } = req.body;

    const nimModel = MODEL_MAPPING[model] || 'deepseek-ai/deepseek-v3.2';

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 16384,
      stream: true
    };

    // ✅ OPEN CONNECTION IMMEDIATELY
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    // ✅ INITIAL HANDSHAKE (prevents Cloudflare kill)
    res.write(`data: {"choices":[{"delta":{"content":""}}]}\n\n`);

    // ✅ KEEP CONNECTION ALIVE
    const keepAlive = setInterval(() => {
      res.write(`:\n\n`);
    }, 15000);

    let attempt = 0;
    const maxAttempts = 15;
    let response;

    while (!response && attempt < maxAttempts) {
      attempt++;

      try {
        console.log(`Attempt ${attempt}/${maxAttempts} for ${nimModel}`);

        response = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              'Authorization': `Bearer ${NIM_API_KEY}`,
              'Content-Type': 'application/json'
            },
            timeout: 90000,
            responseType: 'stream'
          }
        );

        console.log(`✅ Success on attempt ${attempt}`);

      } catch (error) {
        console.log(`❌ Attempt ${attempt} failed: ${error.code || error.response?.status}`);

        if (
          error.code === 'ECONNABORTED' ||
          error.response?.status === 503 ||
          error.response?.status === 504
        ) {
          await new Promise(r => setTimeout(r, 4000));
        } else if (error.response?.status === 429) {
          await new Promise(r => setTimeout(r, 25000));
        } else {
          clearInterval(keepAlive);
          res.write(`data: {"error":"Upstream error"}\n\n`);
          res.end();
          return;
        }
      }
    }

    if (!response) {
      clearInterval(keepAlive);
      res.write(`data: {"error":"Timeout after retries"}\n\n`);
      res.end();
      return;
    }

    // ✅ STREAM DATA TO CLIENT
    response.data.on('data', (chunk) => {
      res.write(chunk);
    });

    response.data.on('end', () => {
      clearInterval(keepAlive);
      res.end();
    });

    response.data.on('error', (err) => {
      console.error('Stream error:', err);
      clearInterval(keepAlive);
      res.end();
    });

  } catch (error) {
    console.error('Proxy error:', error.message);

    if (!res.headersSent) {
      res.status(500).json({
        error: {
          message: error.message || 'Internal server error'
        }
      });
    }
  }
});

// 404 fallback
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});
