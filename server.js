const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// ✅ Stable model mapping
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
    data: Object.keys(MODEL_MAPPING).map(m => ({
      id: m,
      object: 'model',
      created: Date.now(),
      owned_by: 'nvidia-nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  const startTime = Date.now();

  // 🔥 total allowed time (safe for Render + CF)
  const MAX_TOTAL_TIME = 12 * 60 * 1000; // 12 minutes

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const nimModel = MODEL_MAPPING[model] || 'deepseek-ai/deepseek-v3.1';

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 16384,
      stream: stream || false
    };

    let attempt = 0;
    let response;

    while (!response) {
      attempt++;

      // 🔥 stop if too long
      if (Date.now() - startTime > MAX_TOTAL_TIME) {
        throw new Error('Global timeout reached');
      }

      try {
        console.log(`Attempt ${attempt} for ${nimModel}`);

        response = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              'Authorization': `Bearer ${NIM_API_KEY}`,
              'Content-Type': 'application/json'
            },
            timeout: 60000, // ⬅️ longer per attempt
            responseType: stream ? 'stream' : 'json'
          }
        );

        console.log(`Success on attempt ${attempt}`);

      } catch (error) {
        const status = error.response?.status;
        const code = error.code;

        console.log(`Attempt ${attempt} failed: ${code || status}`);

        // 🔥 exponential backoff
        let delay = Math.min(2000 * Math.pow(1.5, attempt), 20000);

        if (status === 429) delay = 20000; // rate limit
        if (status === 503 || status === 504) delay = 5000;

        if (
          code === 'ECONNABORTED' ||
          status === 429 ||
          status === 503 ||
          status === 504
        ) {
          await new Promise(r => setTimeout(r, delay));
        } else {
          throw error;
        }
      }
    }

    // 🔥 STREAMING (critical for CF survival)
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      res.flushHeaders();

      response.data.on('data', chunk => {
        res.write(chunk);
      });

      response.data.on('end', () => {
        res.end();
      });

      response.data.on('error', err => {
        console.error('Stream error:', err);
        res.end();
      });

      return;
    }

    // 🔥 normal response
    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: response.data.choices.map(c => ({
        index: c.index,
        message: c.message,
        finish_reason: c.finish_reason
      })),
      usage: response.data.usage || {}
    });

  } catch (error) {
    console.error('Proxy error FULL:', {
      message: error.message,
      status: error.response?.status,
      data: error.response?.data
    });

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: error.response?.data || {
          message: error.message,
          type: 'proxy_error'
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
  console.log(`Proxy running on port ${PORT}`);
});
