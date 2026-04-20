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

// ✅ FORCE ONLY V3.1
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v3.1',
  'gpt-4': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1'
};

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to NVIDIA NIM Proxy' });
});

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

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // ✅ NO FALLBACK TO V3.2 EVER
    const nimModel = MODEL_MAPPING[model] || 'deepseek-ai/deepseek-v3.1-terminus';

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 16384,
      stream: stream || false
    };

    let response;
    let attempt = 0;
    const maxAttempts = 5; // ✅ sane retry

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
            timeout: 45000,
            responseType: stream ? 'stream' : 'json'
          }
        );

        console.log(`Success on attempt ${attempt}`);

      } catch (error) {
        console.log(`Attempt ${attempt} failed: ${error.code || error.response?.status}`);

        if (
          error.code === 'ECONNABORTED' ||
          error.response?.status === 503 ||
          error.response?.status === 504
        ) {
          await new Promise(r => setTimeout(r, 2000));
        } else if (error.response?.status === 429) {
          await new Promise(r => setTimeout(r, 10000));
        } else {
          throw error;
        }
      }
    }

    if (!response) {
      throw new Error('Model did not respond after retries');
    }

    // ✅ STREAMING
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      response.data.on('data', chunk => res.write(chunk));
      response.data.on('end', () => res.end());
      response.data.on('error', err => {
        console.error('Stream error:', err);
        res.end();
      });

      return;
    }

    // ✅ NORMAL RESPONSE
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: model,
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
    };

    res.json(openaiResponse);

  } catch (error) {
    console.error('Proxy error:', error.message);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.response?.data?.error?.message || error.message,
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
  console.log(`Proxy running on port ${PORT}`);
});
