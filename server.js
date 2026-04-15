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

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gpt-4': 'deepseek-ai/deepseek-v3.2',
  'gpt-4-turbo': 'meta/llama-3.1-405b-instruct'
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

  // ✅ FIX 1: prevent early browser timeout issues
  res.setTimeout(120000);
  req.setTimeout(120000);

  console.log("Request received");

  try {
    const { model, messages, temperature, max_tokens } = req.body;

    const nimModel =
      MODEL_MAPPING[model] || MODEL_MAPPING['gpt-3.5-turbo'];

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 16384,
      stop: null,
      stream: false
    };

    let response;
    let attempt = 0;

    while (!response) {
      attempt++;

      try {
        console.log(`Attempt ${attempt} for model ${nimModel}...`);

        response = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              'Authorization': `Bearer ${NIM_API_KEY}`,
              'Content-Type': 'application/json'
            },

            // ✅ FIX 2: shorter timeout to avoid hanging connections
            timeout: 120000,
            responseType: 'json'
          }
        );

        console.log(`✓ Success on attempt ${attempt}!`);

      } catch (error) {
        console.log(`Attempt ${attempt} failed:`, error.code || error.message);

        // retry only on temporary failures
        if (
          error.code === 'ECONNABORTED' ||
          error.response?.status === 503 ||
          error.response?.status === 504
        ) {
          await new Promise(r => setTimeout(r, 2000));
        } else {
          throw error;
        }
      }
    }

    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: (response.data.choices || []).map((choice, i) => ({
        index: i,
        message: choice.message || {
          role: "assistant",
          content: choice.text || ""
        },
        finish_reason: choice.finish_reason || "stop"
      })),
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    // ✅ FIX 3: safe response headers + explicit send
    res.setHeader("Content-Type", "application/json");
    res.status(200).json(openaiResponse);

  } catch (error) {
    console.error('Proxy error:', error.message);

    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
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
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});
