const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const NIM_API_BASE =
  process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NIM_API_KEY;

const httpAgent = new http.Agent({
  keepAlive: true
});

const httpsAgent = new https.Agent({
  keepAlive: true
});

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v4-pro'
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
      owned_by: 'nvidia-nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  req.setTimeout(0);
  res.setTimeout(0);

  let responseClosed = false;
  let currentController = null;

  // FIX: use response close instead of request close
  res.on('close', () => {
    if (!res.writableEnded) {
      responseClosed = true;

      if (currentController) {
        currentController.abort();
      }

      console.log('Client truly disconnected → aborting retries');
    }
  });

  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;

    const nimModel =
      MODEL_MAPPING[model] ||
      'deepseek-ai/deepseek-v4-pro';

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 16384,
      stream: !!stream
    };

    const GLOBAL_TIMEOUT = 180000;
    const REQUEST_TIMEOUT = 30000;
    const MAX_ATTEMPTS = 8;

    let response = null;
    const startTime = Date.now();

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      if (responseClosed) {
        console.log('Stopping retries because response closed');
        return;
      }

      if (Date.now() - startTime > GLOBAL_TIMEOUT) {
        throw new Error('Global timeout reached');
      }

      try {
        currentController = new AbortController();

        console.log(
          `Attempt ${attempt}/${MAX_ATTEMPTS} → ${nimModel}`
        );

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

        console.log(`Success on attempt ${attempt}`);
        break;

      } catch (error) {
        if (responseClosed) {
          console.log('Client already gone');
          return;
        }

        const status = error.response?.status;
        const code = error.code;

        console.log(
          `Attempt ${attempt} failed: ${code || status}`
        );

        if (
          code === 'ECONNABORTED' ||
          code === 'ERR_CANCELED' ||
          status === 429 ||
          status === 500 ||
          status === 502 ||
          status === 503 ||
          status === 504
        ) {
          if (attempt === MAX_ATTEMPTS) {
            break;
          }

          await new Promise(r => setTimeout(r, 1500));
          continue;
        }

        throw error;
      }
    }

    if (!response) {
      throw new Error('Model did not respond after retries');
    }

    if (stream) {
      res.setHeader(
        'Content-Type',
        'text/event-stream'
      );
      res.setHeader(
        'Cache-Control',
        'no-cache'
      );
      res.setHeader(
        'Connection',
        'keep-alive'
      );

      res.flushHeaders();

      const heartbeat = setInterval(() => {
        if (!responseClosed) {
          res.write(': keep-alive\n\n');
        }
      }, 15000);

      response.data.on('data', chunk => {
        if (!responseClosed) {
          res.write(chunk);
        }
      });

      response.data.on('end', () => {
        clearInterval(heartbeat);

        if (!responseClosed) {
          res.end();
        }
      });

      response.data.on('error', err => {
        clearInterval(heartbeat);

        console.error(
          'Stream error:',
          err.message
        );

        if (!responseClosed) {
          res.end();
        }
      });

      return;
    }

    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(
        Date.now() / 1000
      ),
      model,
      choices: response.data.choices,
      usage:
        response.data.usage || {}
    });

  } catch (error) {
    console.error(
      'Proxy error FULL:',
      {
        message: error.message,
        status:
          error.response?.status,
        data:
          error.response?.data
      }
    );

    if (!res.headersSent) {
      res.status(
        error.response?.status || 500
      ).json({
        error: {
          message:
            error.response?.data?.error?.message ||
            error.message,
          type: 'proxy_error',
          code:
            error.response?.status || 500
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
  console.log(
    `Proxy running on port ${PORT}`
  );
});
