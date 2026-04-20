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

    // 🔥 STEP 1: OPEN STREAM IMMEDIATELY (CRITICAL)
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    // 🔥 STEP 2: SEND INITIAL KEEP-ALIVE (VERY IMPORTANT)
    res.write(`data: {"choices":[{"delta":{"content":""}}]}\n\n`);

    let attempt = 0;
    const maxAttempts = 15;
    let response;

    // 🔥 STEP 3: KEEP CONNECTION ALIVE WHILE WAITING
    const keepAlive = setInterval(() => {
      res.write(`:\n\n`); // SSE comment ping
    }, 15000);

    while (!response && attempt < maxAttempts) {
      attempt++;

      try {
        console.log(`Attempt ${attempt}/${maxAttempts}`);

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

        console.log(`Success on attempt ${attempt}`);

      } catch (error) {
        console.log(`Attempt ${attempt} failed: ${error.code || error.response?.status}`);

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
      res.write(`data: {"error":"Timeout"}\n\n`);
      res.end();
      return;
    }

    // 🔥 STEP 4: PIPE REAL STREAM
    response.data.on('data', (chunk) => {
      res.write(chunk);
    });

    response.data.on('end', () => {
      clearInterval(keepAlive);
      res.end();
    });

    response.data.on('error', () => {
      clearInterval(keepAlive);
      res.end();
    });

  } catch (error) {
    console.error('Proxy error:', error.message);

    if (!res.headersSent) {
      res.status(500).json({
        error: {
          message: error.message
        }
      });
    }
  }
});
