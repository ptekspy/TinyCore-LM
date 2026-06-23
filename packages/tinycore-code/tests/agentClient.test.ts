import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { TinycoredClient } from '../src/agentClient.js';

async function withServer(fn: (baseUrl: string, bodies: unknown[]) => Promise<void>): Promise<void> {
  const bodies: unknown[] = [];
  const server = http.createServer((req, res) => {
    if (req.method === 'GET' && req.url === '/health') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ ok: true, model_loaded: false, runtime: 'test' }));
      return;
    }
    if (req.method === 'POST' && req.url === '/agent/step') {
      void readBody(req).then((body) => {
        bodies.push(body);
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: true, content: ['README.md'], metadata: { count: 1 } }));
      });
      return;
    }
    if (req.method === 'POST' && req.url === '/generate') {
      void readBody(req).then((body) => {
        bodies.push(body);
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ text: 'generated', tokens: [1], metrics: { tokens_per_sec: 1 }, runtime: 'test', model: {} }));
      });
      return;
    }
    if (req.method === 'POST' && req.url === '/chat') {
      void readBody(req).then((body) => {
        bodies.push(body);
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ type: 'message', role: 'assistant', content: 'chat response', runtime: 'test' }));
      });
      return;
    }
    res.writeHead(404, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ok: false, error: 'missing' }));
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  assert.equal(typeof address, 'object');
  assert.notEqual(address, null);
  const port = (address as { port: number }).port;
  try {
    await fn(`http://127.0.0.1:${port}`, bodies);
  } finally {
    await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
  }
}

async function readBody(req: http.IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
}

test('TinycoredClient reads health', async () => {
  await withServer(async (baseUrl) => {
    const client = new TinycoredClient(baseUrl);
    assert.deepEqual(await client.health(), { ok: true, model_loaded: false, runtime: 'test' });
  });
});

test('TinycoredClient executes agent step', async () => {
  await withServer(async (baseUrl) => {
    const client = new TinycoredClient(baseUrl);
    assert.deepEqual(await client.agentStep('list_files', {}), {
      ok: true,
      content: ['README.md'],
      metadata: { count: 1 },
    });
  });
});

test('TinycoredClient generates text', async () => {
  await withServer(async (baseUrl, bodies) => {
    const client = new TinycoredClient(baseUrl);
    assert.deepEqual(await client.generate({ prompt: 'TinyCore', max_tokens: 1, temperature: 0.8, top_k: 4, seed: 2026 }), {
      text: 'generated',
      tokens: [1],
      metrics: { tokens_per_sec: 1 },
      runtime: 'test',
      model: {},
    });
    assert.deepEqual(bodies.at(-1), { prompt: 'TinyCore', max_tokens: 1, temperature: 0.8, top_k: 4, seed: 2026 });
  });
});

test('TinycoredClient sends chat messages', async () => {
  await withServer(async (baseUrl, bodies) => {
    const client = new TinycoredClient(baseUrl);
    assert.deepEqual(await client.chat({ messages: [{ role: 'user', content: 'hello' }], max_tokens: 8, temperature: 0, top_k: 1, seed: 7 }), {
      type: 'message',
      role: 'assistant',
      content: 'chat response',
      runtime: 'test',
    });
    assert.deepEqual(bodies.at(-1), {
      messages: [{ role: 'user', content: 'hello' }],
      max_tokens: 8,
      temperature: 0,
      top_k: 1,
      seed: 7,
    });
  });
});

test('TinycoredClient can send direct chat prompts', async () => {
  await withServer(async (baseUrl, bodies) => {
    const client = new TinycoredClient(baseUrl);
    assert.deepEqual(await client.chat({ prompt: 'Q:add|A:', max_tokens: 16, temperature: 0 }), {
      type: 'message',
      role: 'assistant',
      content: 'chat response',
      runtime: 'test',
    });
    assert.deepEqual(bodies.at(-1), {
      prompt: 'Q:add|A:',
      max_tokens: 16,
      temperature: 0,
    });
  });
});
