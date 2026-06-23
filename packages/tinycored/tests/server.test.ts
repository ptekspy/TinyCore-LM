import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import type { AddressInfo } from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { createTinycoredServer, type ModelRuntime } from '../src/index.js';

async function withServer(fn: (baseUrl: string) => Promise<void>, modelRuntime?: ModelRuntime): Promise<void> {
  const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'tinycored-'));
  await fs.writeFile(path.join(repoRoot, 'README.md'), 'TinyCore server test\n');
  const server = createTinycoredServer({ repoRoot, modelRuntime });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  assert.equal(typeof address, 'object');
  assert.notEqual(address, null);
  const baseUrl = `http://127.0.0.1:${(address as AddressInfo).port}`;
  try {
    await fn(baseUrl);
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  }
}

test('health endpoint returns server status', async () => {
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/health`);
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), { ok: true, model_loaded: false, runtime: 'typescript-mvp' });
  });
});

test('agent step endpoint executes repo tools', async () => {
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/agent/step`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ tool: 'read_file', args: { path: 'README.md' } }),
    });
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.ok, true);
    assert.equal(body.content, 'TinyCore server test\n');
  });
});

test('generate endpoint is explicit about missing model runtime', async () => {
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/generate`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt: 'hello' }),
    });
    assert.equal(res.status, 501);
    const body = await res.json();
    assert.equal(body.ok, false);
    assert.match(body.error, /Model runtime is not wired/);
  });
});

test('generate endpoint uses configured model runtime', async () => {
  const runtime: ModelRuntime = {
    name: 'fake-python',
    async generate(request) {
      assert.equal(request.top_k, 4);
      return {
        text: `${request.prompt} generated`,
        tokens: [1, 2, 3],
        metrics: { tokens_per_sec: 123 },
        runtime: 'fake-python',
        model: { architecture: 'test' },
      };
    },
  };
  await withServer(async (baseUrl) => {
    const health = await fetch(`${baseUrl}/health`);
    assert.deepEqual(await health.json(), { ok: true, model_loaded: true, runtime: 'fake-python' });

    const res = await fetch(`${baseUrl}/generate`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt: 'TinyCore', max_tokens: 3, top_k: 4 }),
    });
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.text, 'TinyCore generated');
    assert.deepEqual(body.tokens, [1, 2, 3]);
  }, runtime);
});

test('chat endpoint generates from messages with configured runtime', async () => {
  const runtime: ModelRuntime = {
    name: 'fake-python',
    async generate(request) {
      assert.match(request.prompt, /<user>\nHello/);
      assert.equal(request.top_k, 2);
      return {
        text: 'assistant reply',
        tokens: [9],
        metrics: { tokens_per_sec: 9 },
        runtime: 'fake-python',
        model: { architecture: 'test' },
      };
    },
  };
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'Hello' }], max_tokens: 8, top_k: 2 }),
    });
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.type, 'message');
    assert.equal(body.role, 'assistant');
    assert.equal(body.content, 'assistant reply');
    assert.equal(body.runtime, 'fake-python');
  }, runtime);
});

test('chat endpoint strips echoed prompt from runtime text', async () => {
  const runtime: ModelRuntime = {
    name: 'echoing-runtime',
    async generate(request) {
      return {
        text: `${request.prompt}assistant reply`,
        tokens: [1],
        metrics: {},
        runtime: 'echoing-runtime',
        model: { architecture: 'test' },
      };
    },
  };
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'Hello' }], max_tokens: 8 }),
    });
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.content, 'assistant reply');
  }, runtime);
});

test('chat endpoint can use a direct prompt for compact local model evals', async () => {
  const runtime: ModelRuntime = {
    name: 'direct-prompt-runtime',
    async generate(request) {
      assert.equal(request.prompt, 'Q:add|A:');
      assert.equal(request.temperature, 0);
      return {
        text: 'Q:add|A:def add(a,b):ret',
        tokens: [1],
        metrics: {},
        runtime: 'direct-prompt-runtime',
        model: { architecture: 'test' },
      };
    },
  };
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt: 'Q:add|A:', max_tokens: 16, temperature: 0 }),
    });
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.type, 'message');
    assert.equal(body.content, 'def add(a,b):ret');
    assert.equal(body.runtime, 'direct-prompt-runtime');
  }, runtime);
});

test('chat endpoint can execute explicit tool calls without runtime', async () => {
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        tool_call: { id: 'tool_1', tool: 'read_file', args: { path: 'README.md' } },
      }),
    });
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.type, 'tool_result');
    assert.equal(body.tool_call_id, 'tool_1');
    assert.equal(body.ok, true);
    assert.equal(body.content, 'TinyCore server test\n');
  });
});

test('chat endpoint requires runtime for message generation', async () => {
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'Hello' }] }),
    });
    assert.equal(res.status, 501);
    const body = await res.json();
    assert.match(body.error, /requires a configured model runtime/);
  });
});
