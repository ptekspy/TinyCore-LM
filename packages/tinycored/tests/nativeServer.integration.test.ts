import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsPromises from 'node:fs/promises';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import { createTinycoredServer } from '../src/index.js';
import { NativeModelRuntime } from '../src/nativeRuntime.js';

const ROOT = path.resolve(process.cwd(), '../..');
const NATIVE_GENERATOR = path.join(ROOT, 'runtime', 'tinycore.cpp', 'build', 'tinycore-generate');
const PERMUTED_TINYCORE_BUNDLE = path.join(
  ROOT,
  'reports',
  'runs',
  'instruction_code_permuted_tinycore',
  'tinycore_recurrent_lr4_state8.tcmdl',
);

test(
  'tinycored serves selected native TinyCore bundle through generate and direct-prompt chat',
  { skip: !fs.existsSync(NATIVE_GENERATOR) || !fs.existsSync(PERMUTED_TINYCORE_BUNDLE) },
  async () => {
    const repoRoot = await fsPromises.mkdtemp(path.join(path.dirname(process.cwd()), 'tinycored-native-server-'));
    const server = createTinycoredServer({
      repoRoot,
      modelRuntime: new NativeModelRuntime({
        artifactPath: PERMUTED_TINYCORE_BUNDLE,
        nativeBin: NATIVE_GENERATOR,
        timeoutMs: 60_000,
      }),
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const address = server.address();
    assert.equal(typeof address, 'object');
    assert.notEqual(address, null);
    const baseUrl = `http://127.0.0.1:${(address as AddressInfo).port}`;

    try {
      const generate = await postJson(baseUrl, '/generate', {
        prompt: 'Q:add|A:',
        max_tokens: 16,
        temperature: 0,
        top_k: 0,
        seed: 1337,
      });
      assert.equal(generate.text, 'Q:add|A:def add(a,b):ret');
      assert.equal(generate.runtime, 'native');

      const chat = await postJson(baseUrl, '/chat', {
        prompt: 'Q:add|A:',
        max_tokens: 16,
        temperature: 0,
        top_k: 0,
        seed: 1337,
      });
      assert.equal(chat.type, 'message');
      assert.equal(chat.content, 'def add(a,b):ret');
      assert.equal(chat.runtime, 'native');
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
      });
      await fsPromises.rm(repoRoot, { recursive: true, force: true });
    }
  },
);

async function postJson(baseUrl: string, pathname: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
  const res = await fetch(`${baseUrl}${pathname}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  assert.equal(res.status, 200);
  return (await res.json()) as Record<string, unknown>;
}
