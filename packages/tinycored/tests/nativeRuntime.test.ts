import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { NativeModelRuntime } from '../src/nativeRuntime.js';

test('NativeModelRuntime invokes tinycore-generate compatible binary', async () => {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'tinycored-native-'));
  const logPath = path.join(tmp, 'args.json');
  const fakeBin = path.join(tmp, 'fake-native.mjs');
  await fs.writeFile(
    fakeBin,
    [
      '#!/usr/bin/env node',
      `await import('node:fs/promises').then((fs) => fs.writeFile(${JSON.stringify(logPath)}, JSON.stringify(process.argv.slice(2))));`,
      'console.log(JSON.stringify({',
      '  text: "TinyCore native",',
      '  tokens: [84, 105],',
      '  new_tokens: [105],',
      '  generation: { temperature: 0.7, top_k: 4, seed: 99 },',
      '  runtime: "native",',
      '  model: { architecture: "tinycore_recurrent_v0" }',
      '}));',
    ].join('\n'),
    { mode: 0o755 },
  );

  const runtime = new NativeModelRuntime({
    artifactPath: '/model.tcmdl',
    nativeBin: fakeBin,
  });
  const response = await runtime.generate({
    prompt: 'TinyCore',
    max_tokens: 5,
    temperature: 0.7,
    top_k: 4,
    seed: 99,
  });

  assert.equal(response.text, 'TinyCore native');
  assert.equal(response.runtime, 'native');
  assert.deepEqual(response.metrics, {});
  assert.deepEqual(JSON.parse(await fs.readFile(logPath, 'utf8')), [
    '/model.tcmdl',
    'TinyCore',
    '5',
    '0.7',
    '4',
    '99',
  ]);
});
