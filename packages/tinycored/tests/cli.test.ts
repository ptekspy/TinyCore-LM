import test from 'node:test';
import assert from 'node:assert/strict';
import { parseCliArgs } from '../src/cli.js';

test('parseCliArgs uses defaults', () => {
  const parsed = parseCliArgs([]);
  assert.equal(parsed.host, '127.0.0.1');
  assert.equal(parsed.port, 8787);
  assert.equal(parsed.repoRoot, process.cwd());
});

test('parseCliArgs accepts server and runtime options', () => {
  const parsed = parseCliArgs([
    '--repo-root',
    '/repo',
    '--artifact-dir',
    '/model',
    '--host',
    '0.0.0.0',
    '--port',
    '9000',
    '--python',
    'python3.12',
  ]);
  assert.deepEqual(parsed, {
    repoRoot: '/repo',
    artifactDir: '/model',
    host: '0.0.0.0',
    port: 9000,
    pythonBin: 'python3.12',
  });
});

test('parseCliArgs accepts native runtime options', () => {
  const parsed = parseCliArgs([
    '--repo-root',
    '/repo',
    '--tcmdl',
    '/model.tcmdl',
    '--native-bin',
    '/bin/tinycore-generate',
  ]);
  assert.deepEqual(parsed, {
    repoRoot: '/repo',
    tcmdl: '/model.tcmdl',
    nativeBin: '/bin/tinycore-generate',
    host: '127.0.0.1',
    port: 8787,
  });
});

test('parseCliArgs rejects invalid ports', () => {
  assert.throws(() => parseCliArgs(['--port', 'nope']), /Invalid port/);
});
