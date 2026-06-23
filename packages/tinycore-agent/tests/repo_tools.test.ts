import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRepoToolRegistry, runAgentTask, type ModelClient } from '../src/index.js';

async function makeRepo(): Promise<string> {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tinycore-agent-'));
  await fs.mkdir(path.join(root, 'src'));
  await fs.writeFile(path.join(root, 'README.md'), 'hello TinyCore\n');
  await fs.writeFile(path.join(root, 'src', 'index.ts'), 'export const answer = 42;\n');
  await fs.mkdir(path.join(root, 'node_modules'));
  await fs.writeFile(path.join(root, 'node_modules', 'ignored.js'), 'ignored\n');
  return root;
}

test('repo tools list, read, and search files inside root', async () => {
  const root = await makeRepo();
  const tools = createRepoToolRegistry({ root });

  const listed = await tools.run('list_files', {});
  assert.equal(listed.ok, true);
  assert.deepEqual(listed.content, ['README.md', 'src/index.ts']);

  const read = await tools.run('read_file', { path: 'src/index.ts' });
  assert.equal(read.ok, true);
  assert.equal(read.content, 'export const answer = 42;\n');

  const search = await tools.run('search_text', { query: 'answer' });
  assert.equal(search.ok, true);
  assert.deepEqual(search.content, [{ path: 'src/index.ts', line: 1, text: 'export const answer = 42;' }]);
});

test('repo tools reject path traversal', async () => {
  const root = await makeRepo();
  const tools = createRepoToolRegistry({ root });

  const result = await tools.run('read_file', { path: '../outside.txt' });
  assert.equal(result.ok, false);
  assert.match(String(result.content), /escapes repository root/);
});

test('run_command rejects destructive commands by default', async () => {
  const root = await makeRepo();
  const tools = createRepoToolRegistry({ root });

  const result = await tools.run('run_command', { command: ['rm', '-rf', 'src'] });
  assert.equal(result.ok, false);
  assert.match(String(result.content), /Refusing destructive command/);
});

test('run_command executes non-destructive commands', async () => {
  const root = await makeRepo();
  const tools = createRepoToolRegistry({ root });

  const result = await tools.run('run_command', {
    command: ['node', '-e', 'process.stdout.write("ok")'],
  });

  assert.equal(result.ok, true);
  assert.deepEqual(result.content, { stdout: 'ok', stderr: '' });
});

test('git tools return status and diff', async () => {
  const root = await makeRepo();
  const tools = createRepoToolRegistry({ root });

  const init = await tools.run('run_command', { command: ['git', 'init'] });
  assert.equal(init.ok, true);
  await tools.run('run_command', { command: ['git', 'config', 'user.email', 'tinycore@example.test'] });
  await tools.run('run_command', { command: ['git', 'config', 'user.name', 'TinyCore Test'] });
  await tools.run('run_command', { command: ['git', 'add', 'README.md', 'src/index.ts'] });
  await tools.run('run_command', { command: ['git', 'commit', '-m', 'initial'] });
  await fs.appendFile(path.join(root, 'README.md'), 'changed\n');

  const status = await tools.run('git_status', {});
  assert.equal(status.ok, true);
  assert.match(JSON.stringify(status.content), /README.md/);

  const diff = await tools.run('git_diff', {});
  assert.equal(diff.ok, true);
  assert.match(JSON.stringify(diff.content), /changed/);
});

test('search_symbols finds TypeScript symbols', async () => {
  const root = await makeRepo();
  const tools = createRepoToolRegistry({ root });

  const result = await tools.run('search_symbols', { query: 'answer' });

  assert.equal(result.ok, true);
  assert.deepEqual(result.content, [
    {
      path: 'src/index.ts',
      line: 1,
      kind: 'value',
      name: 'answer',
      text: 'export const answer = 42;',
    },
  ]);
});

test('write_file_via_patch applies a unique replacement', async () => {
  const root = await makeRepo();
  const tools = createRepoToolRegistry({ root });

  const result = await tools.run('write_file_via_patch', {
    path: 'README.md',
    oldText: 'hello TinyCore\n',
    newText: 'hello TinyCore agent\n',
  });

  assert.equal(result.ok, true);
  assert.equal(await fs.readFile(path.join(root, 'README.md'), 'utf8'), 'hello TinyCore agent\n');
});

test('write_file_via_patch rejects ambiguous replacements', async () => {
  const root = await makeRepo();
  await fs.writeFile(path.join(root, 'README.md'), 'same\nsame\n');
  const tools = createRepoToolRegistry({ root });

  const result = await tools.run('write_file_via_patch', {
    path: 'README.md',
    oldText: 'same\n',
    newText: 'changed\n',
  });

  assert.equal(result.ok, false);
  assert.match(String(result.content), /matched more than once/);
});

test('repo tools can be used by the agent loop', async () => {
  const root = await makeRepo();
  let calls = 0;
  const model: ModelClient = {
    async step() {
      calls += 1;
      if (calls === 1) {
        return { id: 'read_1', type: 'tool_call', tool: 'read_file', args: { path: 'README.md' } };
      }
      return { type: 'final', content: 'read complete' };
    },
  };

  const result = await runAgentTask({
    model,
    tools: createRepoToolRegistry({ root }),
    messages: [{ role: 'user', content: 'read the README' }],
    repoContext: { root },
    maxSteps: 3,
  });

  assert.equal(result.ok, true);
  assert.equal(result.messages[1]?.role, 'tool');
  assert.match(JSON.stringify(result.messages[1]), /hello TinyCore/);
});
