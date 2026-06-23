import test from 'node:test';
import assert from 'node:assert/strict';
import { runAgentTask, type ModelClient, type ToolRegistry } from '../src/index.js';

const repoContext = { root: '/tmp/repo' };

function tools(): ToolRegistry {
  return {
    definitions() {
      return [{ name: 'echo', description: 'Echo args', inputSchema: { type: 'object' } }];
    },
    async run(name, args) {
      return { ok: true, content: { name, args } };
    },
  };
}

test('returns final model response', async () => {
  const model: ModelClient = {
    async step() {
      return { type: 'final', content: 'done' };
    },
  };

  const result = await runAgentTask({
    model,
    tools: tools(),
    messages: [{ role: 'user', content: 'go' }],
    repoContext,
    maxSteps: 4,
  });

  assert.equal(result.ok, true);
  assert.equal(result.content, 'done');
  assert.equal(result.steps, 1);
});

test('runs tool calls and appends tool results', async () => {
  let calls = 0;
  const model: ModelClient = {
    async step() {
      calls += 1;
      if (calls === 1) {
        return { id: 'call_1', type: 'tool_call', tool: 'echo', args: { value: 7 } };
      }
      return { type: 'final', content: 'observed' };
    },
  };

  const result = await runAgentTask({
    model,
    tools: tools(),
    messages: [{ role: 'user', content: 'use a tool' }],
    repoContext,
    maxSteps: 4,
  });

  assert.equal(result.ok, true);
  assert.equal(result.steps, 2);
  assert.equal(result.messages[1]?.role, 'tool');
});

test('stops after max steps', async () => {
  const model: ModelClient = {
    async step() {
      return { id: 'call_loop', type: 'tool_call', tool: 'echo', args: {} };
    },
  };

  const result = await runAgentTask({
    model,
    tools: tools(),
    messages: [{ role: 'user', content: 'loop' }],
    repoContext,
    maxSteps: 2,
  });

  assert.equal(result.ok, false);
  assert.equal(result.steps, 2);
  assert.equal(result.content, 'Agent exceeded max steps.');
}
);
