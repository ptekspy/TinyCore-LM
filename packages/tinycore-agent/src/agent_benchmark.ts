import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { createRepoToolRegistry } from './repo_tools.js';
import type { ToolResult } from './index.js';

const execFileAsync = promisify(execFile);

export type AgentBenchmarkReport = {
  run_id: string;
  task: string;
  repo_root: string;
  success: boolean;
  tests_passed_before_patch: boolean;
  tests_passed_after_patch: boolean;
  tool_calls: number;
  invalid_tool_calls: number;
  hallucinated_file_rate: number;
  files_touched: number;
  lines_changed: number;
  wall_clock_time_sec: number;
  notes: string[];
};

export type AgentBenchmarkSuiteReport = {
  run_id: string;
  task_success_rate: number;
  tasks_total: number;
  tasks_succeeded: number;
  total_tool_calls: number;
  total_invalid_tool_calls: number;
  total_files_touched: number;
  total_lines_changed: number;
  hallucinated_file_rate: number;
  wall_clock_time_sec: number;
  tasks: AgentBenchmarkReport[];
};

type BenchmarkTask = {
  runId: string;
  task: string;
  testCommand: string[];
  setup(root: string): Promise<void>;
  repair(call: ToolCaller): Promise<void>;
};

type ToolCaller = (
  tool: string,
  args: Record<string, unknown>,
  options?: { expectedFailure?: boolean },
) => Promise<ToolResult>;

export async function runSyntheticAgentBenchmark(options: {
  repoRoot?: string;
  keepRepo?: boolean;
} = {}): Promise<AgentBenchmarkReport> {
  return runBenchmarkTask(arithmeticBugTask(), options);
}

export async function runSyntheticAgentBenchmarkSuite(options: {
  keepRepos?: boolean;
} = {}): Promise<AgentBenchmarkSuiteReport> {
  const start = performance.now();
  const tasks = [];
  for (const task of [arithmeticBugTask(), missingFunctionTask()]) {
    tasks.push(await runBenchmarkTask(task, { keepRepo: options.keepRepos }));
  }
  const succeeded = tasks.filter((task) => task.success).length;
  const totalToolCalls = tasks.reduce((sum, task) => sum + task.tool_calls, 0);
  const totalInvalidToolCalls = tasks.reduce((sum, task) => sum + task.invalid_tool_calls, 0);
  return {
    run_id: 'synthetic_agent_suite_v0',
    task_success_rate: succeeded / tasks.length,
    tasks_total: tasks.length,
    tasks_succeeded: succeeded,
    total_tool_calls: totalToolCalls,
    total_invalid_tool_calls: totalInvalidToolCalls,
    total_files_touched: tasks.reduce((sum, task) => sum + task.files_touched, 0),
    total_lines_changed: tasks.reduce((sum, task) => sum + task.lines_changed, 0),
    hallucinated_file_rate: 0,
    wall_clock_time_sec: (performance.now() - start) / 1000,
    tasks,
  };
}

async function runBenchmarkTask(task: BenchmarkTask, options: {
  repoRoot?: string;
  keepRepo?: boolean;
} = {}): Promise<AgentBenchmarkReport> {
  const start = performance.now();
  const repoRoot = options.repoRoot ?? await fs.mkdtemp(path.join(os.tmpdir(), 'tinycore-agent-eval-'));
  await task.setup(repoRoot);
  const tools = createRepoToolRegistry({ root: repoRoot });
  let toolCalls = 0;
  let invalidToolCalls = 0;
  const notes: string[] = [];

  const call: ToolCaller = async (tool, args, options = {}) => {
    toolCalls += 1;
    const result = await tools.run(tool, args);
    if (!result.ok && !options.expectedFailure) {
      invalidToolCalls += 1;
      notes.push(`${tool} failed: ${String(result.content).slice(0, 160)}`);
    }
    return result;
  };

  const before = await call('run_tests', { command: task.testCommand }, { expectedFailure: true });
  await task.repair(call);
  const after = await call('run_tests', { command: task.testCommand });
  const diff = await call('git_diff', {});
  const stats = diffStats(String((diff.content as { stdout?: unknown }).stdout ?? ''));

  const testsPassedBefore = before.ok;
  const testsPassedAfter = after.ok;
  const success = !testsPassedBefore && testsPassedAfter;
  if (!options.keepRepo && options.repoRoot === undefined) {
    await fs.rm(repoRoot, { recursive: true, force: true });
  }

  return {
    run_id: task.runId,
    task: task.task,
    repo_root: repoRoot,
    success,
    tests_passed_before_patch: testsPassedBefore,
    tests_passed_after_patch: testsPassedAfter,
    tool_calls: toolCalls,
    invalid_tool_calls: invalidToolCalls,
    hallucinated_file_rate: 0,
    files_touched: stats.filesTouched,
    lines_changed: stats.linesChanged,
    wall_clock_time_sec: (performance.now() - start) / 1000,
    notes,
  };
}

export async function createSyntheticBugRepo(root: string): Promise<void> {
  await fs.mkdir(path.join(root, 'src'), { recursive: true });
  await fs.mkdir(path.join(root, 'test'), { recursive: true });
  await fs.writeFile(
    path.join(root, 'package.json'),
    JSON.stringify({ type: 'module', scripts: { test: 'node --test' } }, null, 2) + '\n',
  );
  await fs.writeFile(
    path.join(root, 'src', 'math.js'),
    [
      'export function add(a, b) {',
      '  return a - b;',
      '}',
      '',
    ].join('\n'),
  );
  await fs.writeFile(
    path.join(root, 'test', 'math.test.js'),
    [
      "import test from 'node:test';",
      "import assert from 'node:assert/strict';",
      "import { add } from '../src/math.js';",
      '',
      "test('add sums two numbers', () => {",
      '  assert.equal(add(2, 3), 5);',
      '});',
      '',
    ].join('\n'),
  );
  await initGit(root);
}

async function createSyntheticMissingFunctionRepo(root: string): Promise<void> {
  await fs.mkdir(path.join(root, 'src'), { recursive: true });
  await fs.mkdir(path.join(root, 'test'), { recursive: true });
  await fs.writeFile(
    path.join(root, 'package.json'),
    JSON.stringify({ type: 'module', scripts: { test: 'node test/string.test.js' } }, null, 2) + '\n',
  );
  await fs.writeFile(
    path.join(root, 'src', 'string.js'),
    [
      'export function slugify(input) {',
      "  throw new Error('TODO');",
      '}',
      '',
    ].join('\n'),
  );
  await fs.writeFile(
    path.join(root, 'test', 'string.test.js'),
    [
      "import assert from 'node:assert/strict';",
      "import { slugify } from '../src/string.js';",
      '',
      "assert.equal(slugify(' Hello World '), 'hello-world');",
      '',
    ].join('\n'),
  );
  await initGit(root);
}

function arithmeticBugTask(): BenchmarkTask {
  return {
    runId: 'synthetic_bugfix_v0',
    task: 'fix failing unit test due to arithmetic bug',
    testCommand: ['node', 'test/math.test.js'],
    setup: createSyntheticBugRepo,
    async repair(call) {
      await call('read_file', { path: 'src/math.js' });
      await call('search_text', { query: 'return a - b' });
      await call('write_file_via_patch', {
        path: 'src/math.js',
        oldText: '  return a - b;\n',
        newText: '  return a + b;\n',
      });
    },
  };
}

function missingFunctionTask(): BenchmarkTask {
  return {
    runId: 'synthetic_missing_function_v0',
    task: 'implement missing string slugify function',
    testCommand: ['node', 'test/string.test.js'],
    setup: createSyntheticMissingFunctionRepo,
    async repair(call) {
      await call('read_file', { path: 'src/string.js' });
      await call('search_symbols', { query: 'slugify' });
      await call('write_file_via_patch', {
        path: 'src/string.js',
        oldText: "  throw new Error('TODO');\n",
        newText: "  return input.trim().toLowerCase().replaceAll(' ', '-');\n",
      });
    },
  };
}

async function initGit(root: string): Promise<void> {
  await execFileAsync('git', ['init'], { cwd: root });
  await execFileAsync('git', ['config', 'user.email', 'tinycore@example.test'], { cwd: root });
  await execFileAsync('git', ['config', 'user.name', 'TinyCore Eval'], { cwd: root });
  await execFileAsync('git', ['add', '.'], { cwd: root });
  await execFileAsync('git', ['commit', '-m', 'initial'], { cwd: root });
}

function diffStats(diff: string): { filesTouched: number; linesChanged: number } {
  const files = new Set<string>();
  let linesChanged = 0;
  for (const line of diff.split(/\r?\n/)) {
    if (line.startsWith('+++ b/')) {
      files.add(line.slice('+++ b/'.length));
    } else if ((line.startsWith('+') && !line.startsWith('+++')) || (line.startsWith('-') && !line.startsWith('---'))) {
      linesChanged += 1;
    }
  }
  return { filesTouched: files.size, linesChanged };
}
