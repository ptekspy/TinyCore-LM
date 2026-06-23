import test from 'node:test';
import assert from 'node:assert/strict';
import { runSyntheticAgentBenchmark, runSyntheticAgentBenchmarkSuite } from '../src/index.js';

test('synthetic agent benchmark fixes failing arithmetic test', async () => {
  const report = await runSyntheticAgentBenchmark();

  assert.equal(report.run_id, 'synthetic_bugfix_v0');
  assert.equal(report.success, true);
  assert.equal(report.tests_passed_before_patch, false);
  assert.equal(report.tests_passed_after_patch, true);
  assert.equal(report.invalid_tool_calls, 0);
  assert.equal(report.files_touched, 1);
  assert.equal(report.lines_changed, 2);
  assert.equal(report.hallucinated_file_rate, 0);
  assert.ok(report.tool_calls >= 5);
});

test('synthetic agent benchmark suite aggregates multiple tasks', async () => {
  const report = await runSyntheticAgentBenchmarkSuite();

  assert.equal(report.run_id, 'synthetic_agent_suite_v0');
  assert.equal(report.tasks_total, 2);
  assert.equal(report.tasks_succeeded, 2);
  assert.equal(report.task_success_rate, 1);
  assert.equal(report.total_invalid_tool_calls, 0);
  assert.equal(report.total_files_touched, 2);
  assert.equal(report.total_lines_changed, 4);
  assert.deepEqual(report.tasks.map((task) => task.run_id), [
    'synthetic_bugfix_v0',
    'synthetic_missing_function_v0',
  ]);
});
