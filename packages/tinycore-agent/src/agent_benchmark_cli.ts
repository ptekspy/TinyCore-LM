#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { runSyntheticAgentBenchmark, runSyntheticAgentBenchmarkSuite } from './agent_benchmark.js';

export async function main(argv: string[] = process.argv.slice(2)): Promise<number> {
  let output = 'reports/runs/agent_eval_report.json';
  let keepRepo = false;
  let suite = false;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--output') {
      output = requiredValue(argv, (index += 1), arg);
    } else if (arg === '--keep-repo') {
      keepRepo = true;
    } else if (arg === '--suite') {
      suite = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  const report = suite
    ? await runSyntheticAgentBenchmarkSuite({ keepRepos: keepRepo })
    : await runSyntheticAgentBenchmark({ keepRepo });
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report, null, 2));
  return ("success" in report ? report.success : report.task_success_rate === 1) ? 0 : 1;
}

function requiredValue(argv: string[], index: number, flag: string): string {
  const value = argv[index];
  if (value === undefined || value.startsWith('--')) {
    throw new Error(`Expected value after ${flag}`);
  }
  return value;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    process.exitCode = await main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
