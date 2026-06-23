#!/usr/bin/env node
import { createTinycoredServer } from './index.js';
import { NativeModelRuntime } from './nativeRuntime.js';
import { PythonModelRuntime } from './pythonRuntime.js';

export type TinycoredCliOptions = {
  repoRoot: string;
  host: string;
  port: number;
  artifactDir?: string;
  tcmdl?: string;
  nativeBin?: string;
  pythonBin?: string;
};

export function parseCliArgs(argv: string[]): TinycoredCliOptions {
  const options: TinycoredCliOptions = {
    repoRoot: process.cwd(),
    host: '127.0.0.1',
    port: 8787,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--help' || arg === '-h') {
      throw new HelpRequested();
    }
    if (arg === '--repo-root') {
      options.repoRoot = requiredValue(argv, (index += 1), arg);
    } else if (arg === '--host') {
      options.host = requiredValue(argv, (index += 1), arg);
    } else if (arg === '--port') {
      options.port = parsePort(requiredValue(argv, (index += 1), arg));
    } else if (arg === '--artifact-dir') {
      options.artifactDir = requiredValue(argv, (index += 1), arg);
    } else if (arg === '--tcmdl') {
      options.tcmdl = requiredValue(argv, (index += 1), arg);
    } else if (arg === '--native-bin') {
      options.nativeBin = requiredValue(argv, (index += 1), arg);
    } else if (arg === '--python') {
      options.pythonBin = requiredValue(argv, (index += 1), arg);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

export async function startTinycored(options: TinycoredCliOptions): Promise<void> {
  if (options.artifactDir !== undefined && options.tcmdl !== undefined) {
    throw new Error('Choose only one model runtime: --artifact-dir or --tcmdl');
  }
  const modelRuntime =
    options.tcmdl !== undefined
      ? new NativeModelRuntime({ artifactPath: options.tcmdl, nativeBin: options.nativeBin })
      : options.artifactDir !== undefined
        ? new PythonModelRuntime({ artifactDir: options.artifactDir, pythonBin: options.pythonBin })
        : undefined;
  const server = createTinycoredServer({ repoRoot: options.repoRoot, modelRuntime });
  await new Promise<void>((resolve) => server.listen(options.port, options.host, resolve));
  const address = server.address();
  const actualPort = typeof address === 'object' && address !== null ? address.port : options.port;
  console.log(`tinycored listening on http://${options.host}:${actualPort}`);
  console.log(`repo_root=${options.repoRoot}`);
  console.log(`model_runtime=${modelRuntime?.name ?? 'none'}`);
}

function requiredValue(argv: string[], index: number, flag: string): string {
  const value = argv[index];
  if (value === undefined || value.startsWith('--')) {
    throw new Error(`Expected value after ${flag}`);
  }
  return value;
}

function parsePort(value: string): number {
  const port = Number(value);
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    throw new Error(`Invalid port: ${value}`);
  }
  return port;
}

function usage(): string {
  return [
    'usage: tinycored [--repo-root PATH] [--artifact-dir PATH | --tcmdl PATH] [--host HOST] [--port PORT] [--python PYTHON] [--native-bin PATH]',
    '',
    'Starts the local TinyCore server. If --artifact-dir is provided, /generate and message /chat use the Python checkpoint runtime. If --tcmdl is provided, they use the native runtime.',
  ].join('\n');
}

class HelpRequested extends Error {}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    await startTinycored(parseCliArgs(process.argv.slice(2)));
  } catch (error) {
    if (error instanceof HelpRequested) {
      console.log(usage());
      process.exit(0);
    }
    console.error(error instanceof Error ? error.message : String(error));
    console.error(usage());
    process.exit(1);
  }
}
