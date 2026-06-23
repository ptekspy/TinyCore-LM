import { spawn } from 'node:child_process';
import type { GenerateRequest, GenerateResponse, ModelRuntime } from './pythonRuntime.js';

export type NativeModelRuntimeOptions = {
  artifactPath: string;
  nativeBin?: string;
  timeoutMs?: number;
};

type NativeGenerateResponse = GenerateResponse & {
  generation?: {
    temperature?: number;
    top_k?: number;
    seed?: number;
  };
};

export class NativeModelRuntime implements ModelRuntime {
  readonly name = 'native';
  private readonly artifactPath: string;
  private readonly nativeBin: string;
  private readonly timeoutMs: number;

  constructor(options: NativeModelRuntimeOptions) {
    this.artifactPath = options.artifactPath;
    this.nativeBin = options.nativeBin ?? 'runtime/tinycore.cpp/build/tinycore-generate';
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const maxTokens = request.max_tokens ?? 64;
    const temperature = request.temperature ?? 0;
    const topK = request.top_k ?? 0;
    const seed = request.seed ?? 1337;
    const response = await runNativeJsonProcess(
      this.nativeBin,
      [this.artifactPath, request.prompt, String(maxTokens), String(temperature), String(topK), String(seed)],
      this.timeoutMs,
    );
    return {
      ...response,
      metrics: response.metrics ?? {},
    };
  }
}

function runNativeJsonProcess(command: string, args: string[], timeoutMs: number): Promise<NativeGenerateResponse> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      reject(new Error(`native model runtime timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    child.stdout.on('data', (chunk: Buffer) => stdout.push(chunk));
    child.stderr.on('data', (chunk: Buffer) => stderr.push(chunk));
    child.on('error', (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(Buffer.concat(stderr).toString('utf8') || `native model runtime exited with ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(Buffer.concat(stdout).toString('utf8')) as NativeGenerateResponse);
      } catch (error) {
        reject(error);
      }
    });
  });
}
