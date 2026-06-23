import { spawn } from 'node:child_process';

export type GenerateRequest = {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  top_k?: number;
  seed?: number;
};

export type GenerateResponse = {
  text: string;
  tokens: number[];
  metrics: Record<string, number>;
  runtime: string;
  model: Record<string, unknown>;
};

export interface ModelRuntime {
  readonly name: string;
  generate(request: GenerateRequest): Promise<GenerateResponse>;
}

export type PythonModelRuntimeOptions = {
  artifactDir: string;
  pythonBin?: string;
  timeoutMs?: number;
};

export class PythonModelRuntime implements ModelRuntime {
  readonly name = 'python';
  private readonly artifactDir: string;
  private readonly pythonBin: string;
  private readonly timeoutMs: number;

  constructor(options: PythonModelRuntimeOptions) {
    this.artifactDir = options.artifactDir;
    this.pythonBin = options.pythonBin ?? 'python3';
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const payload = {
      artifact_dir: this.artifactDir,
      prompt: request.prompt,
      max_tokens: request.max_tokens ?? 64,
      temperature: request.temperature ?? 0.8,
      seed: request.seed ?? 1337,
    };
    return runJsonProcess(this.pythonBin, ['-m', 'tinycore_model.generate_cli'], payload, this.timeoutMs);
  }
}

function runJsonProcess(
  command: string,
  args: string[],
  payload: Record<string, unknown>,
  timeoutMs: number,
): Promise<GenerateResponse> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['pipe', 'pipe', 'pipe'] });
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      reject(new Error(`model runtime timed out after ${timeoutMs}ms`));
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
        reject(new Error(Buffer.concat(stderr).toString('utf8') || `model runtime exited with ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(Buffer.concat(stdout).toString('utf8')) as GenerateResponse);
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(JSON.stringify(payload));
  });
}
