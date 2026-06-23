import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import type { ToolDefinition, ToolRegistry, ToolResult } from './index.js';

const execFileAsync = promisify(execFile);

const DEFAULT_IGNORES = new Set([
  '.git',
  '.pytest_cache',
  '__pycache__',
  'node_modules',
  'dist',
  '.mypy_cache',
  '.ruff_cache',
]);

const DESTRUCTIVE_COMMANDS = new Set([
  'rm',
  'rmdir',
  'mv',
  'cp',
  'chmod',
  'chown',
  'git-reset',
  'git-checkout',
  'git-clean',
]);

export type RepoToolOptions = {
  root: string;
  maxReadBytes?: number;
  maxSearchResults?: number;
  commandTimeoutMs?: number;
  allowDestructiveCommands?: boolean;
  extraIgnores?: string[];
};

type ToolHandler = (args: Record<string, unknown>) => Promise<ToolResult>;

export class RepoToolRegistry implements ToolRegistry {
  private readonly root: string;
  private readonly maxReadBytes: number;
  private readonly maxSearchResults: number;
  private readonly commandTimeoutMs: number;
  private readonly allowDestructiveCommands: boolean;
  private readonly ignores: Set<string>;
  private readonly handlers: Map<string, ToolHandler>;

  constructor(options: RepoToolOptions) {
    this.root = path.resolve(options.root);
    this.maxReadBytes = options.maxReadBytes ?? 256_000;
    this.maxSearchResults = options.maxSearchResults ?? 100;
    this.commandTimeoutMs = options.commandTimeoutMs ?? 20_000;
    this.allowDestructiveCommands = options.allowDestructiveCommands ?? false;
    this.ignores = new Set([...DEFAULT_IGNORES, ...(options.extraIgnores ?? [])]);
    this.handlers = new Map<string, ToolHandler>([
      ['list_files', this.listFiles.bind(this)],
      ['read_file', this.readFile.bind(this)],
      ['search_text', this.searchText.bind(this)],
      ['search_symbols', this.searchSymbols.bind(this)],
      ['write_file_via_patch', this.writeFileViaPatch.bind(this)],
      ['git_status', this.gitStatus.bind(this)],
      ['git_diff', this.gitDiff.bind(this)],
      ['run_command', this.runCommand.bind(this)],
      ['run_tests', this.runTests.bind(this)],
      ['format_files', this.formatFiles.bind(this)],
    ]);
  }

  definitions(): ToolDefinition[] {
    return [
      {
        name: 'list_files',
        description: 'List files under the repository root.',
        inputSchema: { type: 'object', properties: { directory: { type: 'string' } } },
      },
      {
        name: 'read_file',
        description: 'Read a UTF-8 file from the repository root.',
        inputSchema: { type: 'object', required: ['path'], properties: { path: { type: 'string' } } },
      },
      {
        name: 'search_text',
        description: 'Search repository files for a literal text query.',
        inputSchema: { type: 'object', required: ['query'], properties: { query: { type: 'string' } } },
      },
      {
        name: 'search_symbols',
        description: 'Search TypeScript and Python files for simple class/function/const symbols.',
        inputSchema: { type: 'object', properties: { query: { type: 'string' } } },
      },
      {
        name: 'write_file_via_patch',
        description: 'Apply a validated single-file text replacement patch.',
        inputSchema: {
          type: 'object',
          required: ['path', 'oldText', 'newText'],
          properties: {
            path: { type: 'string' },
            oldText: { type: 'string' },
            newText: { type: 'string' },
          },
        },
      },
      {
        name: 'git_status',
        description: 'Return git status --short for the repository.',
        inputSchema: { type: 'object', properties: {} },
      },
      {
        name: 'git_diff',
        description: 'Return git diff for the repository.',
        inputSchema: { type: 'object', properties: { staged: { type: 'boolean' } } },
      },
      {
        name: 'run_command',
        description: 'Run a non-destructive command in the repository.',
        inputSchema: {
          type: 'object',
          required: ['command'],
          properties: { command: { type: 'array', items: { type: 'string' } } },
        },
      },
      {
        name: 'run_tests',
        description: 'Run the repository test command.',
        inputSchema: { type: 'object', properties: { command: { type: 'array', items: { type: 'string' } } } },
      },
      {
        name: 'format_files',
        description: 'Run a repository format command when provided.',
        inputSchema: { type: 'object', properties: { command: { type: 'array', items: { type: 'string' } } } },
      },
    ];
  }

  async run(name: string, args: Record<string, unknown>): Promise<ToolResult> {
    const handler = this.handlers.get(name);
    if (!handler) {
      return { ok: false, content: `Unknown tool: ${name}` };
    }
    try {
      return await handler(args);
    } catch (error) {
      return {
        ok: false,
        content: error instanceof Error ? error.message : String(error),
      };
    }
  }

  private async listFiles(args: Record<string, unknown>): Promise<ToolResult> {
    const directory = this.resolveInside(stringArg(args.directory, '.'));
    const files: string[] = [];
    await this.walk(directory, files);
    return { ok: true, content: files.sort(), metadata: { count: files.length } };
  }

  private async readFile(args: Record<string, unknown>): Promise<ToolResult> {
    const target = this.resolveInside(requiredStringArg(args.path, 'path'));
    const stat = await fs.stat(target);
    if (!stat.isFile()) {
      return { ok: false, content: 'Path is not a file' };
    }
    if (stat.size > this.maxReadBytes) {
      return { ok: false, content: `File exceeds maxReadBytes (${this.maxReadBytes})` };
    }
    return { ok: true, content: await fs.readFile(target, 'utf8'), metadata: { bytes: stat.size } };
  }

  private async searchText(args: Record<string, unknown>): Promise<ToolResult> {
    const query = requiredStringArg(args.query, 'query');
    const files: string[] = [];
    await this.walk(this.root, files);
    const matches: Array<{ path: string; line: number; text: string }> = [];
    for (const file of files) {
      if (matches.length >= this.maxSearchResults) {
        break;
      }
      const fullPath = this.resolveInside(file);
      const stat = await fs.stat(fullPath);
      if (stat.size > this.maxReadBytes) {
        continue;
      }
      const content = await fs.readFile(fullPath, 'utf8');
      const lines = content.split(/\r?\n/);
      lines.forEach((text, index) => {
        if (matches.length < this.maxSearchResults && text.includes(query)) {
          matches.push({ path: file, line: index + 1, text });
        }
      });
    }
    return { ok: true, content: matches, metadata: { count: matches.length } };
  }

  private async searchSymbols(args: Record<string, unknown>): Promise<ToolResult> {
    const query = stringArg(args.query, '');
    const files: string[] = [];
    await this.walk(this.root, files);
    const matches: Array<{ path: string; line: number; kind: string; name: string; text: string }> = [];
    const symbolPattern =
      /^\s*(?:export\s+)?(?:(class|function|interface|type)\s+([A-Za-z_$][\w$]*)|(?:const|let|var)\s+([A-Za-z_$][\w$]*)|def\s+([A-Za-z_]\w*)|class\s+([A-Za-z_]\w*))/;
    for (const file of files) {
      if (!/\.(ts|tsx|js|jsx|py)$/.test(file)) {
        continue;
      }
      const fullPath = this.resolveInside(file);
      const stat = await fs.stat(fullPath);
      if (stat.size > this.maxReadBytes) {
        continue;
      }
      const lines = (await fs.readFile(fullPath, 'utf8')).split(/\r?\n/);
      lines.forEach((text, index) => {
        const match = symbolPattern.exec(text);
        if (!match) {
          return;
        }
        const kind = match[1] ?? (text.trimStart().startsWith('def ') ? 'function' : 'value');
        const name = match[2] ?? match[3] ?? match[4] ?? match[5] ?? '';
        if (query === '' || name.includes(query)) {
          matches.push({ path: file, line: index + 1, kind, name, text });
        }
      });
    }
    return { ok: true, content: matches.slice(0, this.maxSearchResults), metadata: { count: matches.length } };
  }

  private async writeFileViaPatch(args: Record<string, unknown>): Promise<ToolResult> {
    const relativePath = requiredStringArg(args.path, 'path');
    const oldText = requiredStringArg(args.oldText, 'oldText');
    const newText = typeof args.newText === 'string' ? args.newText : undefined;
    if (newText === undefined) {
      throw new Error('Expected string arg: newText');
    }
    const target = this.resolveInside(relativePath);
    const stat = await fs.stat(target);
    if (!stat.isFile()) {
      return { ok: false, content: 'Path is not a file' };
    }
    if (stat.size > this.maxReadBytes) {
      return { ok: false, content: `File exceeds maxReadBytes (${this.maxReadBytes})` };
    }
    const original = await fs.readFile(target, 'utf8');
    const first = original.indexOf(oldText);
    if (first === -1) {
      return { ok: false, content: 'Patch oldText was not found' };
    }
    if (original.indexOf(oldText, first + oldText.length) !== -1) {
      return { ok: false, content: 'Patch oldText matched more than once' };
    }
    const updated = original.slice(0, first) + newText + original.slice(first + oldText.length);
    await fs.writeFile(target, updated, 'utf8');
    return {
      ok: true,
      content: {
        path: relativePath,
        replacedBytes: Buffer.byteLength(oldText),
        insertedBytes: Buffer.byteLength(newText),
      },
    };
  }

  private async gitStatus(): Promise<ToolResult> {
    return this.exec(['git', 'status', '--short']);
  }

  private async gitDiff(args: Record<string, unknown>): Promise<ToolResult> {
    const command = boolArg(args.staged, false) ? ['git', 'diff', '--staged'] : ['git', 'diff'];
    return this.exec(command);
  }

  private async runCommand(args: Record<string, unknown>): Promise<ToolResult> {
    const command = commandArg(args.command);
    this.validateCommand(command);
    return this.exec(command);
  }

  private async runTests(args: Record<string, unknown>): Promise<ToolResult> {
    const command = args.command === undefined ? ['npm', 'test'] : commandArg(args.command);
    this.validateCommand(command);
    return this.exec(command);
  }

  private async formatFiles(args: Record<string, unknown>): Promise<ToolResult> {
    if (args.command === undefined) {
      return { ok: false, content: 'No format command configured. Provide command: string[].' };
    }
    const command = commandArg(args.command);
    this.validateCommand(command);
    return this.exec(command);
  }

  private async walk(directory: string, files: string[]): Promise<void> {
    const entries = await fs.readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      if (this.ignores.has(entry.name)) {
        continue;
      }
      const fullPath = path.join(directory, entry.name);
      const relative = path.relative(this.root, fullPath);
      if (entry.isDirectory()) {
        await this.walk(fullPath, files);
      } else if (entry.isFile()) {
        files.push(relative);
      }
    }
  }

  private resolveInside(relativePath: string): string {
    const resolved = path.resolve(this.root, relativePath);
    const rel = path.relative(this.root, resolved);
    if (rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel))) {
      return resolved;
    }
    throw new Error(`Path escapes repository root: ${relativePath}`);
  }

  private validateCommand(command: string[]): void {
    if (command.length === 0) {
      throw new Error('Command cannot be empty');
    }
    const normalized = command[0] === 'git' && command[1] ? `git-${command[1]}` : command[0];
    if (!this.allowDestructiveCommands && DESTRUCTIVE_COMMANDS.has(normalized)) {
      throw new Error(`Refusing destructive command: ${command.join(' ')}`);
    }
  }

  private async exec(command: string[]): Promise<ToolResult> {
    const [file, ...args] = command;
    const result = await execFileAsync(file, args, {
      cwd: this.root,
      timeout: this.commandTimeoutMs,
      maxBuffer: this.maxReadBytes,
    });
    return {
      ok: true,
      content: {
        stdout: result.stdout,
        stderr: result.stderr,
      },
      metadata: { command },
    };
  }
}

export function createRepoToolRegistry(options: RepoToolOptions): RepoToolRegistry {
  return new RepoToolRegistry(options);
}

function requiredStringArg(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`Expected non-empty string arg: ${name}`);
  }
  return value;
}

function stringArg(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.length > 0 ? value : fallback;
}

function boolArg(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function commandArg(value: unknown): string[] {
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    throw new Error('Expected command as string[]');
  }
  return value;
}
