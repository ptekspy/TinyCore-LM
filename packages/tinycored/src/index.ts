import http, { type IncomingMessage, type ServerResponse } from 'node:http';
import { createRepoToolRegistry } from '@tinycore/agent';
import type { GenerateRequest, ModelRuntime } from './pythonRuntime.js';
export { PythonModelRuntime, type GenerateRequest, type GenerateResponse, type ModelRuntime } from './pythonRuntime.js';
export { NativeModelRuntime } from './nativeRuntime.js';

export type TinycoredOptions = {
  repoRoot: string;
  modelRuntime?: ModelRuntime;
};

type JsonValue = Record<string, unknown> | unknown[] | string | number | boolean | null;
type ChatMessage = { role: 'system' | 'user' | 'assistant' | 'tool'; content: string };
type ChatToolCall = { id?: string; tool: string; args?: Record<string, unknown> };

export function createTinycoredServer(options: TinycoredOptions): http.Server {
  const tools = createRepoToolRegistry({ root: options.repoRoot });
  const modelRuntime = options.modelRuntime;

  return http.createServer(async (req, res) => {
    try {
      if (req.method === 'GET' && req.url === '/health') {
        writeJson(res, 200, {
          ok: true,
          model_loaded: modelRuntime !== undefined,
          runtime: modelRuntime?.name ?? 'typescript-mvp',
        });
        return;
      }

      if (req.method === 'POST' && req.url === '/agent/step') {
        const body = await readJson(req);
        const tool = requiredString(body.tool, 'tool');
        const args = isRecord(body.args) ? body.args : {};
        const result = await tools.run(tool, args);
        writeJson(res, result.ok ? 200 : 400, result as unknown as JsonValue);
        return;
      }

      if (req.method === 'POST' && req.url === '/generate') {
        if (modelRuntime === undefined) {
          writeJson(res, 501, {
            ok: false,
            error: 'Model runtime is not wired yet. Use /agent/step for repo tools.',
          });
          return;
        }
        const body = await readJson(req);
        const response = await modelRuntime.generate(parseGenerateRequest(body));
        writeJson(res, 200, response as unknown as JsonValue);
        return;
      }

      if (req.method === 'POST' && req.url === '/chat') {
        const body = await readJson(req);
        if (isRecord(body.tool_call)) {
          const toolCall = parseChatToolCall(body.tool_call);
          const result = await tools.run(toolCall.tool, toolCall.args ?? {});
          writeJson(res, result.ok ? 200 : 400, {
            type: 'tool_result',
            tool_call_id: toolCall.id ?? 'chat_tool_call',
            ok: result.ok,
            content: result.content,
            metadata: result.metadata ?? {},
          });
          return;
        }
        if (modelRuntime === undefined) {
          writeJson(res, 501, {
            ok: false,
            error: 'Chat runtime requires a configured model runtime or tool_call.',
          });
          return;
        }
        const prompt =
          typeof body.prompt === 'string' && body.prompt.length > 0
            ? body.prompt
            : chatPrompt(parseChatMessages(body.messages));
        const response = await modelRuntime.generate({
          prompt,
          max_tokens: optionalNumber(body.max_tokens, 'max_tokens') ?? 128,
          temperature: optionalNumber(body.temperature, 'temperature') ?? 0.8,
          top_k: optionalNumber(body.top_k, 'top_k'),
          seed: optionalNumber(body.seed, 'seed'),
        });
        writeJson(res, 200, {
          type: 'message',
          role: 'assistant',
          content: response.text.startsWith(prompt) ? response.text.slice(prompt.length) : response.text,
          metrics: response.metrics,
          runtime: response.runtime,
          model: response.model,
        });
        return;
      }

      writeJson(res, 404, { ok: false, error: 'Not found' });
    } catch (error) {
      writeJson(res, 400, {
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
}

function writeJson(res: ServerResponse, statusCode: number, value: JsonValue): void {
  res.writeHead(statusCode, { 'content-type': 'application/json; charset=utf-8' });
  res.end(`${JSON.stringify(value)}\n`);
}

async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  if (chunks.length === 0) {
    return {};
  }
  const parsed = JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
  if (!isRecord(parsed)) {
    throw new Error('Expected JSON object body');
  }
  return parsed;
}

function requiredString(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`Expected non-empty string: ${name}`);
  }
  return value;
}

function parseGenerateRequest(body: Record<string, unknown>): GenerateRequest {
  return {
    prompt: requiredString(body.prompt, 'prompt'),
    max_tokens: optionalNumber(body.max_tokens, 'max_tokens'),
    temperature: optionalNumber(body.temperature, 'temperature'),
    top_k: optionalNumber(body.top_k, 'top_k'),
    seed: optionalNumber(body.seed, 'seed'),
  };
}

function parseChatToolCall(value: Record<string, unknown>): ChatToolCall {
  return {
    id: typeof value.id === 'string' ? value.id : undefined,
    tool: requiredString(value.tool, 'tool_call.tool'),
    args: isRecord(value.args) ? value.args : {},
  };
}

function parseChatMessages(value: unknown): ChatMessage[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error('Expected non-empty messages array');
  }
  return value.map((item, index) => {
    if (!isRecord(item)) {
      throw new Error(`messages[${index}] must be an object`);
    }
    const role = item.role;
    if (role !== 'system' && role !== 'user' && role !== 'assistant' && role !== 'tool') {
      throw new Error(`messages[${index}].role is invalid`);
    }
    return { role, content: requiredString(item.content, `messages[${index}].content`) };
  });
}

function chatPrompt(messages: ChatMessage[]): string {
  return messages.map((message) => `<${message.role}>\n${message.content}`).join('\n\n') + '\n\n<assistant>\n';
}

function optionalNumber(value: unknown, name: string): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Expected number: ${name}`);
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
