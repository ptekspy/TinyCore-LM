export type TinycoredHealth = {
  ok: boolean;
  model_loaded: boolean;
  runtime: string;
};

export type TinycoredToolResult = {
  ok: boolean;
  content: unknown;
  metadata?: Record<string, unknown>;
  error?: string;
};

export type TinycoredGenerateResponse = {
  text: string;
  tokens: number[];
  metrics: Record<string, number>;
  generation?: {
    temperature?: number;
    top_k?: number;
    seed?: number;
  };
  runtime: string;
  model: Record<string, unknown>;
};

export type TinycoredChatMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
};

export type TinycoredChatResponse =
  | {
      type: 'message';
      role: 'assistant';
      content: string;
      metrics?: Record<string, number>;
      runtime?: string;
      model?: Record<string, unknown>;
    }
  | {
      type: 'tool_result';
      tool_call_id: string;
      ok: boolean;
      content: unknown;
      metadata?: Record<string, unknown>;
    };

export class TinycoredClient {
  constructor(private readonly baseUrl: string) {}

  async health(): Promise<TinycoredHealth> {
    return this.request<TinycoredHealth>('GET', '/health');
  }

  async agentStep(tool: string, args: Record<string, unknown>): Promise<TinycoredToolResult> {
    return this.request<TinycoredToolResult>('POST', '/agent/step', { tool, args });
  }

  async generate(input: {
    prompt: string;
    max_tokens?: number;
    temperature?: number;
    top_k?: number;
    seed?: number;
  }): Promise<TinycoredGenerateResponse> {
    return this.request<TinycoredGenerateResponse>('POST', '/generate', input);
  }

  async chat(input: {
    prompt?: string;
    messages?: TinycoredChatMessage[];
    tool_call?: { id?: string; tool: string; args?: Record<string, unknown> };
    max_tokens?: number;
    temperature?: number;
    top_k?: number;
    seed?: number;
  }): Promise<TinycoredChatResponse> {
    return this.request<TinycoredChatResponse>('POST', '/chat', input);
  }

  private async request<T>(method: 'GET' | 'POST', pathname: string, body?: unknown): Promise<T> {
    const response = await fetch(`${this.baseUrl}${pathname}`, {
      method,
      headers: body === undefined ? undefined : { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const payload = (await response.json()) as unknown;
    if (!response.ok) {
      const message = typeof payload === 'object' && payload !== null && 'error' in payload
        ? String((payload as { error: unknown }).error)
        : `tinycored request failed with ${response.status}`;
      throw new Error(message);
    }
    return payload as T;
  }
}
