export type AgentMessage =
  | { role: 'system' | 'user' | 'assistant'; content: string }
  | { role: 'tool'; toolCallId: string; content: unknown };

export type ToolCall = {
  id: string;
  type: 'tool_call';
  tool: string;
  args: Record<string, unknown>;
};

export type ModelStep = ToolCall | { type: 'final'; content: string };

export interface ModelClient {
  step(input: {
    messages: AgentMessage[];
    tools: ToolDefinition[];
    repoContext: RepoContext;
    memory: Record<string, unknown>;
  }): Promise<ModelStep>;
}

export interface ToolRegistry {
  definitions(): ToolDefinition[];
  run(name: string, args: Record<string, unknown>): Promise<{ ok: boolean; content: unknown; metadata?: Record<string, unknown> }>;
}

export async function runAgentTask(input: {
  model: ModelClient;
  tools: ToolRegistry;
  messages: AgentMessage[];
  repoContext: RepoContext;
  maxSteps: number;
}) {
  const messages = [...input.messages];

  for (let step = 0; step < input.maxSteps; step++) {
    const next = await input.model.step({
      messages,
      tools: input.tools.definitions(),
      repoContext: input.repoContext,
      memory: {},
    });

    if (next.type === 'final') {
      return { ok: true, content: next.content, steps: step + 1 };
    }

    const result = await input.tools.run(next.tool, next.args);
    messages.push({ role: 'tool', toolCallId: next.id, content: result });
  }

  return { ok: false, content: 'Agent exceeded max steps.', steps: input.maxSteps };
}

type ToolDefinition = { name: string; description: string; inputSchema: unknown };
type RepoContext = { root: string; openFiles?: string[]; diagnostics?: unknown[]; gitDiff?: string };
