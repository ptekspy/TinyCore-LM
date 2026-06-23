export type AgentMessage =
  | { role: 'system' | 'user' | 'assistant'; content: string }
  | { role: 'tool'; toolCallId: string; content: unknown };

export type ToolCall = {
  id: string;
  type: 'tool_call';
  tool: string;
  args: Record<string, unknown>;
};

export type FinalStep = {
  type: 'final';
  content: string;
};

export type ModelStep = ToolCall | FinalStep;

export type ToolDefinition = {
  name: string;
  description: string;
  inputSchema: unknown;
};

export type ToolResult = {
  ok: boolean;
  content: unknown;
  metadata?: Record<string, unknown>;
};

export type RepoContext = {
  root: string;
  openFiles?: string[];
  diagnostics?: unknown[];
  gitDiff?: string;
};

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
  run(name: string, args: Record<string, unknown>): Promise<ToolResult>;
}

export type AgentTaskInput = {
  model: ModelClient;
  tools: ToolRegistry;
  messages: AgentMessage[];
  repoContext: RepoContext;
  maxSteps: number;
  memory?: Record<string, unknown>;
};

export type AgentTaskResult =
  | { ok: true; content: string; steps: number; messages: AgentMessage[] }
  | { ok: false; content: string; steps: number; messages: AgentMessage[] };

export async function runAgentTask(input: AgentTaskInput): Promise<AgentTaskResult> {
  const messages = [...input.messages];
  const memory = input.memory ?? {};

  for (let step = 0; step < input.maxSteps; step += 1) {
    const next = await input.model.step({
      messages,
      tools: input.tools.definitions(),
      repoContext: input.repoContext,
      memory,
    });

    if (next.type === 'final') {
      messages.push({ role: 'assistant', content: next.content });
      return { ok: true, content: next.content, steps: step + 1, messages };
    }

    const result = await input.tools.run(next.tool, next.args);
    messages.push({
      role: 'tool',
      toolCallId: next.id,
      content: {
        tool_call_id: next.id,
        type: 'tool_result',
        ok: result.ok,
        content: result.content,
        metadata: result.metadata ?? {},
      },
    });
  }

  return {
    ok: false,
    content: 'Agent exceeded max steps.',
    steps: input.maxSteps,
    messages,
  };
}

export { createRepoToolRegistry, RepoToolRegistry, type RepoToolOptions } from './repo_tools.js';
export {
  createSyntheticBugRepo,
  runSyntheticAgentBenchmark,
  runSyntheticAgentBenchmarkSuite,
  type AgentBenchmarkReport,
  type AgentBenchmarkSuiteReport,
} from './agent_benchmark.js';
