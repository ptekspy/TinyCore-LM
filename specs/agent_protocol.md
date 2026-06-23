# Agent Protocol Spec

## Goal

Turn the model into a VSCode coding agent by giving it tools, repo context, patch application, and test feedback.

## Tool call schema

```json
{
  "id": "call_123",
  "type": "tool_call",
  "tool": "read_file",
  "args": {
    "path": "src/index.ts"
  }
}
```

## Tool result schema

```json
{
  "tool_call_id": "call_123",
  "type": "tool_result",
  "ok": true,
  "content": "file contents or structured data",
  "metadata": {}
}
```

## Required tools

```txt
list_files
read_file
write_file_via_patch
search_text
search_symbols
run_command
git_status
git_diff
run_tests
format_files
```

## Agent loop

```ts
while (steps < maxSteps) {
  const next = await model.step({messages, tools, repoContext, memory});
  if (next.type === 'final') return next;
  const result = await toolRegistry.run(next.tool, next.args);
  messages.push(asToolResult(result));
}
```

## Safety

- All file writes must go through patch validation.
- Never run destructive shell commands without explicit user approval.
- Default to read/search before edit.
- After edit, run targeted tests/typecheck if available.

## Patch flow

```txt
model proposes patch
  -> validate target file exists
  -> validate hunk applies
  -> apply patch
  -> run format/test
  -> return diff summary
```
