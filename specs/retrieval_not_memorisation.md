# Retrieval, Tools, and Not Memorising Everything

## Principle

TinyCore-LM should be a compact reasoning and tool-use model, not a giant memorised database.

For coding agents, facts should often live outside weights:

```txt
repo files
package docs
type definitions
local memory
search indexes
test output
terminal results
```

## External memory layers

```txt
repo index: files, symbols, imports, tests
project memory: conventions, decisions, architecture notes
session memory: current task steps, edits, errors
retrieval memory: docs/snippets keyed by embeddings/text search
```

## Why this matters for weight size

If the model has tools and retrieval, the weights can focus on:

```txt
planning
reasoning
code transformation
error repair
tool choice
language understanding
```

rather than memorising every package API.

## MVP

Use text search first. Vector search later.

Required search tools:

```txt
ripgrep-backed search_text
file tree listing
symbol extraction for TypeScript via tsserver/compiler API later
```
