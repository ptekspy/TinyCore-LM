# ADR-0005: Retrieval Not Memorisation

## Decision

For agent use, TinyCore should rely on tools/retrieval for volatile or project-specific knowledge rather than memorising everything in model weights.

## Rationale

A coding model does not need every library API in parameters if it can inspect local docs, types, source files, and test output.

## Consequences

- Agent tooling is not optional long-term.
- Repo indexing is part of the product stack.
- Model weights should focus on reasoning/tool-use behaviour.
