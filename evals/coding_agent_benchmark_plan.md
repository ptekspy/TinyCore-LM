# Coding Agent Benchmark Plan

## Tiny benchmark repos

Create synthetic TypeScript repos with known tasks:

1. Failing unit test due to logic bug.
2. TypeScript type error.
3. Missing function implementation.
4. Rename exported function and update imports.
5. Add tests for utility.
6. Fix lint/format issue.

## Metrics

```txt
success: tests pass and requested behaviour correct
diff size: changed lines/files
tool calls: total and invalid
hallucination: references to non-existent files/functions
time: wall clock
```

## Purpose

Agent utility may reveal value even before base model competes on general language modelling.
