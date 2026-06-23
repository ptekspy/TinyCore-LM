# TinyCore-LM Codex Pack

Codex: read this file first, then `01_CODEX_MASTER_PROMPT.md`, then `02_PROJECT_MANIFEST.yaml`, then all files under `specs/`, then the task graph under `tasks/task_dag.json`.

Human-readable polish is irrelevant. Correctness, consistency, implementation sequence, testability, and explicit unknowns matter.

## Mission

Build a research-grade prototype of a new weight-efficient LLM architecture. The core hypothesis is:

> Current LLMs store too many independent layer weights. A model can achieve useful capability with dramatically fewer stored unique weights by using shared low-bit basis weights, layer-specific routing coefficients, low-rank corrections, recurrent virtual depth, and retrieval/tool memory instead of memorising everything in parameters.

This is not a normal model compression project. Do not merely quantise an existing Transformer after training. The architecture must be designed so the trained model natively contains fewer stored unique weights.

## Project names

Canonical name: `TinyCore-LM`

Related components:

- `TinyCore`: model architecture
- `TCMDL`: TinyCore Model file format
- `tinycore.py`: PyTorch research implementation
- `tinycore.cpp`: native inference runtime target
- `tinycored`: local model/agent server
- `TinyCore Code`: VSCode extension / coding agent

## Build philosophy

1. Start with the smallest complete loop that can train and generate.
2. Always compare against a plain Transformer baseline of similar compute and similar stored bytes.
3. Treat weight size, memory usage, throughput, and quality as first-class metrics.
4. Do not claim success until the evals prove it.
5. Keep all interfaces explicit so future Codex runs can continue from any phase.

## Initial deliverable

Create a monorepo with:

```txt
packages/
  tinycore-model/          # PyTorch model, training, generation
  tinycore-data/           # tokenizer/data pipeline
  tinycore-eval/           # model + agent evals
  tinycore-format/         # TCMDL tools/spec implementation
  tinycored/               # local server
  tinycore-code/           # VSCode extension
  tinycore-agent/          # tool loop, repo index, patch engine
runtime/
  tinycore.cpp/            # later native runtime
specs/
  mirrored copies of these spec docs
```

Preferred languages:

- PyTorch/Python for model research and training.
- TypeScript for server, VSCode extension, agent orchestration, repo indexer, and tooling.
- C++ or Rust for native runtime later; C++ is acceptable if aiming for llama.cpp-like ecosystem familiarity.

## Absolute non-negotiable

The first working model can be terrible. It cannot be fake. It must train, produce tokens, report stored unique weight bytes, and run against a baseline.
