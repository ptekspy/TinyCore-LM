# Codex Master Prompt

You are implementing TinyCore-LM: a research prototype for a new class of weight-efficient language model.

Do not build a standard GPT clone and call it new. The fundamental invariant is:

```txt
many computational steps
few unique stored weights
explicit weight composition
native low-bit readiness
retrieval/tool memory instead of memorising everything
```

## Required mental model

A normal decoder-only Transformer stores independent matrices per layer:

```txt
Layer 1: Wq, Wk, Wv, Wo, Wup, Wgate, Wdown
Layer 2: Wq, Wk, Wv, Wo, Wup, Wgate, Wdown
...
Layer N: Wq, Wk, Wv, Wo, Wup, Wgate, Wdown
```

TinyCore-LM stores:

```txt
shared basis matrices
+ tiny layer/depth routing coefficients
+ low-rank corrections
+ small norms/gates
+ optional expert dictionary
+ recurrent state parameters
```

Effective weights are composed at runtime or cached:

```txt
W_eff(layer, kind) = compose(basis[kind], route[layer, kind]) + low_rank_delta[layer, kind]
```

The architecture should allow more virtual depth than unique parameter depth:

```txt
unique block families: 4..12
virtual layers/steps: 24..96
```

## Primary research target

Prove or disprove this target:

```txt
At equal or lower stored-weight bytes, TinyCore-LM achieves better validation loss and/or better coding-task utility than a plain Transformer baseline.
```

## MVP requirements

MVP-0 repo scaffold:

- Python package can be installed locally.
- TypeScript packages compile.
- CI commands documented.
- Config schemas exist.

MVP-1 tiny training:

- Train a toy TinyCore model on a small corpus.
- Train a plain Transformer baseline on same corpus.
- Generate text from both.
- Report:
  - train loss
  - validation loss
  - tokens/sec
  - stored unique parameter bytes
  - estimated activation/KV memory
  - wall-clock time

MVP-2 architecture ablation:

- Compare:
  - full unique Transformer
  - shared-block Transformer
  - basis-composed TinyCore
  - basis + low-rank TinyCore
  - basis + low-rank + recurrent virtual depth TinyCore

MVP-3 agent shell:

- Existing model can be used behind the agent initially.
- Tool loop must support repo read/search/edit/test.
- Later swap in TinyCore model through local server.

## Implementation discipline

Use explicit configs. Do not hide decisions in code.

Every experimental run must save:

```txt
config
commit hash if available
dataset manifest
model manifest
metrics.json
loss curve data
sample generations
```

## Refusal conditions for Codex implementation choices

Reject any implementation that:

- stores full independent weights per virtual layer without composition
- lacks a baseline comparison
- cannot measure stored unique weight bytes
- cannot reproduce a training run from config
- silently changes architecture shapes
- claims breakthrough without eval evidence

## Style

Machine-first, dense, precise. Use TODO markers only when they identify real blocked work. Do not write motivational filler.
