# Evaluation Spec

## Evaluation principle

TinyCore wins only if it improves the trade-off curve:

```txt
quality per stored byte
quality per watt/token
quality per VRAM MB
agent utility per stored byte
```

## Model evals

Minimum:

```txt
train loss
validation loss
perplexity
generation sample quality
stored unique bytes
effective materialized bytes
tokens/sec
peak memory
```

## Baselines

Always include:

```txt
baseline_transformer_v0
shared_layer_transformer_v0 if implemented
```

## Ablations

```txt
TinyCore basis only
TinyCore basis + low-rank
TinyCore basis + low-rank + recurrent mixer
TinyCore with different basis counts
TinyCore with different low-rank sizes
TinyCore with different virtual depth
```

## Agent evals later

Small repo tasks:

```txt
fix TypeScript type error
add unit test
rename symbol safely
implement small function from TODO
repair failing test
summarise repo module
```

Metrics:

```txt
task success
tests pass
typecheck pass
files touched
lines changed
invalid tool calls
hallucinated file paths
```

## Report format

Each run writes `report.json`:

```json
{
  "run_id": "...",
  "model_type": "tinycore_lora_v0",
  "baseline_run_id": "...",
  "dataset": "...",
  "metrics": {
    "val_loss": 0,
    "stored_unique_bytes": 0,
    "tokens_per_sec": 0
  },
  "conclusion": "better|worse|inconclusive",
  "notes": []
}
```
