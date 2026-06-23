# Phase 3 - Function Calling And MCP Tool-Use Training Runbook

## Objective

Build TinyCore-LM's third training corpus from function-calling and MCP tool-use
datasets, then train a larger RTX 5090 CUDA run that improves tool schema
selection, JSON argument generation, and MCP-style tool-call behavior.

Datasets:

- `Salesforce/xlam-function-calling-60k`
- `glaiveai/glaive-function-calling-v2`
- `MCPToolBench/MCPToolBenchPP`

## First Message To The AI On The 5090 Machine

Use this prompt exactly:

```text
You are preparing TinyCore-LM's third training round. From the repo root,
ingest the approved Hugging Face function-calling datasets into
data/training/function_calling_stage3_v0, preserve tool schemas and tool-call
records, skip secret-looking rows, write JSONL shards and a manifest, then run
the CUDA smoke and full function-calling stage-3 training config. Use only
public or explicitly accepted gated datasets. Do not reveal HF_TOKEN.
```

## Requirements

- Network access to Hugging Face.
- `HF_TOKEN` is required for gated xLAM raw-file downloads.
- The Hugging Face account behind `HF_TOKEN` must have accepted the
  `Salesforce/xlam-function-calling-60k` dataset terms.
- Public/gated dataset content only; no private datasets.

```bash
export HF_TOKEN=...
```

Optional helper dependency:

```bash
python3 -m pip install -e ".[dev,hf]"
```

The current ingestor uses the Hugging Face HTTP API directly, so the `hf` extra
is mainly for future dataset inspection/debugging.

## Discovery Dry Run

```bash
python3 tools/ingest_function_calling_stage3.py \
  --output-dir data/training/function_calling_stage3_v0 \
  --dry-run
```

Inspect:

```bash
sed -n '1,160p' data/training/function_calling_stage3_v0/manifest.json
python3 - <<'PY'
import json
plan=json.load(open("data/training/function_calling_stage3_v0/dataset_plan.json"))
for item in plan:
    print(item["dataset_id"], "gated=", item["gated"], "files=", len(item["files"]))
    for name in item["files"][:8]:
        print(" ", name)
PY
```

If xLAM fails with HTTP 401, accept the dataset terms on Hugging Face and retry
with `HF_TOKEN` set. Do not skip xLAM silently unless the user explicitly
chooses a non-gated control run.

## Full Ingestion

```bash
python3 tools/ingest_function_calling_stage3.py \
  --output-dir data/training/function_calling_stage3_v0 \
  --val-fraction 0.02 \
  --max-record-chars 24000
```

Useful controls:

```bash
# small smoke corpus
python3 tools/ingest_function_calling_stage3.py \
  --output-dir data/training/function_calling_stage3_v0 \
  --max-rows-per-dataset 1000 \
  --max-record-chars 24000

# keep raw downloaded JSON files for audit/debugging
python3 tools/ingest_function_calling_stage3.py \
  --output-dir data/training/function_calling_stage3_v0 \
  --keep-raw

# include MCPToolBench++ demo files as well as full files
python3 tools/ingest_function_calling_stage3.py \
  --output-dir data/training/function_calling_stage3_v0 \
  --include-demo-files
```

The ingestor writes:

```text
data/training/function_calling_stage3_v0/dataset_plan.json
data/training/function_calling_stage3_v0/train.jsonl
data/training/function_calling_stage3_v0/val.jsonl
data/training/function_calling_stage3_v0/eval_holdout.jsonl
data/training/function_calling_stage3_v0/manifest.json
```

## Ingestion Policy

The ingestor:

- pulls dataset metadata from the Hugging Face API;
- downloads JSON files from the dataset repositories;
- skips MCPToolBench++ files with `demo` in the basename by default;
- preserves Glaive `system` plus `chat` text directly;
- preserves xLAM and MCPToolBench++ records as canonical JSON so schemas,
  arguments, and tool names are not flattened away;
- truncates individual records at `--max-record-chars`;
- skips rows that look like private keys, API tokens, passwords, or secrets;
- splits train/validation deterministically by row id.

## Verify Corpus

```bash
python3 - <<'PY'
import json
from tinycore_model.data import make_token_splits, dataset_manifest

train, val, tok = make_token_splits(corpus_name="function_calling_stage3_v0", repeat=1)
manifest = dataset_manifest(train, val, tok, "function_calling_stage3_v0", repeat=1)
print("train_tokens", train.numel())
print("val_tokens", val.numel())
print(json.dumps(manifest, indent=2))
PY
```

Inspect source balance:

```bash
python3 - <<'PY'
import json
m=json.load(open("data/training/function_calling_stage3_v0/manifest.json"))
print("train_rows", m.get("train_rows"))
print("val_rows", m.get("val_rows"))
print("rows_by_dataset")
for dataset, rows in m.get("rows_by_dataset", {}).items():
    print(" ", dataset, rows)
print("num_skipped", m.get("num_skipped"))
for item in m.get("skipped", [])[:20]:
    print(item)
PY
```

## Training Config

Use:

```text
configs/function_calling_stage3_5090_tinycore.yaml
```

Important settings:

```yaml
device: cuda
dataset:
  name: function_calling_stage3_v0
model:
  d_model: 512
  n_heads: 8
  n_virtual_layers: 16
  max_seq_len: 768
  precision_target: bf16
training:
  batch_size: 16
  seq_len: 768
  max_steps: 40000
  eval_interval: 2000
eval:
  suite_name: function_calling_stage3_holdout_v0
```

## Smoke Run

```bash
python3 - <<'PY'
from dataclasses import replace
from tinycore_model import load_benchmark_config, run_benchmark

cfg = load_benchmark_config("configs/function_calling_stage3_5090_tinycore.yaml")
cfg = replace(
    cfg,
    run_group="function_calling_stage3_5090_smoke",
    training=replace(cfg.training, max_steps=20, eval_interval=10, batch_size=2),
)
report = run_benchmark(cfg, "reports/runs/function_calling_stage3_5090_smoke_report.json")
print(report["conclusion"])
for model in report["models"]:
    print(model["name"], model["val_loss"], model["stored_unique_bytes_bf16"])
PY
```

Do not start the full run if the smoke run fails.

## Full Run

```bash
python3 benchmarks/run_instruction_code_benchmark.py \
  --config configs/function_calling_stage3_5090_tinycore.yaml \
  --output reports/runs/function_calling_stage3_5090_report.json
```

Expected artifacts:

```text
reports/runs/function_calling_stage3_5090_report.json
reports/runs/function_calling_stage3_5090_tinycore/baseline_transformer_function_calling_v0/
reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0/
```

## If Out Of Memory

Apply one change at a time:

1. `training.batch_size: 8`
2. `training.batch_size: 4`
3. `training.seq_len: 512` and `model.max_seq_len: 512`
4. `model.d_model: 384`, `training.seq_len: 512`, and `model.max_seq_len: 512`

Run the smoke run after every change.

## Export TinyCore Artifact

```bash
python3 -m tinycore_format.cli export-tensors \
  reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0

python3 -m tinycore_format.cli convert \
  reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0 \
  reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0.tcmdl

python3 -m tinycore_format.cli verify \
  reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0.tcmdl
```

## What To Report Back

Report:

- timestamp and GPU details from `nvidia-smi`;
- whether `HF_TOKEN` was used, without revealing it;
- whether xLAM gated access succeeded;
- dataset files ingested and rows per dataset;
- train/val row counts and token counts;
- skipped row count and representative skip reasons;
- final report path;
- selected checkpoint step for each model;
- `val_loss`, `instruction_eval_mean_score`, `instruction_eval_passed`,
  `reference_completion_loss`, `instruction_eval_score_per_100kib_bf16`,
  and stored bytes;
- export/verify status for the TinyCore `.tcmdl`.
