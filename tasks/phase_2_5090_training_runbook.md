# Phase 2 - RTX 5090 Laptop Training Runbook

## Objective

Train the larger TinyCore instruction/code model on the RTX 5090 laptop using
the generated `tinycore_instruction_code_5090_v0` corpus, then export the best
TinyCore checkpoint for native runtime testing.

## Hardware Assumptions

- Target GPU: RTX 5090 Laptop GPU class.
- Treat VRAM as machine-specific. Many RTX 5090 laptops advertise 24 GB GDDR7,
  but confirm the actual machine with `nvidia-smi`.
- Start from the provided config. If out-of-memory happens, reduce batch size
  before changing model dimensions.

## First Message To The AI On The 5090 Machine

Use this prompt exactly:

```text
You are training TinyCore-LM on this RTX 5090 laptop. Work from the repo root.
Do not use internet data. Use only the generated local corpus
data/training/instruction_code_5090_v0. First verify CUDA and the dataset
manifest, then run a short smoke benchmark, then run the full 5090 config. Keep
all commands and outputs in your final report. If CUDA is unavailable or the run
OOMs, stop, report the exact error, and suggest the smallest config reduction.
```

## Setup

Run from the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install PyTorch using the command from the official PyTorch local install page
for this machine's driver/CUDA combination:

```bash
python3 -m pip install -e ".[dev]"
npm install
```

Verify CUDA:

```bash
nvidia-smi
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

If `cuda_available` is `False`, do not start training. Fix the PyTorch/CUDA
install first.

## Dataset

The corpus is already generated here:

```text
data/training/instruction_code_5090_v0/
```

Expected manifest:

```text
train_examples: 60000
val_examples: 6000
num_documents: 66000
```

Regenerate only if the files are missing or hashes do not match:

```bash
python3 tools/generate_large_instruction_corpus.py \
  --output-dir data/training/instruction_code_5090_v0 \
  --train-examples 60000 \
  --val-examples 6000
```

Verify loader token counts:

```bash
python3 - <<'PY'
from tinycore_model.data import make_token_splits
train, val, _ = make_token_splits(corpus_name="tinycore_instruction_code_5090_v0", repeat=1)
print("train_tokens", train.numel())
print("val_tokens", val.numel())
PY
```

Expected approximate counts:

```text
train_tokens: 7.2M
val_tokens: 0.72M
```

## Config

Use:

```text
configs/instruction_code_5090_tinycore.yaml
```

Important settings:

```yaml
device: cuda
dataset:
  name: tinycore_instruction_code_5090_v0
  repeat: 1
model:
  d_model: 384
  n_heads: 8
  n_layers: 8
  n_virtual_layers: 12
  basis_rank: 8
  low_rank: 0
  recurrent_state_dim: 64
  max_seq_len: 256
  precision_target: bf16
training:
  batch_size: 64
  seq_len: 256
  max_steps: 20000
  eval_interval: 1000
  select_best_eval_checkpoint: true
eval:
  suite_name: instruction_code_5090_holdout_v0
```

The benchmark loop uses bf16 autocast on CUDA when `precision_target: bf16`.

## Smoke Run

Before the full run, execute a tiny CUDA smoke run:

```bash
python3 - <<'PY'
from dataclasses import replace
from tinycore_model import load_benchmark_config, run_benchmark

cfg = load_benchmark_config("configs/instruction_code_5090_tinycore.yaml")
cfg = replace(
    cfg,
    run_group="instruction_code_5090_smoke",
    training=replace(cfg.training, max_steps=20, eval_interval=10, batch_size=8),
)
report = run_benchmark(cfg, "reports/runs/instruction_code_5090_smoke_report.json")
print(report["conclusion"])
for model in report["models"]:
    print(model["name"], model["val_loss"], model["stored_unique_bytes_bf16"])
PY
```

If this fails, do not start the full run.

## Full Run

Start monitoring:

```bash
watch -n 2 nvidia-smi
```

In another terminal:

```bash
python3 benchmarks/run_instruction_code_benchmark.py \
  --config configs/instruction_code_5090_tinycore.yaml \
  --output reports/runs/instruction_code_5090_report.json
```

Expected outputs:

```text
reports/runs/instruction_code_5090_report.json
reports/runs/instruction_code_5090_tinycore/baseline_transformer_5090_v0/
reports/runs/instruction_code_5090_tinycore/tinycore_recurrent_5090_v0/
```

## If Out Of Memory

Apply one change at a time:

1. Set `training.batch_size: 32`.
2. If still OOM, set `training.batch_size: 16`.
3. If still OOM, set `training.seq_len: 192` and `model.max_seq_len: 192`.
4. If still OOM, set `model.d_model: 256`, `n_heads: 8`, `basis_rank: 6`,
   `low_rank: 12`, and keep `n_virtual_layers: 12`.

After each change, rerun the smoke run first.

## What To Report

After the full run, report:

- GPU name, VRAM, driver, CUDA, PyTorch version.
- Exact config file and any edits.
- Wall clock time.
- Final and selected checkpoint step for each model.
- `val_loss`.
- `instruction_eval_mean_score`.
- `instruction_eval_passed`.
- `reference_completion_loss`.
- `instruction_eval_score_per_100kib_bf16`.
- `stored_unique_bytes_bf16`.
- Any OOMs or warnings.

## Export Best TinyCore Artifact

If `tinycore_recurrent_5090_v0` trains successfully:

```bash
python3 -m tinycore_format.cli export-tensors \
  reports/runs/instruction_code_5090_tinycore/tinycore_recurrent_5090_v0

python3 -m tinycore_format.cli convert \
  reports/runs/instruction_code_5090_tinycore/tinycore_recurrent_5090_v0 \
  reports/runs/instruction_code_5090_tinycore/tinycore_recurrent_5090_v0.tcmdl

python3 -m tinycore_format.cli verify \
  reports/runs/instruction_code_5090_tinycore/tinycore_recurrent_5090_v0.tcmdl
```

If the native binary exists, test greedy generation:

```bash
runtime/tinycore.cpp/build/tinycore-generate \
  reports/runs/instruction_code_5090_tinycore/tinycore_recurrent_5090_v0.tcmdl \
  'Q:py clamp|A:' 64 0 0 1337
```

Then test through tinycored:

```bash
npm --workspace @tinycore/tinycored run build
node packages/tinycored/dist/src/cli.js \
  --tcmdl reports/runs/instruction_code_5090_tinycore/tinycore_recurrent_5090_v0.tcmdl \
  --native-bin runtime/tinycore.cpp/build/tinycore-generate
```

In another terminal:

```bash
curl -s http://127.0.0.1:8787/chat \
  -H 'content-type: application/json' \
  -d '{"prompt":"Q:py clamp|A:","max_tokens":64,"temperature":0,"top_k":0,"seed":1337}'
```

Stop the server when finished.

## Success Criteria

The run is successful if:

- CUDA smoke run passes.
- Full report is written.
- TinyCore has competitive or better `instruction_eval_score_per_100kib_bf16`
  than the baseline.
- The selected TinyCore checkpoint exports to `.tcmdl` and verifies.
