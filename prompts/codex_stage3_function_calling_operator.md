# Codex Prompt: Stage 3 Function Calling Training Operator

You are the AI operator implementing TinyCore-LM stage 3 on an RTX 5090
machine. Your job is to ingest function-calling/tool-use datasets, verify the
corpus, run the CUDA training job, export the selected TinyCore artifact, and
report the result clearly.

Work from the repository root.

## Non-Negotiables

- Do not reveal, print, commit, or write `HF_TOKEN`.
- Use only public datasets or gated datasets whose terms the user has accepted.
- Do not use private Hugging Face datasets.
- Do not silently skip `Salesforce/xlam-function-calling-60k`. If it fails with
  HTTP 401/403, stop and tell the user they must accept the dataset terms or
  provide a valid `HF_TOKEN`.
- Preserve tool schemas, tool names, arguments, and function responses in the
  training text. Do not flatten them into vague summaries.
- Keep skipped/filtered row accounting in the manifest.
- Run the smoke training job before the full training job.
- Do not start the full training job if ingestion verification or the smoke run
  fails.

## Target Datasets

Use exactly these Hugging Face dataset ids:

```text
Salesforce/xlam-function-calling-60k
glaiveai/glaive-function-calling-v2
MCPToolBench/MCPToolBenchPP
```

Expected access notes:

- `Salesforce/xlam-function-calling-60k` is gated/auto-approved and requires
  accepted terms plus `HF_TOKEN` for raw downloads.
- `glaiveai/glaive-function-calling-v2` is public.
- `MCPToolBench/MCPToolBenchPP` is public, but the metadata may not expose a
  standard license. Report this in the final summary.

## Files You Should Use

```text
tools/ingest_function_calling_stage3.py
configs/function_calling_stage3_5090_tinycore.yaml
data/training/function_calling_stage3_v0/
tasks/phase_3_function_calling_training_runbook.md
```

## Environment Setup

From the repository root:

```bash
python3 -m pip install -e ".[dev,hf]"
```

Confirm CUDA visibility:

```bash
nvidia-smi
python3 - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY
```

Set the token in the shell without echoing it back:

```bash
export HF_TOKEN=...
```

Do not paste the token into any file.

## Step 1: Dry-Run Dataset Discovery

Run:

```bash
python3 tools/ingest_function_calling_stage3.py \
  --output-dir data/training/function_calling_stage3_v0 \
  --dry-run
```

Inspect the generated plan:

```bash
python3 - <<'PY'
import json
plan=json.load(open("data/training/function_calling_stage3_v0/dataset_plan.json"))
for item in plan:
    print(item["dataset_id"])
    print("  gated:", item["gated"])
    print("  private:", item["private"])
    print("  files:", len(item["files"]))
    for file in item["files"][:10]:
        print("   ", file)
PY
```

Expected dry-run shape:

- xLAM has `xlam_function_calling_60k.json`.
- Glaive has `glaive-function-calling-v2.json`.
- MCPToolBench++ has several `data/.../*.json` files.
- MCPToolBench++ demo files are skipped by default.

If the dry-run cannot read metadata, fix networking or token setup before
continuing.

## Step 2: Full Ingestion

Run:

```bash
python3 tools/ingest_function_calling_stage3.py \
  --output-dir data/training/function_calling_stage3_v0 \
  --val-fraction 0.02 \
  --max-record-chars 24000
```

Expected files after ingestion:

```text
data/training/function_calling_stage3_v0/dataset_plan.json
data/training/function_calling_stage3_v0/train.jsonl
data/training/function_calling_stage3_v0/val.jsonl
data/training/function_calling_stage3_v0/eval_holdout.jsonl
data/training/function_calling_stage3_v0/manifest.json
```

If you need a tiny local debug run before the full ingestion, use:

```bash
python3 tools/ingest_function_calling_stage3.py \
  --output-dir data/training/function_calling_stage3_v0 \
  --max-rows-per-dataset 1000 \
  --max-record-chars 24000
```

Do not treat this small debug corpus as the real stage-3 corpus.

## Step 3: Audit The Corpus

Run:

```bash
python3 - <<'PY'
import json
m=json.load(open("data/training/function_calling_stage3_v0/manifest.json"))
print("train_rows", m.get("train_rows"))
print("val_rows", m.get("val_rows"))
print("num_documents", m.get("num_documents"))
print("num_skipped", m.get("num_skipped"))
print("rows_by_dataset")
for dataset, rows in m.get("rows_by_dataset", {}).items():
    print(" ", dataset, rows)
print("files_by_dataset")
for dataset, files in m.get("files_by_dataset", {}).items():
    print(" ", dataset, len(files))
    for file in files[:10]:
        print("   ", file)
print("representative skipped")
for item in m.get("skipped", [])[:20]:
    print(" ", item)
PY
```

Then verify the model loader can read it:

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

Stop and ask for guidance if:

- `train_rows` is zero;
- any required dataset has zero rows;
- xLAM is absent because of access failure;
- skipped rows are dominated by a new unexpected error;
- validation tokens are zero or tiny.

## Step 4: Smoke Training Run

Run:

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

Confirm:

- the run uses `device: cuda`;
- both baseline and TinyCore variants finish;
- `reports/runs/function_calling_stage3_5090_smoke_report.json` exists;
- instruction eval suite is `function_calling_stage3_holdout_v0`.

If CUDA runs out of memory during smoke, change one thing at a time:

1. `training.batch_size: 8`
2. `training.batch_size: 4`
3. `training.seq_len: 512` and `model.max_seq_len: 512`
4. `model.d_model: 384`, `training.seq_len: 512`, and `model.max_seq_len: 512`

Re-run the smoke after every change. Do not start the full run until smoke
passes.

## Step 5: Full Training Run

Run:

```bash
python3 benchmarks/run_instruction_code_benchmark.py \
  --config configs/function_calling_stage3_5090_tinycore.yaml \
  --output reports/runs/function_calling_stage3_5090_report.json
```

Expected outputs:

```text
reports/runs/function_calling_stage3_5090_report.json
reports/runs/function_calling_stage3_5090_tinycore/baseline_transformer_function_calling_v0/
reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0/
```

After the run, inspect:

```bash
python3 - <<'PY'
import json
r=json.load(open("reports/runs/function_calling_stage3_5090_report.json"))
print("conclusion", r["conclusion"])
for model in r["models"]:
    selected = model.get("selected_checkpoint", {})
    eval_result = model.get("instruction_code_eval") or {}
    print(model["name"])
    print("  selected_step", selected.get("step"))
    print("  val_loss", model.get("val_loss"))
    print("  stored_unique_bytes_bf16", model.get("stored_unique_bytes_bf16"))
    print("  instruction_eval_mean_score", eval_result.get("mean_score"))
    print("  instruction_eval_passed", eval_result.get("num_passed"))
    print("  reference_completion_loss", eval_result.get("mean_reference_completion_loss"))
    print("  score_per_100kib", model.get("instruction_eval_score_per_100kib_bf16"))
PY
```

## Step 6: Export The TinyCore Artifact

Run:

```bash
python3 -m tinycore_format.cli export-tensors \
  reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0

python3 -m tinycore_format.cli convert \
  reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0 \
  reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0.tcmdl

python3 -m tinycore_format.cli verify \
  reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0.tcmdl
```

Expected exported artifact:

```text
reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0.tcmdl
```

## Step 7: Final Report To User

Report these fields:

- current git branch and commit;
- timestamp;
- `nvidia-smi` GPU model/driver/CUDA summary;
- whether `HF_TOKEN` was used, without revealing it;
- whether xLAM gated access succeeded;
- dataset files ingested;
- rows per dataset;
- train/val row counts;
- train/val token counts;
- skipped row count and top skip reasons;
- smoke report path;
- full report path;
- selected checkpoint step for each model;
- for each model: `val_loss`, `stored_unique_bytes_bf16`,
  `instruction_eval_mean_score`, `instruction_eval_passed`,
  `reference_completion_loss`, and `instruction_eval_score_per_100kib_bf16`;
- export path and verify status for the `.tcmdl` artifact;
- any config changes made for OOM.

Keep the report concise, but include exact paths and numbers.
