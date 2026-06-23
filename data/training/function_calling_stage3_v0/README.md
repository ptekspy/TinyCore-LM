# Function Calling Stage 3 Corpus

This directory is the target for TinyCore-LM's third training round.

The corpus is built from:

- `Salesforce/xlam-function-calling-60k`
- `glaiveai/glaive-function-calling-v2`
- `MCPToolBench/MCPToolBenchPP`

The ingestor writes:

```text
dataset_plan.json
train.jsonl
val.jsonl
eval_holdout.jsonl
manifest.json
```

`Salesforce/xlam-function-calling-60k` is gated on Hugging Face. Before the
5090 run, accept the dataset terms in a browser and set `HF_TOKEN`.

Run from the repo root:

```bash
export HF_TOKEN=...

python3 tools/ingest_function_calling_stage3.py \
  --output-dir data/training/function_calling_stage3_v0 \
  --dry-run

python3 tools/ingest_function_calling_stage3.py \
  --output-dir data/training/function_calling_stage3_v0 \
  --val-fraction 0.02 \
  --max-record-chars 24000
```

Then train:

```bash
python3 benchmarks/run_instruction_code_benchmark.py \
  --config configs/function_calling_stage3_5090_tinycore.yaml \
  --output reports/runs/function_calling_stage3_5090_report.json
```

See `tasks/phase_3_function_calling_training_runbook.md` for the full operator
instructions, smoke run, OOM fallback settings, and artifact export commands.
