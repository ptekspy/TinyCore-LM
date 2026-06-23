# Phase 2 - Data and Training Pipeline

## Objective

Move from toy inline data to reproducible tokenizer/dataset/checkpoint pipeline.

## Inputs

specs/tokenizer.md, specs/data_pipeline.md, schemas/training_run.schema.json

## Outputs

Tokenizer trainer, dataset packer, run manifests, checkpoint saving/loading.

## Acceptance criteria

A run can be reproduced from config. Metrics and config are saved with checkpoint.

## Notes for Codex

Do not use questionable data. Keep MVP corpus simple.

## Current status

Added `tinycore_instruction_code_v0` and
`tinycore_instruction_code_compact_v0`, generated local instruction/code
corpora, plus four-prompt evals that record generated substring score,
reference-completion loss, and behavior score per stored 100 KiB. Run them with:

```bash
python3 benchmarks/run_instruction_code_benchmark.py --config configs/instruction_code_toy.yaml
python3 benchmarks/run_instruction_code_benchmark.py --config configs/instruction_code_capacity.yaml --output reports/runs/instruction_code_capacity_report.json
python3 benchmarks/run_instruction_code_benchmark.py --config configs/instruction_code_long_tinycore.yaml --output reports/runs/instruction_code_long_tinycore_report.json
python3 benchmarks/run_instruction_code_benchmark.py --config configs/instruction_code_permuted_tinycore.yaml --output reports/runs/instruction_code_permuted_tinycore_report.json
```

The long TinyCore config enables best-eval checkpoint selection, restoring the
checkpoint with the highest instruction eval score before artifact save.
The permuted TinyCore config rotates compact corpus document order across
repeats. This breaks fixed-order memorization and currently gives the compressed
`tinycore_recurrent_lr4_state8` model a 4/4 compact eval pass while staying
smaller than baseline.
The selected long-run TinyCore artifacts have also been exported to `.tcmdl`
and verified with the native generator on compact prompts.
`tinycored` can serve those native bundles through `/generate`, and `/chat`
accepts a direct `prompt` for compact local evals while preserving message-based
chat for VSCode.

For the RTX 5090 laptop run, use:

```bash
python3 tools/generate_large_instruction_corpus.py --output-dir data/training/instruction_code_5090_v0 --train-examples 60000 --val-examples 6000
python3 benchmarks/run_instruction_code_benchmark.py --config configs/instruction_code_5090_tinycore.yaml --output reports/runs/instruction_code_5090_report.json
```

Full operator instructions are in `tasks/phase_2_5090_training_runbook.md`.

For the second RTX 5090 pass over top TypeScript GitHub repositories:

```bash
python3 tools/ingest_github_typescript_repos.py --output-dir data/training/typescript_github_top100_v0 --top-n 100 --candidate-pool 200 --dry-run
python3 tools/ingest_github_typescript_repos.py --output-dir data/training/typescript_github_top100_v0 --top-n 100 --candidate-pool 200 --max-doc-pages-per-repo 40 --max-doc-bytes 300000
python3 benchmarks/run_instruction_code_benchmark.py --config configs/typescript_github_5090_tinycore.yaml --output reports/runs/typescript_github_5090_report.json
```

Full ingestion and training instructions are in
`tasks/phase_2_github_typescript_ingestion_runbook.md`.
