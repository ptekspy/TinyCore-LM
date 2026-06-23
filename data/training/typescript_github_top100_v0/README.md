# TypeScript GitHub Top-100 Corpus

This directory is the target for TinyCore-LM's second training pass.

It is intentionally not pre-filled on this machine. The RTX 5090 training
machine should discover the current public TypeScript repositories from GitHub,
rank them by `stars + forks`, ingest eligible repository files, attempt to
discover each repo's public docs website, ingest capped same-site docs pages,
and write:

```text
repo_candidates.json
selected_repos.json
train.jsonl
val.jsonl
eval_holdout.jsonl
manifest.json
```

Run from the repo root:

```bash
export GITHUB_TOKEN=...

python3 tools/ingest_github_typescript_repos.py \
  --output-dir data/training/typescript_github_top100_v0 \
  --top-n 100 \
  --candidate-pool 200 \
  --dry-run

python3 tools/ingest_github_typescript_repos.py \
  --output-dir data/training/typescript_github_top100_v0 \
  --top-n 100 \
  --candidate-pool 200 \
  --max-repo-kb 250000 \
  --max-files-per-repo 1200 \
  --max-file-bytes 200000 \
  --max-doc-pages-per-repo 40 \
  --max-doc-bytes 300000 \
  --val-fraction 0.02
```

Docs ingestion is enabled by default. Use `--skip-doc-sites` only for a
code-only control run. Docs rows are intended to teach install, quickstart,
usage, and API commands alongside the repository source.

Then train:

```bash
python3 benchmarks/run_instruction_code_benchmark.py \
  --config configs/typescript_github_5090_tinycore.yaml \
  --output reports/runs/typescript_github_5090_report.json
```

See `tasks/phase_2_github_typescript_ingestion_runbook.md` for the full
operator instructions, smoke run, OOM fallback settings, and artifact export
commands.
