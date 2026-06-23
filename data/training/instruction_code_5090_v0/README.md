# TinyCore 5090 Instruction/Code Corpus v0

This directory contains a larger deterministic local corpus for the RTX 5090
laptop training pass.

Files:

- `train.jsonl`: 60,000 generated training examples.
- `val.jsonl`: 6,000 generated validation examples.
- `eval_holdout.jsonl`: 8 hand-specified holdout prompts for operator review.
- `manifest.json`: counts, generator path, seed, hashes, and license notes.

Each JSONL row has:

```json
{"id":"...","split":"train","category":"python","prompt":"...","response":"...","text":"..."}
```

The corpus is synthetic and generated from repository-owned templates. It does
not include scraped text, third-party code, or private data.

Regenerate it with:

```bash
python3 tools/generate_large_instruction_corpus.py \
  --output-dir data/training/instruction_code_5090_v0 \
  --train-examples 60000 \
  --val-examples 6000
```

The training config that consumes this corpus is:

```bash
configs/instruction_code_5090_tinycore.yaml
```
