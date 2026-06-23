# Phase 2 - GitHub TypeScript Top-100 Ingestion Runbook

## Objective

Build the second training pass dataset by discovering the current top public
TypeScript repositories on GitHub, ranked from stars and forks, then ingesting
eligible repository files and discovered public documentation pages into
`data/training/typescript_github_top100_v0`.

This runbook is for the AI/operator on the RTX 5090 laptop.

## First Message To The AI On The 5090 Machine

Use this prompt exactly:

```text
You are preparing TinyCore-LM's second training pass. From the repo root,
discover the current top TypeScript GitHub repositories by stars and forks,
ingest only public repositories with allowlisted permissive licenses, attempt
to discover each repo's public docs website from GitHub homepage/package/README
links, crawl same-site docs pages under the configured limits, skip generated
files and secrets, write the JSONL corpus and manifest, then run the CUDA smoke
and full TypeScript GitHub training config. Do not ingest private repos,
credentials, or non-allowlisted licenses unless the user explicitly changes
policy.
```

## Requirements

- Network access to GitHub.
- A GitHub token is strongly recommended for rate limits.
- Public repository metadata/content only.
- Do not use private repositories.

Create a token with the smallest useful scope. For public metadata and public
archive downloads, no repo write permission is needed.

```bash
export GITHUB_TOKEN=...
```

## Discovery Dry Run

First, discover and rank repositories without downloading code:

```bash
python3 tools/ingest_github_typescript_repos.py \
  --output-dir data/training/typescript_github_top100_v0 \
  --top-n 100 \
  --candidate-pool 200 \
  --dry-run
```

Inspect:

```bash
sed -n '1,80p' data/training/typescript_github_top100_v0/manifest.json
python3 - <<'PY'
import json
repos=json.load(open("data/training/typescript_github_top100_v0/selected_repos.json"))
for repo in repos[:20]:
    print(repo["full_name"], repo["stars"], repo["forks"], repo["license_spdx"], repo["size_kb"])
PY
```

## Full Ingestion

Run:

```bash
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

Default license allowlist:

```text
0BSD, Apache-2.0, BSD-2-Clause, BSD-3-Clause, CC0-1.0, ISC, MIT, MPL-2.0, Unlicense
```

If the user explicitly approves more licenses, pass `--allow-license SPDX-ID`.
Do not silently broaden the allowlist.

The ingestor writes:

```text
data/training/typescript_github_top100_v0/repo_candidates.json
data/training/typescript_github_top100_v0/selected_repos.json
data/training/typescript_github_top100_v0/train.jsonl
data/training/typescript_github_top100_v0/val.jsonl
data/training/typescript_github_top100_v0/eval_holdout.jsonl
data/training/typescript_github_top100_v0/manifest.json
```

## Ingestion Policy

The ingestor:

- merges GitHub repository search results sorted by stars and forks;
- ranks candidates by `stars + forks`;
- takes the top 100 by that combined score;
- skips unknown or non-allowlisted licenses by default;
- skips repositories above `--max-repo-kb`;
- extracts TypeScript, JavaScript, JSON, and Markdown only;
- skips generated/vendor/build paths such as `node_modules`, `dist`, `build`,
  `coverage`, `.next`, and `vendor`;
- skips lockfiles and TypeScript declaration files;
- skips files that look binary, minified, very large, or secret-bearing;
- attempts to discover docs sites from GitHub `homepage`, `package.json`
  `homepage`/`documentation`/`docs`, and README links whose label or URL looks
  like docs, install, quickstart, guide, API, or usage content;
- skips GitHub issue/repo pages as docs-site seeds;
- crawls only the same docs host as each seed URL;
- prefers docs-like paths such as `/docs`, `/guide`, `/install`,
  `/quickstart`, `/usage`, and `/api`;
- respects `robots.txt` when the docs site exposes one;
- writes docs pages as `source_kind: docs_site` rows.

The docs crawl exists to improve prompts like "how do I install x?" and
"what is the quickstart command for x?". It is intentionally capped. If a docs
site uses a client-side app that hides text from static HTML, expect few or no
docs rows for that repo.

## Verify Corpus

```bash
python3 - <<'PY'
import json
from tinycore_model.data import make_token_splits, dataset_manifest

train, val, tok = make_token_splits(corpus_name="typescript_github_top100_v0", repeat=1)
manifest = dataset_manifest(train, val, tok, "typescript_github_top100_v0", repeat=1)
print("train_tokens", train.numel())
print("val_tokens", val.numel())
print(json.dumps(manifest, indent=2))
PY
```

Also inspect skipped repos:

```bash
python3 - <<'PY'
import json
m=json.load(open("data/training/typescript_github_top100_v0/manifest.json"))
print("train_rows", m.get("train_rows"))
print("val_rows", m.get("val_rows"))
print("code_rows", m.get("code_rows"))
print("doc_rows", m.get("doc_rows"))
print("skipped", len(m.get("skipped", [])))
for item in m.get("skipped", [])[:30]:
    print(item)
PY
```

Inspect docs coverage:

```bash
python3 - <<'PY'
import json
m=json.load(open("data/training/typescript_github_top100_v0/manifest.json"))
for repo, info in sorted(m.get("docs_by_repo", {}).items()):
    if info.get("rows"):
        print(repo, "docs_rows=", info["rows"], "seeds=", info.get("seed_urls", [])[:2])
PY
```

If too many top repositories are skipped by license or size, report that before
changing policy.

Some very high-ranked TypeScript repositories are extremely large or have
unknown license metadata. Skipping them under the default policy is expected.
Do not treat this as a failed ingestion unless the final train/val row counts
are too small for training.

## Training Config

Use:

```text
configs/typescript_github_5090_tinycore.yaml
```

Important settings:

```yaml
device: cuda
dataset:
  name: typescript_github_top100_v0
  repeat: 1
model:
  d_model: 384
  n_heads: 8
  n_virtual_layers: 12
  max_seq_len: 512
  precision_target: bf16
training:
  batch_size: 32
  seq_len: 512
  max_steps: 30000
  eval_interval: 1500
  select_best_eval_checkpoint: true
eval:
  suite_name: typescript_github_holdout_v0
```

## Smoke Run

```bash
python3 - <<'PY'
from dataclasses import replace
from tinycore_model import load_benchmark_config, run_benchmark

cfg = load_benchmark_config("configs/typescript_github_5090_tinycore.yaml")
cfg = replace(
    cfg,
    run_group="typescript_github_5090_smoke",
    training=replace(cfg.training, max_steps=20, eval_interval=10, batch_size=4),
)
report = run_benchmark(cfg, "reports/runs/typescript_github_5090_smoke_report.json")
print(report["conclusion"])
for model in report["models"]:
    print(model["name"], model["val_loss"], model["stored_unique_bytes_bf16"])
PY
```

Do not start the full run if the smoke run fails.

## Full Run

```bash
python3 benchmarks/run_instruction_code_benchmark.py \
  --config configs/typescript_github_5090_tinycore.yaml \
  --output reports/runs/typescript_github_5090_report.json
```

Expected artifacts:

```text
reports/runs/typescript_github_5090_report.json
reports/runs/typescript_github_5090_tinycore/baseline_transformer_ts_github_v0/
reports/runs/typescript_github_5090_tinycore/tinycore_recurrent_ts_github_v0/
```

## If Out Of Memory

Apply one change at a time:

1. `training.batch_size: 16`
2. `training.batch_size: 8`
3. `training.seq_len: 384` and `model.max_seq_len: 384`
4. `training.seq_len: 256` and `model.max_seq_len: 256`

Run the smoke run after every change.

## Export TinyCore Artifact

```bash
python3 -m tinycore_format.cli export-tensors \
  reports/runs/typescript_github_5090_tinycore/tinycore_recurrent_ts_github_v0

python3 -m tinycore_format.cli convert \
  reports/runs/typescript_github_5090_tinycore/tinycore_recurrent_ts_github_v0 \
  reports/runs/typescript_github_5090_tinycore/tinycore_recurrent_ts_github_v0.tcmdl

python3 -m tinycore_format.cli verify \
  reports/runs/typescript_github_5090_tinycore/tinycore_recurrent_ts_github_v0.tcmdl
```

## What To Report Back

Report:

- timestamp and machine GPU details from `nvidia-smi`;
- GitHub token used or not used, without revealing the token;
- selected repository count and skipped count by reason;
- top 20 selected repos with stars, forks, license, size;
- train/val row counts, code/docs row counts, and token counts;
- docs coverage by repo, including seed URLs and any repeated crawl errors;
- any secret/license/size skips;
- final report path;
- selected checkpoint step for each model;
- `val_loss`, `instruction_eval_mean_score`, `instruction_eval_passed`,
  `reference_completion_loss`, `instruction_eval_score_per_100kib_bf16`,
  and stored bytes;
- export/verify status for the TinyCore `.tcmdl`.

## Success Criteria

- Corpus manifest exists and records selected/skipped repositories.
- No private repos or non-allowlisted licenses were ingested without explicit
  user approval.
- CUDA smoke run passes.
- Full training report is written.
- TinyCore artifact exports to `.tcmdl` and verifies.
