# TinyCore-LM

This repo now contains the first executable research loop for the TinyCore-LM
spec pack.

## Quickstart

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 benchmarks/run_first_benchmark.py --config configs/first_benchmark_toy.yaml
python3 benchmarks/run_ablation.py --config configs/ablation_toy.yaml
python3 benchmarks/run_instruction_code_benchmark.py --config configs/instruction_code_toy.yaml
python3 benchmarks/run_instruction_code_benchmark.py --config configs/instruction_code_capacity.yaml --output reports/runs/instruction_code_capacity_report.json
python3 benchmarks/run_instruction_code_benchmark.py --config configs/instruction_code_long_tinycore.yaml --output reports/runs/instruction_code_long_tinycore_report.json
python3 benchmarks/run_instruction_code_benchmark.py --config configs/instruction_code_permuted_tinycore.yaml --output reports/runs/instruction_code_permuted_tinycore_report.json
python3 tools/generate_large_instruction_corpus.py --output-dir data/training/instruction_code_5090_v0 --train-examples 60000 --val-examples 6000
python3 benchmarks/run_instruction_code_benchmark.py --config configs/instruction_code_5090_tinycore.yaml --output reports/runs/instruction_code_5090_report.json
python3 tools/ingest_github_typescript_repos.py --output-dir data/training/typescript_github_top100_v0 --top-n 100 --candidate-pool 200 --dry-run
python3 tools/ingest_github_typescript_repos.py --output-dir data/training/typescript_github_top100_v0 --top-n 100 --candidate-pool 200 --max-doc-pages-per-repo 40 --max-doc-bytes 300000
python3 benchmarks/run_instruction_code_benchmark.py --config configs/typescript_github_5090_tinycore.yaml --output reports/runs/typescript_github_5090_report.json
npm install
npm test
node packages/tinycore-agent/dist/src/agent_benchmark_cli.js --output reports/runs/agent_eval_report.json
node packages/tinycore-agent/dist/src/agent_benchmark_cli.js --suite --output reports/runs/agent_eval_suite_report.json
node packages/tinycored/dist/src/cli.js --repo-root . --artifact-dir reports/runs/ablation_toy/tinycore_basis_v0 --port 8787
python3 -m tinycore_format.cli verify reports/runs/ablation_toy/tinycore_recurrent_v0
python3 -m tinycore_format.cli export-tensors reports/runs/ablation_toy/tinycore_recurrent_v0
python3 -m tinycore_format.cli convert reports/runs/ablation_toy/tinycore_recurrent_v0 reports/runs/ablation_toy/tinycore_recurrent_v0.tcmdl
python3 -m tinycore_format.cli verify reports/runs/ablation_toy/tinycore_recurrent_v0.tcmdl
python3 -m tinycore_format.cli extract reports/runs/ablation_toy/tinycore_recurrent_v0.tcmdl reports/runs/ablation_toy/tinycore_recurrent_v0_extracted
python3 -m tinycore_format.cli estimate-size configs/ablation_toy.yaml
python3 -m tinycore_format.cli export-tensors reports/runs/instruction_code_long_tinycore/tinycore_recurrent_lr4_state8
python3 -m tinycore_format.cli convert reports/runs/instruction_code_long_tinycore/tinycore_recurrent_lr4_state8 reports/runs/instruction_code_long_tinycore/tinycore_recurrent_lr4_state8.tcmdl
runtime/tinycore.cpp/build/tinycore-generate reports/runs/instruction_code_long_tinycore/tinycore_recurrent_lr4_state8.tcmdl 'Q:add|A:' 16 0 0 1337
python3 -m tinycore_format.cli export-tensors reports/runs/instruction_code_permuted_tinycore/tinycore_recurrent_lr4_state8
python3 -m tinycore_format.cli convert reports/runs/instruction_code_permuted_tinycore/tinycore_recurrent_lr4_state8 reports/runs/instruction_code_permuted_tinycore/tinycore_recurrent_lr4_state8.tcmdl
runtime/tinycore.cpp/build/tinycore-generate reports/runs/instruction_code_permuted_tinycore/tinycore_recurrent_lr4_state8.tcmdl 'Q:json|A:' 32 0 0 1337
node packages/tinycored/dist/src/cli.js --tcmdl reports/runs/instruction_code_permuted_tinycore/tinycore_recurrent_lr4_state8.tcmdl --native-bin runtime/tinycore.cpp/build/tinycore-generate
curl -s http://127.0.0.1:8787/chat -H 'content-type: application/json' -d '{"prompt":"Q:add|A:","max_tokens":16,"temperature":0,"top_k":0,"seed":1337}'
printf '%s' '{"artifact_dir":"reports/runs/ablation_toy/tinycore_basis_v0","prompt":"TinyCore","max_tokens":8,"temperature":0}' | python3 -m tinycore_model.generate_cli
cmake -S runtime/tinycore.cpp -B runtime/tinycore.cpp/build
cmake --build runtime/tinycore.cpp/build
runtime/tinycore.cpp/build/tinycore-inspect-manifest reports/runs/ablation_toy/tinycore_recurrent_v0
runtime/tinycore.cpp/build/tinycore-inspect-manifest reports/runs/ablation_toy/tinycore_recurrent_v0.tcmdl
```

The first benchmark writes:

- `reports/runs/first_benchmark_report.json`
- `reports/progress.json`

The ablation benchmark writes `reports/runs/ablation_report.json` and compares
a standard baseline, a shared-layer transformer compression baseline,
basis-only TinyCore, basis+low-rank TinyCore, and recurrent-state TinyCore.
The instruction/code benchmark writes `reports/runs/instruction_code_report.json`
and adds tiny generated instruction/code corpora plus four-prompt behavioral
evals for Python, JSON, and shared-weight explanations. The capacity config
writes `reports/runs/instruction_code_capacity_report.json` and sweeps recurrent
TinyCore low-rank, basis, and state budgets against the baseline.
The long TinyCore config writes
`reports/runs/instruction_code_long_tinycore_report.json` and can restore the
best instruction-eval checkpoint before saving artifacts, which prevents a
lower final token loss from overwriting a better greedy-generation checkpoint.
The permuted TinyCore config writes
`reports/runs/instruction_code_permuted_tinycore_report.json`; it rotates compact
document order across repeats so the compressed TinyCore learns prompt mappings
instead of a fixed corpus cycle. In the current run, `tinycore_recurrent_lr4_state8`
is smaller than baseline, lower loss, and passes all four compact eval prompts.
The 5090 training package uses `data/training/instruction_code_5090_v0`, a
generated 60k/6k train/validation JSONL corpus. Follow
`tasks/phase_2_5090_training_runbook.md` on the RTX 5090 laptop.
The second 5090 pass uses `tools/ingest_github_typescript_repos.py` to discover
the current top public TypeScript GitHub repositories by stars and forks, ingest
allowlisted permissive-license source plus discovered docs-site pages for
install, quickstart, usage, and API commands, then train
`configs/typescript_github_5090_tinycore.yaml`. Follow
`tasks/phase_2_github_typescript_ingestion_runbook.md`.
The selected long-run TinyCore artifacts can be exported to `.tcmdl`; the native
generator matches the Python runtime on compact instruction/code prompts.

Each benchmark also writes per-model artifact directories under
`reports/runs/<run_group>/<model_name>/` with `checkpoint.pt`, `manifest.json`,
`config.yaml`, `tokenizer.json`, and `metrics.json`. The format tools can add
native-friendly `tensor_index.json` and `tensors.bin` sidecars from the PyTorch
checkpoint.

The TypeScript agent protocol lives in `packages/tinycore-agent`.
It currently includes the agent loop plus repo tools for listing, reading,
literal search, simple symbol search, git status/diff, safe command execution,
test execution, and validated single-file patch replacement.
It also includes a synthetic coding-agent benchmark that fixes a failing test
in a disposable repo and writes `reports/runs/agent_eval_report.json`.
The suite mode currently runs two tasks and writes
`reports/runs/agent_eval_suite_report.json`.

The local server shell lives in `packages/tinycored`. It exposes `GET /health`,
`POST /agent/step` for repo-tool execution, and explicit 501 responses for
unsupported chat modes. `POST /generate` can call the Python checkpoint runtime
when configured with a model artifact directory. `POST /chat` supports direct
message generation through that runtime and explicit `tool_call` execution via
repo tools.

The VSCode shell lives in `packages/tinycore-code`. The Python-phase format
tools live in `packages/tinycore-format` and inspect/verify manifest checkpoint
directories before any binary TCMDL work begins.
The extension `Ask` and `Explain Selected Code` commands call the `/chat`
endpoint on `tinycored`.

The native runtime spike lives in `runtime/tinycore.cpp`. It only inspects the
Python-phase manifest or the deterministic `.tcmdl` bundle header. The format
tools can pack and extract `.tcmdl` bundles with checksum verification, but
the native inspector also validates the bundle payload length, indexes payload
files, reads the embedded manifest section metadata, and checks the raw tensor
sidecar metadata. It also probes selected float32 tensors from `tensors.bin` to
confirm native byte loading, and runs a tiny native embedding/head dot-product
probe. It now also composes one TinyCore Q projection from basis weights,
routing coefficients, and low-rank deltas, then runs a single-token RMSNorm
plus Q-projection block probe. It also runs a one-token attention sublayer
probe through K/V/O and residual addition, followed by recurrent-state and MLP
reference probes for the same token. The spike can now apply final norm and
lm_head to produce logits after one TinyCore virtual layer, and can run the
same one-token reference path across all virtual layers. It also runs a
two-token causal-attention sequence probe across all virtual layers. A
prompt-level native greedy generation probe for `TinyCore` now produces
`TinyCoreeeerree `, matching the Python runtime for the same artifact. Full
model kernels are still intentionally out of scope.

You can run the current native generator directly:

```bash
runtime/tinycore.cpp/build/tinycore-generate reports/runs/ablation_toy/tinycore_recurrent_v0.tcmdl TinyCore 8
```

Optional native sampling arguments are `temperature`, `top_k`, and `seed`:

```bash
runtime/tinycore.cpp/build/tinycore-generate reports/runs/ablation_toy/tinycore_recurrent_v0.tcmdl TinyCore 8 0.8 4 2026
```

You can also run `tinycored` against the native `.tcmdl` backend:

```bash
npm --workspace @tinycore/tinycored run build
node packages/tinycored/dist/src/cli.js --tcmdl reports/runs/instruction_code_permuted_tinycore/tinycore_recurrent_lr4_state8.tcmdl --native-bin runtime/tinycore.cpp/build/tinycore-generate
```

Then send a local generation request:

```bash
curl -s http://127.0.0.1:8787/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"TinyCore","max_tokens":8,"temperature":0,"top_k":0}'
```

For compact local model evals, `/chat` also accepts a direct `prompt` body and
returns only the generated completion:

```bash
curl -s http://127.0.0.1:8787/chat \
  -H 'content-type: application/json' \
  -d '{"prompt":"Q:add|A:","max_tokens":16,"temperature":0,"top_k":0,"seed":1337}'
```

The VSCode shell uses `tinycore.serverUrl` and forwards generation settings
from `tinycore.askMaxTokens`, `tinycore.explainMaxTokens`,
`tinycore.temperature`, `tinycore.topK`, and `tinycore.seed`.

The shared native runtime now lives in `native_runtime.hpp` and
`native_runtime.cpp`, linked into both the inspector and generator through a
CMake object-library target. Artifact loading, prompt logits, and greedy
generation are shared by the inspector probes and `tinycore-generate`.
Temperature/top-k sampling is available in the native runtime while greedy stays
the deterministic regression path. The native CTest suite checks tensor parsing,
composed Q projection math, greedy generation, and sampled top-k invariants
directly, while the Python test suite pins the native output for both packed and
extracted artifacts when the native binary is present. `tinycored` can now use
either the Python checkpoint runtime via `--artifact-dir` or the native `.tcmdl`
runtime via `--tcmdl`, and the VSCode client can pass deterministic or sampled
generation settings through `/chat`. Chat responses strip echoed prompt text
when a runtime returns full prompt-plus-completion output. Native generation
also clamps requested new tokens to the artifact context window.

```bash
ctest --test-dir runtime/tinycore.cpp/build --output-on-failure
```

The benchmark is intentionally tiny. Its job is to prove the loop is real:
baseline, shared-layer baseline, and TinyCore variants train on the same toy
corpus, generate tokens, and report stored unique bytes.
