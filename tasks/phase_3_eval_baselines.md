# Phase 3 - Evaluation and Ablations

## Objective

Make the research loop truthful by comparing model variants against baselines.

## Inputs

specs/evaluation.md, tasks/task_dag.json

## Outputs

Ablation runner, report.json, parameter accounting comparison.
Function-calling stage-3 corpus, config, and 5090 runbook.

## Acceptance criteria

At least three variants compared: baseline, basis-only TinyCore, basis+low-rank TinyCore.
Function-calling/tool-use training can be launched from
`configs/function_calling_stage3_5090_tinycore.yaml` after ingesting
`data/training/function_calling_stage3_v0`.

## Notes for Codex

The result may show TinyCore worse. That is acceptable and useful.

Stage 3 training data comes from:

- `Salesforce/xlam-function-calling-60k`
- `glaiveai/glaive-function-calling-v2`
- `MCPToolBench/MCPToolBenchPP`

Use `tasks/phase_3_function_calling_training_runbook.md` for the RTX 5090
ingestion, smoke run, full run, and export steps.
