# Phase 3 - Tool-Calling Evaluation

## Objective

Measure whether a TinyCore artifact can choose tools, emit valid JSON
arguments, avoid hallucinated tools, and answer from function responses.

This eval should run twice:

1. **Before stage 3 training** against the best currently available artifact.
2. **After stage 3 training** against the stage-3 TinyCore artifact.

The before/after delta is the useful signal.

## Eval Command

Pre-stage-3 baseline, using the current compact TinyCore artifact:

```bash
python3 benchmarks/run_stage_3_tool_eval.py \
  --artifact-dir reports/runs/instruction_code_permuted_tinycore/tinycore_recurrent_lr4_state8 \
  --output reports/runs/stage_3_tool_eval_after_stage2_report.json
```

Post-stage-3 eval, after the 5090 full run and export:

```bash
python3 benchmarks/run_stage_3_tool_eval.py \
  --artifact-dir reports/runs/function_calling_stage3_5090_tinycore/tinycore_recurrent_function_calling_v0 \
  --output reports/runs/stage_3_tool_eval_after_stage3_report.json
```

## Metrics

The report includes:

- `overall_score`
- `num_passed`
- `tool_call_valid_rate`
- `tool_name_accuracy`
- `argument_schema_pass_rate`
- `argument_match_rate`
- `no_tool_precision`
- `final_answer_after_tool_rate`

## Cases

The suite covers:

- valid weather function call;
- required currency conversion arguments;
- integer argument type checking;
- MCP-style file read;
- MCP-style repository search;
- no-tool creative request;
- no hallucinated news tool when only calculator is available;
- function response to final answer.

## What Good Looks Like

The current stage-2 artifact is expected to score poorly. That is fine. It is a
baseline.

After stage 3, look for improvement in:

- `tool_name_accuracy`
- `argument_schema_pass_rate`
- `argument_match_rate`
- `no_tool_precision`

Any failed post-stage-3 cases should become stage 3.5 corrective examples.
