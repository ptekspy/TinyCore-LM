from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .data import ByteTokenizer


@dataclass(frozen=True)
class InstructionEvalCase:
    name: str
    prompt: str
    reference_completion: str
    expected_substrings: tuple[str, ...]


INSTRUCTION_CODE_SMOKE_CASES = [
    InstructionEvalCase(
        name="py_add",
        prompt="Q: py add\nA:",
        reference_completion="\ndef add(a, b):\n    return a + b\n",
        expected_substrings=("def add", "return a + b"),
    ),
    InstructionEvalCase(
        name="py_reverse",
        prompt="Q: py reverse\nA:",
        reference_completion="\ndef reverse_text(text):\n    return text[::-1]\n",
        expected_substrings=("def reverse", "[::-1]"),
    ),
    InstructionEvalCase(
        name="shared_weights",
        prompt="Q: shared weights?\nA:",
        reference_completion="\nreuse basis weights across virtual layers\n",
        expected_substrings=("reuse", "basis"),
    ),
    InstructionEvalCase(
        name="json_ok",
        prompt="Q: json ok\nA:",
        reference_completion='\n{"ok": true, "tokens": [1, 2, 3]}\n',
        expected_substrings=('{"ok": true', "tokens"),
    ),
]

COMPACT_INSTRUCTION_CODE_CASES = [
    InstructionEvalCase(
        name="compact_add",
        prompt="Q:add|A:",
        reference_completion="def add(a,b):return a+b\n",
        expected_substrings=("def add", "return a+b"),
    ),
    InstructionEvalCase(
        name="compact_reverse",
        prompt="Q:rev|A:",
        reference_completion="def rev(s):return s[::-1]\n",
        expected_substrings=("def rev", "[::-1]"),
    ),
    InstructionEvalCase(
        name="compact_json",
        prompt="Q:json|A:",
        reference_completion='{"ok":true,"tokens":[1,2,3]}\n',
        expected_substrings=('{"ok":true', "tokens"),
    ),
    InstructionEvalCase(
        name="compact_basis",
        prompt="Q:basis|A:",
        reference_completion="reuse basis weights across virtual layers\n",
        expected_substrings=("reuse", "basis"),
    ),
]

INSTRUCTION_CODE_5090_HOLDOUT_CASES = [
    InstructionEvalCase(
        name="holdout_clamp",
        prompt="Q:py clamp|A:",
        reference_completion="def clamp(x,lo,hi):\n    return min(max(x,lo),hi)\n",
        expected_substrings=("def clamp", "min(max"),
    ),
    InstructionEvalCase(
        name="holdout_json_training",
        prompt="Q:json training|A:",
        reference_completion='{"run_group":"instruction_code_5090","device":"cuda"}\n',
        expected_substrings=('"run_group"', '"device":"cuda"'),
    ),
    InstructionEvalCase(
        name="holdout_basis_tradeoff",
        prompt="Q:basis tradeoff|A:",
        reference_completion="Use shared basis weights, low-rank deltas, and validation quality per stored byte.\n",
        expected_substrings=("basis", "stored byte"),
    ),
    InstructionEvalCase(
        name="holdout_native_verify",
        prompt="Q:native verify|A:",
        reference_completion="Compare Python and native greedy tokens for the same tcmdl artifact.\n",
        expected_substrings=("Python", "native"),
    ),
    InstructionEvalCase(
        name="holdout_best_checkpoint",
        prompt="Q:best checkpoint|A:",
        reference_completion="Select the checkpoint with the highest instruction eval score, then passes, then lower reference loss.\n",
        expected_substrings=("highest instruction eval score", "reference loss"),
    ),
]

TYPESCRIPT_GITHUB_HOLDOUT_CASES = [
    InstructionEvalCase(
        name="github_repo_summary",
        prompt="Q:repo summary|A:",
        reference_completion="Summarize modules, entry points, tests, build scripts, and risky generated files before editing.\n",
        expected_substrings=("entry points", "tests"),
    ),
    InstructionEvalCase(
        name="github_patch_rule",
        prompt="Q:typescript patch rule|A:",
        reference_completion="Prefer small typed changes, run the relevant test command, and avoid editing generated bundles.\n",
        expected_substrings=("typed changes", "generated"),
    ),
    InstructionEvalCase(
        name="github_license_rule",
        prompt="Q:github ingest license|A:",
        reference_completion="Ingest only public repositories with allowlisted permissive licenses unless the user explicitly changes policy.\n",
        expected_substrings=("public repositories", "licenses"),
    ),
    InstructionEvalCase(
        name="github_secret_rule",
        prompt="Q:github ingest secrets|A:",
        reference_completion="Skip files that look like private keys, API tokens, passwords, or generated bundles.\n",
        expected_substrings=("private keys", "tokens"),
    ),
]

FUNCTION_CALLING_STAGE3_CASES = [
    InstructionEvalCase(
        name="function_weather_call",
        prompt="Q:function call weather|A:",
        reference_completion='<functioncall> {"name":"get_weather","arguments":{"location":"London"}}\n',
        expected_substrings=("<functioncall>", "get_weather", "location"),
    ),
    InstructionEvalCase(
        name="tool_schema_rule",
        prompt="Q:tool schema rule|A:",
        reference_completion="Use the provided tool name and required JSON arguments; do not invent unavailable tools.\n",
        expected_substrings=("required JSON arguments", "do not invent"),
    ),
    InstructionEvalCase(
        name="mcp_tool_call_rule",
        prompt="Q:mcp tool call|A:",
        reference_completion="Choose the MCP server tool whose input_schema matches the task, then emit valid arguments.\n",
        expected_substrings=("MCP server tool", "input_schema", "arguments"),
    ),
    InstructionEvalCase(
        name="function_response_rule",
        prompt="Q:function response|A:",
        reference_completion="After a function response, summarize the result for the user without fabricating extra tool output.\n",
        expected_substrings=("function response", "without fabricating"),
    ),
]

EVAL_SUITES = {
    "instruction_code_smoke_v0": INSTRUCTION_CODE_SMOKE_CASES,
    "instruction_code_compact_v0": COMPACT_INSTRUCTION_CODE_CASES,
    "instruction_code_5090_holdout_v0": INSTRUCTION_CODE_5090_HOLDOUT_CASES,
    "typescript_github_holdout_v0": TYPESCRIPT_GITHUB_HOLDOUT_CASES,
    "function_calling_stage3_holdout_v0": FUNCTION_CALLING_STAGE3_CASES,
}


@torch.no_grad()
def run_instruction_eval(
    model: nn.Module,
    tokenizer: ByteTokenizer,
    device: torch.device,
    suite_name: str = "instruction_code_smoke_v0",
    max_new_tokens: int = 48,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if suite_name not in EVAL_SUITES:
        known = ", ".join(sorted(EVAL_SUITES))
        raise ValueError(f"Unknown eval suite {suite_name!r}; known suites: {known}")

    was_training = model.training
    model.eval()
    cases = []
    try:
        for case in EVAL_SUITES[suite_name]:
            prompt_ids = tokenizer.encode(case.prompt)
            tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated = model.generate(tokens, max_new_tokens, temperature)  # type: ignore[attr-defined]
            completion_ids = generated[0, len(prompt_ids) :].tolist()
            completion = tokenizer.decode(completion_ids)
            normalized = completion.lower()
            matched = [
                expected
                for expected in case.expected_substrings
                if expected.lower() in normalized
            ]
            reference_loss = _completion_loss(model, tokenizer, device, case)
            score = len(matched) / len(case.expected_substrings)
            cases.append(
                {
                    "name": case.name,
                    "prompt": case.prompt,
                    "completion": completion,
                    "expected_substrings": list(case.expected_substrings),
                    "matched_substrings": matched,
                    "reference_completion_loss": reference_loss,
                    "reference_completion_ppl": float(torch.exp(torch.tensor(reference_loss)).item()),
                    "score": score,
                    "passed": score == 1.0,
                }
            )
    finally:
        model.train(was_training)

    mean_score = sum(item["score"] for item in cases) / max(1, len(cases))
    mean_reference_loss = sum(item["reference_completion_loss"] for item in cases) / max(1, len(cases))
    return {
        "suite_name": suite_name,
        "num_cases": len(cases),
        "num_passed": sum(1 for item in cases if item["passed"]),
        "mean_score": mean_score,
        "mean_reference_completion_loss": mean_reference_loss,
        "mean_reference_completion_ppl": float(torch.exp(torch.tensor(mean_reference_loss)).item()),
        "cases": cases,
    }


def _completion_loss(
    model: nn.Module,
    tokenizer: ByteTokenizer,
    device: torch.device,
    case: InstructionEvalCase,
) -> float:
    prompt_ids = tokenizer.encode(case.prompt)
    reference_ids = tokenizer.encode(case.prompt + case.reference_completion)
    tokens = torch.tensor([reference_ids], dtype=torch.long, device=device)
    logits = model(tokens[:, :-1])  # type: ignore[operator]
    targets = tokens[:, 1:]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    first_completion_target = max(0, len(prompt_ids) - 1)
    mask = torch.zeros_like(losses, dtype=torch.bool)
    mask[:, first_completion_target:] = True
    return float(losses[mask].mean().item())
