from __future__ import annotations

import torch
from torch import nn

from tinycore_model.data import ByteTokenizer
from tinycore_model.instruction_eval import run_instruction_eval


class EchoUsefulAnswer(nn.Module):
    def __init__(self, tokenizer: ByteTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(tokens.size(0), tokens.size(1), self.tokenizer.vocab_size, device=tokens.device)

    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 0.0) -> torch.Tensor:
        del max_new_tokens, temperature
        answer = (
            "def add(a, b):\n"
            "    return a + b\n"
            "def reverse_text(text):\n"
            "    return text[::-1]\n"
            "reuse basis weights\n"
            '{"ok": true, "tokens": [1, 2, 3]}\n'
        )
        suffix = torch.tensor([self.tokenizer.encode(answer)], dtype=torch.long, device=tokens.device)
        return torch.cat([tokens, suffix], dim=1)


class EchoHoldoutAnswer(EchoUsefulAnswer):
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 0.0) -> torch.Tensor:
        del max_new_tokens, temperature
        answer = (
            "def clamp(x,lo,hi):\n"
            "    return min(max(x,lo),hi)\n"
            '{"run_group":"instruction_code_5090","device":"cuda"}\n'
            "Use shared basis weights and validation quality per stored byte.\n"
            "Compare Python and native greedy tokens.\n"
            "Select the checkpoint with the highest instruction eval score and lower reference loss.\n"
        )
        suffix = torch.tensor([self.tokenizer.encode(answer)], dtype=torch.long, device=tokens.device)
        return torch.cat([tokens, suffix], dim=1)


class EchoGithubAnswer(EchoUsefulAnswer):
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 0.0) -> torch.Tensor:
        del max_new_tokens, temperature
        answer = (
            "Summarize modules, entry points, tests, build scripts, and risky generated files.\n"
            "Prefer small typed changes and avoid editing generated bundles.\n"
            "Use public repositories with allowlisted permissive licenses.\n"
            "Skip private keys, API tokens, passwords, and generated bundles.\n"
        )
        suffix = torch.tensor([self.tokenizer.encode(answer)], dtype=torch.long, device=tokens.device)
        return torch.cat([tokens, suffix], dim=1)


def test_instruction_eval_scores_expected_substrings() -> None:
    tokenizer = ByteTokenizer()
    result = run_instruction_eval(
        EchoUsefulAnswer(tokenizer),
        tokenizer,
        torch.device("cpu"),
        max_new_tokens=48,
        temperature=0.0,
    )

    assert result["suite_name"] == "instruction_code_smoke_v0"
    assert result["num_cases"] == 4
    assert result["num_passed"] == 4
    assert result["mean_score"] == 1.0
    assert result["mean_reference_completion_loss"] > 0


def test_5090_holdout_eval_scores_expected_substrings() -> None:
    tokenizer = ByteTokenizer()
    result = run_instruction_eval(
        EchoHoldoutAnswer(tokenizer),
        tokenizer,
        torch.device("cpu"),
        suite_name="instruction_code_5090_holdout_v0",
        max_new_tokens=64,
        temperature=0.0,
    )

    assert result["suite_name"] == "instruction_code_5090_holdout_v0"
    assert result["num_cases"] == 5
    assert result["num_passed"] == 5
    assert result["mean_score"] == 1.0


def test_typescript_github_holdout_eval_scores_expected_substrings() -> None:
    tokenizer = ByteTokenizer()
    result = run_instruction_eval(
        EchoGithubAnswer(tokenizer),
        tokenizer,
        torch.device("cpu"),
        suite_name="typescript_github_holdout_v0",
        max_new_tokens=64,
        temperature=0.0,
    )

    assert result["suite_name"] == "typescript_github_holdout_v0"
    assert result["num_cases"] == 4
    assert result["num_passed"] == 4
    assert result["mean_score"] == 1.0
