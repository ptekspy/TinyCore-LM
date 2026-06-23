from __future__ import annotations

from tinycore_model import ModelVariant
from tinycore_model.benchmark import _conclusion, _instruction_eval_selection_key, _variant_seed


def test_variant_seed_is_stable_by_architecture() -> None:
    first = ModelVariant("tinycore_recurrent_lr4_state8", "tinycore_recurrent_v0")
    renamed = ModelVariant("renamed_recurrent", "tinycore_recurrent_v0")
    other = ModelVariant("tinycore_recurrent_lr8_state16", "tinycore_recurrent_v0", {"low_rank": 8})

    assert _variant_seed(1337, first) == _variant_seed(1337, renamed)
    assert _variant_seed(1337, first) != _variant_seed(1337, other)


def test_conclusion_prefers_compressed_better_before_larger_lowest_loss() -> None:
    results = [
        {"name": "baseline", "val_loss": 0.08, "stored_unique_bytes_bf16": 100.0},
        {"name": "compressed_better", "val_loss": 0.06, "stored_unique_bytes_bf16": 90.0},
        {"name": "larger_best_loss", "val_loss": 0.04, "stored_unique_bytes_bf16": 120.0},
    ]

    assert _conclusion(results) == "best_compressed_better:compressed_better"


def test_instruction_eval_selection_prefers_score_then_passes_then_loss() -> None:
    stronger_score = {
        "instruction_eval_mean_score": 0.75,
        "instruction_eval_passed": 1,
        "reference_completion_loss": 0.5,
    }
    weaker_score_lower_loss = {
        "instruction_eval_mean_score": 0.5,
        "instruction_eval_passed": 4,
        "reference_completion_loss": 0.1,
    }
    same_score_more_passes = {
        "instruction_eval_mean_score": 0.75,
        "instruction_eval_passed": 2,
        "reference_completion_loss": 0.8,
    }
    same_score_passes_lower_loss = {
        "instruction_eval_mean_score": 0.75,
        "instruction_eval_passed": 2,
        "reference_completion_loss": 0.4,
    }

    assert _instruction_eval_selection_key(stronger_score) > _instruction_eval_selection_key(weaker_score_lower_loss)
    assert _instruction_eval_selection_key(same_score_more_passes) > _instruction_eval_selection_key(stronger_score)
    assert _instruction_eval_selection_key(same_score_passes_lower_loss) > _instruction_eval_selection_key(same_score_more_passes)
