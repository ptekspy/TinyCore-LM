from __future__ import annotations

from tinycore_model import load_benchmark_config


def test_loads_ablation_variants() -> None:
    cfg = load_benchmark_config("configs/ablation_toy.yaml")
    assert [variant.name for variant in cfg.variants] == [
        "baseline_transformer_v0",
        "shared_layer_transformer_v0",
        "tinycore_basis_v0",
        "tinycore_lora_v0",
        "tinycore_recurrent_v0",
    ]
    assert cfg.variants[3].overrides["low_rank"] == 4
    assert cfg.variants[4].model_type == "tinycore_recurrent_v0"


def test_loads_instruction_code_eval_config() -> None:
    cfg = load_benchmark_config("configs/instruction_code_toy.yaml")
    assert cfg.dataset.name == "tinycore_instruction_code_compact_v0"
    assert cfg.dataset.repeat == 128
    assert cfg.eval.enabled is True
    assert cfg.eval.suite_name == "instruction_code_compact_v0"
    assert cfg.model.max_seq_len == 64
    assert cfg.training.max_steps == 500


def test_loads_instruction_code_capacity_sweep_config() -> None:
    cfg = load_benchmark_config("configs/instruction_code_capacity.yaml")
    assert cfg.run_group == "instruction_code_capacity"
    assert [variant.name for variant in cfg.variants] == [
        "baseline_transformer_v0",
        "tinycore_recurrent_lr4_state8",
        "tinycore_recurrent_lr8_state16",
        "tinycore_recurrent_basis4_lr8_state16",
    ]
    assert cfg.variants[2].overrides["low_rank"] == 8
    assert cfg.variants[3].overrides["basis_rank"] == 4


def test_loads_instruction_code_long_tinycore_config() -> None:
    cfg = load_benchmark_config("configs/instruction_code_long_tinycore.yaml")
    assert cfg.run_group == "instruction_code_long_tinycore"
    assert cfg.training.max_steps == 1500
    assert cfg.training.eval_interval == 250
    assert cfg.training.select_best_eval_checkpoint is True
    assert [variant.name for variant in cfg.variants] == [
        "baseline_transformer_v0",
        "tinycore_recurrent_lr4_state8",
        "tinycore_recurrent_lr8_state16",
    ]


def test_loads_instruction_code_permuted_tinycore_config() -> None:
    cfg = load_benchmark_config("configs/instruction_code_permuted_tinycore.yaml")
    assert cfg.run_group == "instruction_code_permuted_tinycore"
    assert cfg.dataset.name == "tinycore_instruction_code_compact_permuted_v0"
    assert cfg.training.select_best_eval_checkpoint is True
    assert [variant.name for variant in cfg.variants] == [
        "baseline_transformer_v0",
        "tinycore_recurrent_lr4_state8",
    ]


def test_loads_5090_training_config() -> None:
    cfg = load_benchmark_config("configs/instruction_code_5090_tinycore.yaml")
    assert cfg.device == "cuda"
    assert cfg.dataset.name == "tinycore_instruction_code_5090_v0"
    assert cfg.dataset.repeat == 1
    assert cfg.model.d_model == 384
    assert cfg.model.max_seq_len == 256
    assert cfg.training.batch_size == 64
    assert cfg.training.max_steps == 20000
    assert cfg.training.select_best_eval_checkpoint is True
    assert cfg.eval.suite_name == "instruction_code_5090_holdout_v0"


def test_loads_typescript_github_5090_training_config() -> None:
    cfg = load_benchmark_config("configs/typescript_github_5090_tinycore.yaml")
    assert cfg.device == "cuda"
    assert cfg.dataset.name == "typescript_github_top100_v0"
    assert cfg.dataset.repeat == 1
    assert cfg.model.max_seq_len == 512
    assert cfg.training.batch_size == 32
    assert cfg.training.max_steps == 30000
    assert cfg.eval.suite_name == "typescript_github_holdout_v0"
