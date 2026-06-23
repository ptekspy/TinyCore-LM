from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "tinycore_basis_v0"
    vocab_size: int = 128
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    n_virtual_layers: int = 4
    n_families: int = 1
    basis_rank: int = 2
    low_rank: int = 0
    recurrent_state_dim: int = 8
    mlp_ratio: int = 2
    max_seq_len: int = 32
    dropout: float = 0.0
    rope: bool = False
    tie_embeddings: bool = True
    precision_target: str = "fp32"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    seq_len: int = 32
    max_steps: int = 20
    eval_interval: int = 10
    lr: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    select_best_eval_checkpoint: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "tinycore_builtin_char_corpus_v0"
    train_fraction: float = 0.9
    repeat: int = 32


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str = "TinyCore"
    max_new_tokens: int = 40
    temperature: float = 0.8


@dataclass(frozen=True)
class EvalConfig:
    enabled: bool = False
    suite_name: str = "instruction_code_smoke_v0"
    max_new_tokens: int = 48
    temperature: float = 0.0


@dataclass(frozen=True)
class ModelVariant:
    name: str
    model_type: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    run_group: str = "first_benchmark_toy"
    seed: int = 1337
    device: str = "cpu"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    variants: list[ModelVariant] = field(default_factory=list)


def _known(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    names = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    return {key: value for key, value in data.items() if key in names}


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    variants = [
        ModelVariant(
            name=str(item["name"]),
            model_type=str(item["model_type"]),
            overrides=dict(item.get("overrides", {})),
        )
        for item in raw.get("variants", [])
    ]
    return BenchmarkConfig(
        **_known(
            BenchmarkConfig,
            {
                **raw,
                "dataset": DatasetConfig(**_known(DatasetConfig, raw.get("dataset", {}))),
                "model": ModelConfig(**_known(ModelConfig, raw.get("model", {}))),
                "training": TrainingConfig(**_known(TrainingConfig, raw.get("training", {}))),
                "generation": GenerationConfig(**_known(GenerationConfig, raw.get("generation", {}))),
                "eval": EvalConfig(**_known(EvalConfig, raw.get("eval", {}))),
                "variants": variants,
            },
        )
    )
