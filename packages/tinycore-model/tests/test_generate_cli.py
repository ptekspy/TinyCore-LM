from __future__ import annotations

import torch

from tinycore_model import ModelConfig, TinyCoreLM, save_model_artifact
from tinycore_model.data import ByteTokenizer
from tinycore_model.generate_cli import generate_from_artifact


def test_generate_from_artifact(tmp_path) -> None:
    torch.manual_seed(3)
    cfg = ModelConfig(
        model_type="tinycore_basis_v0",
        vocab_size=128,
        d_model=32,
        n_heads=4,
        n_virtual_layers=2,
        basis_rank=2,
        max_seq_len=16,
    )
    dataset = {
        "name": "test_dataset",
        "source": "synthetic",
        "license_notes": "test",
        "num_documents": 1,
        "num_tokens": 16,
        "tokenizer": "byte_char_mvp",
        "train_shards": [],
        "val_shards": [],
    }
    save_model_artifact(
        artifact_dir=tmp_path,
        model=TinyCoreLM(cfg),
        model_cfg=cfg,
        run_id="generate_test",
        model_name="tinycore_basis_v0",
        dataset_manifest=dataset,
        tokenizer=ByteTokenizer(),
        metrics={"val_loss": 1.0},
        benchmark_config={"run_group": "test"},
    )

    result = generate_from_artifact(
        {
            "artifact_dir": str(tmp_path),
            "prompt": "Tiny",
            "max_tokens": 4,
            "temperature": 0,
            "seed": 123,
        }
    )

    assert result["runtime"] == "python"
    assert result["model"]["architecture"] == "tinycore_basis_v0"
    assert len(result["tokens"]) == 8
    assert result["text"].startswith("Tiny")
