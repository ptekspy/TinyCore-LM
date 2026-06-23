from __future__ import annotations

import json

import torch

from tinycore_format import export_tensor_payload, verify_manifest
from tinycore_model import ModelConfig, TinyCoreLM, load_model_artifact, save_model_artifact
from tinycore_model.data import ByteTokenizer


def test_model_artifact_round_trip(tmp_path) -> None:
    torch.manual_seed(11)
    cfg = ModelConfig(
        model_type="tinycore_basis_v0",
        vocab_size=128,
        d_model=32,
        n_heads=4,
        n_virtual_layers=4,
        basis_rank=2,
        max_seq_len=16,
    )
    model = TinyCoreLM(cfg)
    model.eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    expected = model(tokens)
    dataset = {
        "name": "test_dataset",
        "source": "synthetic",
        "license_notes": "test",
        "num_documents": 1,
        "num_tokens": 32,
        "tokenizer": "byte_char_mvp",
        "train_shards": [],
        "val_shards": [],
    }
    metrics = {"val_loss": 1.0, "stored_unique_bytes_bf16": 2.0}
    manifest = save_model_artifact(
        artifact_dir=tmp_path,
        model=model,
        model_cfg=cfg,
        run_id="round_trip",
        model_name="tinycore_basis_v0",
        dataset_manifest=dataset,
        tokenizer=ByteTokenizer(),
        metrics=metrics,
        benchmark_config={"run_group": "test"},
    )

    loaded, loaded_manifest = load_model_artifact(tmp_path)
    actual = loaded(tokens)
    torch.testing.assert_close(actual, expected)
    assert loaded_manifest["format"] == "TCMDL"
    assert manifest["architecture"] == "tinycore_basis_v0"
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics


def test_export_tensor_payload_sidecar(tmp_path) -> None:
    torch.manual_seed(13)
    cfg = ModelConfig(
        model_type="tinycore_basis_v0",
        vocab_size=128,
        d_model=16,
        n_heads=4,
        n_virtual_layers=2,
        basis_rank=2,
        max_seq_len=8,
    )
    model = TinyCoreLM(cfg)
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
        model=model,
        model_cfg=cfg,
        run_id="tensor_export",
        model_name="tinycore_basis_v0",
        dataset_manifest=dataset,
        tokenizer=ByteTokenizer(),
        metrics={"val_loss": 1.0},
        benchmark_config={"run_group": "test"},
    )

    result = export_tensor_payload(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    tensor_index = json.loads((tmp_path / "tensor_index.json").read_text())

    assert result["ok"] is True
    assert result["num_tensors"] == len(manifest["sections"])
    assert result["total_bytes"] == sum(section["length"] for section in manifest["sections"])
    assert tensor_index["num_tensors"] == result["num_tensors"]
    assert manifest["files"]["tensor_index"] == "tensor_index.json"
    assert manifest["files"]["tensors"] == "tensors.bin"
    assert verify_manifest(tmp_path)["ok"] is True
