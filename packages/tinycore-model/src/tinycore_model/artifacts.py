from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .config import ModelConfig
from .data import ByteTokenizer
from .models import SharedTransformerLM, TinyCoreLM, TransformerLM


def save_model_artifact(
    *,
    artifact_dir: str | Path,
    model: nn.Module,
    model_cfg: ModelConfig,
    run_id: str,
    model_name: str,
    dataset_manifest: dict[str, Any],
    tokenizer: ByteTokenizer,
    metrics: dict[str, Any],
    benchmark_config: dict[str, Any],
) -> dict[str, Any]:
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = path / "checkpoint.pt"
    manifest_path = path / "manifest.json"
    config_path = path / "config.yaml"
    tokenizer_path = path / "tokenizer.json"
    metrics_path = path / "metrics.json"

    torch.save({"state_dict": model.state_dict(), "model_config": asdict(model_cfg)}, checkpoint_path)
    config_path.write_text(yaml.safe_dump(benchmark_config, sort_keys=False))
    tokenizer_path.write_text(json.dumps(tokenizer.manifest(dataset_manifest["name"]), indent=2) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    manifest = {
        "format": "TCMDL",
        "format_version": "0.1.0-python-manifest",
        "architecture": model_cfg.model_type,
        "run_id": run_id,
        "model_name": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": asdict(model_cfg),
        "dataset": dataset_manifest,
        "tokenizer": {
            "type": "byte_char_mvp",
            "vocab_size": tokenizer.vocab_size,
            "section": "tokenizer.json",
        },
        "quantization": {
            "basis": model_cfg.precision_target,
            "routes": model_cfg.precision_target,
            "low_rank": model_cfg.precision_target,
        },
        "sections": _tensor_sections(model),
        "files": {
            "checkpoint": checkpoint_path.name,
            "config": config_path.name,
            "tokenizer": tokenizer_path.name,
            "metrics": metrics_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_model_artifact(artifact_dir: str | Path, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    path = Path(artifact_dir)
    manifest = json.loads((path / "manifest.json").read_text())
    checkpoint = torch.load(path / manifest["files"]["checkpoint"], map_location=map_location, weights_only=True)
    cfg = ModelConfig(**checkpoint["model_config"])
    model: nn.Module
    if cfg.model_type == "baseline_transformer_v0":
        model = TransformerLM(cfg)
    elif cfg.model_type == "shared_layer_transformer_v0":
        model = SharedTransformerLM(cfg)
    elif cfg.model_type in {"tinycore_basis_v0", "tinycore_lora_v0", "tinycore_recurrent_v0"}:
        model = TinyCoreLM(cfg)
    else:
        raise ValueError(f"Unsupported architecture: {cfg.model_type}")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, manifest


def _tensor_sections(model: nn.Module) -> list[dict[str, Any]]:
    sections = []
    offset = 0
    for name, tensor in model.state_dict().items():
        length = tensor.numel() * tensor.element_size()
        sections.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
                "offset": offset,
                "length": length,
            }
        )
        offset += length
    return sections
