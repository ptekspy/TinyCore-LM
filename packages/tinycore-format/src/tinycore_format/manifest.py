from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_FILES = ("checkpoint", "config", "tokenizer", "metrics")


def inspect_manifest(path: str | Path) -> dict[str, Any]:
    if Path(path).suffix == ".tcmdl":
        from .bundle import inspect_tcmdl_bundle

        return inspect_tcmdl_bundle(path)
    manifest_path = _manifest_path(path)
    manifest = json.loads(manifest_path.read_text())
    sections = manifest.get("sections", [])
    total_section_bytes = sum(int(section.get("length", 0)) for section in sections)
    return {
        "path": str(manifest_path),
        "format": manifest.get("format"),
        "format_version": manifest.get("format_version"),
        "architecture": manifest.get("architecture"),
        "model_name": manifest.get("model_name"),
        "run_id": manifest.get("run_id"),
        "num_sections": len(sections),
        "total_section_bytes": total_section_bytes,
        "files": manifest.get("files", {}),
        "model": manifest.get("model", {}),
    }


def verify_manifest(path: str | Path) -> dict[str, Any]:
    if Path(path).suffix == ".tcmdl":
        from .bundle import verify_tcmdl_bundle

        return verify_tcmdl_bundle(path)
    manifest_path = _manifest_path(path)
    artifact_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    if manifest.get("format") != "TCMDL":
        errors.append("manifest.format must be TCMDL")
    for field in ("format_version", "architecture", "model", "sections", "files"):
        if field not in manifest:
            errors.append(f"missing required field: {field}")
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        errors.append("files must be an object")
        files = {}
    for key in REQUIRED_FILES:
        value = files.get(key)
        if not isinstance(value, str):
            errors.append(f"files.{key} must be a string")
            continue
    for key, value in files.items():
        if not isinstance(value, str):
            errors.append(f"files.{key} must be a string")
            continue
        if not (artifact_dir / value).is_file():
            errors.append(f"missing artifact file: {value}")
    sections = manifest.get("sections", [])
    if not isinstance(sections, list):
        errors.append("sections must be a list")
        sections = []
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            errors.append(f"sections[{index}] must be an object")
            continue
        for field in ("name", "dtype", "shape", "offset", "length"):
            if field not in section:
                errors.append(f"sections[{index}] missing {field}")
    return {
        "ok": len(errors) == 0,
        "path": str(manifest_path),
        "errors": errors,
        "num_sections": len(sections),
    }


def estimate_size(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text()) or {}
    model = config.get("model", config)
    vocab_size = int(model.get("vocab_size", 128))
    d_model = int(model.get("d_model", 32))
    n_layers = int(model.get("n_layers", 1))
    n_virtual_layers = int(model.get("n_virtual_layers", n_layers))
    basis_rank = int(model.get("basis_rank", 1))
    low_rank = int(model.get("low_rank", 0))
    mlp_ratio = int(model.get("mlp_ratio", 2))
    recurrent_state_dim = int(model.get("recurrent_state_dim", 0))
    model_type = str(model.get("model_type", "tinycore_basis_v0"))

    embedding = vocab_size * d_model
    pos = int(model.get("max_seq_len", 32)) * d_model
    norms = 3 * d_model
    if model_type == "baseline_transformer_v0":
        per_layer = 4 * d_model * d_model + 2 * d_model * (d_model * mlp_ratio)
        stored = embedding + pos + norms + n_layers * per_layer
        effective = stored
    elif model_type == "shared_layer_transformer_v0":
        per_layer = 4 * d_model * d_model + 2 * d_model * (d_model * mlp_ratio)
        stored = embedding + pos + norms + per_layer
        effective = embedding + pos + norms + n_layers * per_layer
    else:
        matrix_shapes = [
            (d_model, d_model),
            (d_model, d_model),
            (d_model, d_model),
            (d_model, d_model),
            (d_model, d_model * mlp_ratio),
            (d_model, d_model * mlp_ratio),
            (d_model * mlp_ratio, d_model),
        ]
        basis = sum(basis_rank * left * right for left, right in matrix_shapes)
        routes = len(matrix_shapes) * n_virtual_layers * basis_rank
        lora = sum(n_virtual_layers * (left * low_rank + low_rank * right) for left, right in matrix_shapes)
        recurrent = 0
        if model_type == "tinycore_recurrent_v0":
            recurrent = (d_model + recurrent_state_dim) * 3 * recurrent_state_dim + 3 * recurrent_state_dim
            recurrent += recurrent_state_dim * d_model + n_virtual_layers
        stored = embedding + pos + norms + basis + routes + lora + recurrent
        effective = embedding + pos + norms + sum(n_virtual_layers * left * right for left, right in matrix_shapes) + recurrent
    return {
        "model_type": model_type,
        "stored_unique_params_estimate": stored,
        "effective_materialized_params_estimate": effective,
        "stored_unique_bytes_fp32_estimate": stored * 4,
        "stored_unique_bytes_bf16_estimate": stored * 2,
        "stored_unique_bytes_int4_estimate": stored * 0.5,
    }


def _manifest_path(path: str | Path) -> Path:
    value = Path(path)
    return value / "manifest.json" if value.is_dir() else value
