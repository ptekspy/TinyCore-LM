from __future__ import annotations

import json
import struct

import yaml

from tinycore_format import estimate_size, extract_tcmdl_bundle, inspect_manifest, verify_manifest, write_tcmdl_bundle


def test_manifest_inspect_and_verify(tmp_path) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    for name in ("checkpoint.pt", "config.yaml", "tokenizer.json", "metrics.json"):
        (artifact / name).write_text("{}")
    manifest = {
        "format": "TCMDL",
        "format_version": "0.1.0-python-manifest",
        "architecture": "tinycore_basis_v0",
        "model_name": "tinycore_basis_v0",
        "run_id": "test",
        "model": {"d_model": 32},
        "sections": [{"name": "w", "dtype": "float32", "shape": [2, 2], "offset": 0, "length": 16}],
        "files": {
            "checkpoint": "checkpoint.pt",
            "config": "config.yaml",
            "tokenizer": "tokenizer.json",
            "metrics": "metrics.json",
        },
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))

    inspected = inspect_manifest(artifact)
    assert inspected["architecture"] == "tinycore_basis_v0"
    assert inspected["total_section_bytes"] == 16
    assert verify_manifest(artifact)["ok"] is True


def test_manifest_verify_reports_missing_files(tmp_path) -> None:
    manifest = {
        "format": "TCMDL",
        "format_version": "0.1.0-python-manifest",
        "architecture": "tinycore_basis_v0",
        "model": {},
        "sections": [],
        "files": {"checkpoint": "checkpoint.pt", "config": "config.yaml", "tokenizer": "tokenizer.json", "metrics": "metrics.json"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = verify_manifest(tmp_path)
    assert result["ok"] is False
    assert len(result["errors"]) == 4


def test_manifest_verify_checks_optional_files(tmp_path) -> None:
    artifact = _write_artifact(tmp_path)
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["files"]["tensor_index"] = "tensor_index.json"
    (artifact / "manifest.json").write_text(json.dumps(manifest))

    result = verify_manifest(artifact)

    assert result["ok"] is False
    assert "missing artifact file: tensor_index.json" in result["errors"]


def test_estimate_size_from_config(tmp_path) -> None:
    config = {
        "model": {
            "model_type": "tinycore_lora_v0",
            "vocab_size": 128,
            "d_model": 32,
            "n_virtual_layers": 4,
            "basis_rank": 2,
            "low_rank": 4,
            "mlp_ratio": 2,
            "max_seq_len": 32,
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = estimate_size(path)
    assert result["model_type"] == "tinycore_lora_v0"
    assert result["stored_unique_params_estimate"] > 0
    assert result["effective_materialized_params_estimate"] > 0


def test_estimate_size_shared_transformer_has_larger_effective_than_stored(tmp_path) -> None:
    path = tmp_path / "shared.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_type": "shared_layer_transformer_v0",
                    "vocab_size": 128,
                    "d_model": 32,
                    "n_layers": 4,
                    "mlp_ratio": 2,
                    "max_seq_len": 32,
                }
            }
        )
    )
    result = estimate_size(path)
    assert result["model_type"] == "shared_layer_transformer_v0"
    assert result["effective_materialized_params_estimate"] > result["stored_unique_params_estimate"]


def test_tcmdl_bundle_round_trip(tmp_path) -> None:
    artifact = _write_artifact(tmp_path)
    bundle = tmp_path / "model.tcmdl"

    converted = write_tcmdl_bundle(artifact, bundle)
    inspected = inspect_manifest(bundle)
    verified = verify_manifest(bundle)

    assert converted["format_version"] == "0.1.0-bundle"
    assert inspected["architecture"] == "tinycore_basis_v0"
    assert inspected["num_files"] == 5
    assert inspected["total_section_bytes"] == 16
    assert verified["ok"] is True


def test_tcmdl_bundle_extracts_to_valid_artifact(tmp_path) -> None:
    artifact = _write_artifact(tmp_path)
    bundle = tmp_path / "model.tcmdl"
    extracted = tmp_path / "extracted"
    write_tcmdl_bundle(artifact, bundle)

    result = extract_tcmdl_bundle(bundle, extracted)

    assert result["ok"] is True
    assert sorted(result["files"]) == ["checkpoint.pt", "config.yaml", "manifest.json", "metrics.json", "tokenizer.json"]
    assert verify_manifest(extracted)["ok"] is True
    assert (extracted / "manifest.json").read_text() == (artifact / "manifest.json").read_text()


def test_tcmdl_bundle_verify_reports_checksum_mismatch(tmp_path) -> None:
    artifact = _write_artifact(tmp_path)
    bundle = tmp_path / "model.tcmdl"
    write_tcmdl_bundle(artifact, bundle)
    data = bytearray(bundle.read_bytes())
    data[-1] = data[-1] ^ 0x01
    bundle.write_bytes(data)

    result = verify_manifest(bundle)

    assert result["ok"] is False
    assert any("sha256 mismatch" in error for error in result["errors"])


def test_tcmdl_bundle_extract_rejects_unsafe_paths(tmp_path) -> None:
    artifact = _write_artifact(tmp_path)
    bundle = tmp_path / "model.tcmdl"
    write_tcmdl_bundle(artifact, bundle)
    data = bytearray(bundle.read_bytes())
    header_length_offset = len(b"TCMDL\0")
    header_length = struct.unpack("<Q", data[header_length_offset : header_length_offset + 8])[0]
    header_start = header_length_offset + 8
    header = json.loads(data[header_start : header_start + header_length].decode("utf-8"))
    header["files"][0]["path"] = "../manifest.json"
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    rewritten = bytearray()
    rewritten.extend(b"TCMDL\0")
    rewritten.extend(struct.pack("<Q", len(header_bytes)))
    rewritten.extend(header_bytes)
    rewritten.extend(data[header_start + header_length :])
    unsafe_bundle = tmp_path / "unsafe.tcmdl"
    unsafe_bundle.write_bytes(rewritten)

    try:
        extract_tcmdl_bundle(unsafe_bundle, tmp_path / "unsafe-out")
    except ValueError as error:
        assert "unsafe bundle path" in str(error)
    else:
        raise AssertionError("unsafe bundle path was not rejected")


def _write_artifact(tmp_path):
    artifact = tmp_path / "model"
    artifact.mkdir()
    for name in ("checkpoint.pt", "config.yaml", "tokenizer.json", "metrics.json"):
        (artifact / name).write_text("{}")
    manifest = {
        "format": "TCMDL",
        "format_version": "0.1.0-python-manifest",
        "architecture": "tinycore_basis_v0",
        "model_name": "tinycore_basis_v0",
        "run_id": "test",
        "model": {"d_model": 32},
        "sections": [{"name": "w", "dtype": "float32", "shape": [2, 2], "offset": 0, "length": 16}],
        "files": {
            "checkpoint": "checkpoint.pt",
            "config": "config.yaml",
            "tokenizer": "tokenizer.json",
            "metrics": "metrics.json",
        },
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    return artifact
