from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .manifest import verify_manifest


def export_tensor_payload(artifact_dir: str | Path) -> dict[str, Any]:
    artifact = Path(artifact_dir)
    verified = verify_manifest(artifact)
    if not verified["ok"]:
        raise ValueError(f"cannot export tensors from invalid artifact: {verified['errors']}")

    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = torch.load(artifact / manifest["files"]["checkpoint"], map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    sections = manifest.get("sections", [])

    tensor_index: list[dict[str, Any]] = []
    offset = 0
    tensors_path = artifact / "tensors.bin"
    with tensors_path.open("wb") as output:
        for section in sections:
            name = section["name"]
            tensor = state_dict[name].detach().cpu().contiguous()
            raw = tensor.numpy().tobytes(order="C")
            digest = hashlib.sha256(raw).hexdigest()
            length = len(raw)
            if length != int(section["length"]):
                raise ValueError(f"tensor length mismatch for {name}: manifest={section['length']} actual={length}")
            if list(tensor.shape) != section["shape"]:
                raise ValueError(f"tensor shape mismatch for {name}: manifest={section['shape']} actual={list(tensor.shape)}")
            output.write(raw)
            tensor_index.append(
                {
                    "name": name,
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "shape": list(tensor.shape),
                    "offset": offset,
                    "length": length,
                    "sha256": digest,
                }
            )
            offset += length

    index = {
        "format": "TCMDL_TENSOR_INDEX",
        "format_version": "0.1.0",
        "byte_order": "little",
        "tensor_file": tensors_path.name,
        "num_tensors": len(tensor_index),
        "total_bytes": offset,
        "tensors": tensor_index,
    }
    index_path = artifact / "tensor_index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")

    manifest.setdefault("files", {})
    manifest["files"]["tensor_index"] = index_path.name
    manifest["files"]["tensors"] = tensors_path.name
    manifest["tensor_payload"] = {
        "index": index_path.name,
        "data": tensors_path.name,
        "num_tensors": len(tensor_index),
        "total_bytes": offset,
        "sha256": hashlib.sha256(tensors_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "ok": True,
        "artifact_dir": str(artifact),
        "tensor_index": str(index_path),
        "tensor_data": str(tensors_path),
        "num_tensors": len(tensor_index),
        "total_bytes": offset,
    }
