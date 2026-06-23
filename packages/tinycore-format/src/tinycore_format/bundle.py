from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any


MAGIC = b"TCMDL\0"
FORMAT_VERSION = "0.1.0-bundle"
HEADER_LENGTH = struct.Struct("<Q")


def write_tcmdl_bundle(artifact_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    from .manifest import verify_manifest

    artifact = Path(artifact_dir)
    output = Path(output_path)
    verified = verify_manifest(artifact)
    if not verified["ok"]:
        raise ValueError(f"cannot bundle invalid artifact: {verified['errors']}")

    manifest = json.loads((artifact / "manifest.json").read_text())
    file_entries: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    offset = 0
    manifest_files = {"manifest": "manifest.json", **manifest["files"]}
    for name, relative_path in manifest_files.items():
        data = (artifact / relative_path).read_bytes()
        file_entries.append(
            {
                "name": name,
                "path": relative_path,
                "offset": offset,
                "length": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        payload_parts.append(data)
        offset += len(data)

    header = {
        "format": "TCMDL",
        "format_version": FORMAT_VERSION,
        "source_format_version": manifest.get("format_version"),
        "architecture": manifest.get("architecture"),
        "model_name": manifest.get("model_name"),
        "run_id": manifest.get("run_id"),
        "model": manifest.get("model", {}),
        "tensor_sections": manifest.get("sections", []),
        "files": file_entries,
        "payload_length": offset,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(HEADER_LENGTH.pack(len(header_bytes)))
        handle.write(header_bytes)
        for part in payload_parts:
            handle.write(part)
    return inspect_tcmdl_bundle(output)


def extract_tcmdl_bundle(path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    from .manifest import verify_manifest

    bundle = Path(path)
    output = Path(output_dir)
    verified = verify_tcmdl_bundle(bundle)
    if not verified["ok"]:
        raise ValueError(f"cannot extract invalid bundle: {verified['errors']}")

    header, payload_start = _read_header(bundle)
    with bundle.open("rb") as handle:
        handle.seek(payload_start)
        payload = handle.read()

    output.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    for index, entry in enumerate(header.get("files", [])):
        if not isinstance(entry, dict):
            raise ValueError(f"files[{index}] must be an object")
        relative_path = _safe_relative_path(str(entry.get("path", "")))
        offset = int(entry["offset"])
        length = int(entry["length"])
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload[offset : offset + length])
        extracted.append(relative_path.as_posix())

    artifact_check = verify_manifest(output)
    return {
        "ok": artifact_check["ok"],
        "path": str(bundle),
        "output_dir": str(output),
        "files": extracted,
        "errors": artifact_check["errors"],
    }


def inspect_tcmdl_bundle(path: str | Path) -> dict[str, Any]:
    bundle = Path(path)
    header, _payload_start = _read_header(bundle)
    return {
        "path": str(bundle),
        "format": header.get("format"),
        "format_version": header.get("format_version"),
        "source_format_version": header.get("source_format_version"),
        "architecture": header.get("architecture"),
        "model_name": header.get("model_name"),
        "run_id": header.get("run_id"),
        "num_sections": len(header.get("tensor_sections", [])),
        "total_section_bytes": sum(int(section.get("length", 0)) for section in header.get("tensor_sections", [])),
        "num_files": len(header.get("files", [])),
        "payload_length": header.get("payload_length", 0),
        "model": header.get("model", {}),
    }


def verify_tcmdl_bundle(path: str | Path) -> dict[str, Any]:
    bundle = Path(path)
    errors: list[str] = []
    try:
        header, payload_start = _read_header(bundle)
        if header.get("format") != "TCMDL":
            errors.append("header.format must be TCMDL")
        if header.get("format_version") != FORMAT_VERSION:
            errors.append(f"header.format_version must be {FORMAT_VERSION}")
        payload_length = int(header.get("payload_length", -1))
        actual_payload_length = bundle.stat().st_size - payload_start
        if payload_length != actual_payload_length:
            errors.append(f"payload length mismatch: header={payload_length} actual={actual_payload_length}")
        with bundle.open("rb") as handle:
            handle.seek(payload_start)
            payload = handle.read()
        for index, entry in enumerate(header.get("files", [])):
            if not isinstance(entry, dict):
                errors.append(f"files[{index}] must be an object")
                continue
            offset = int(entry.get("offset", -1))
            length = int(entry.get("length", -1))
            if offset < 0 or length < 0 or offset + length > len(payload):
                errors.append(f"files[{index}] has invalid offset/length")
                continue
            digest = hashlib.sha256(payload[offset : offset + length]).hexdigest()
            if digest != entry.get("sha256"):
                errors.append(f"files[{index}] sha256 mismatch")
    except (OSError, ValueError, json.JSONDecodeError, struct.error) as error:
        errors.append(str(error))
    return {"ok": len(errors) == 0, "path": str(bundle), "errors": errors}


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"unsafe bundle path: {value}")
    return path


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        magic = handle.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("file does not start with TCMDL magic")
        raw_length = handle.read(HEADER_LENGTH.size)
        if len(raw_length) != HEADER_LENGTH.size:
            raise ValueError("file is missing TCMDL header length")
        header_length = HEADER_LENGTH.unpack(raw_length)[0]
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError("file is missing TCMDL header bytes")
    return json.loads(header_bytes.decode("utf-8")), len(MAGIC) + HEADER_LENGTH.size + header_length
