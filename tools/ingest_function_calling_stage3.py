from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets"

SECRET_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bASIA[0-9A-Z]{16}\b",
        r"\bghp_[A-Za-z0-9_]{30,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{40,}\b",
        r"\bsk-[A-Za-z0-9]{20,}\b",
        r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b",
        r"(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]",
    ]
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    license_note: str
    gated: bool = False
    explicit_files: tuple[str, ...] = ()
    file_prefix: str = ""
    skip_demo_files: bool = True


DATASET_SPECS = [
    DatasetSpec(
        dataset_id="Salesforce/xlam-function-calling-60k",
        display_name="xLAM function calling 60k",
        license_note="Hugging Face metadata reports license: cc-by-4.0; access is gated/auto-approved by the dataset owner.",
        gated=True,
        explicit_files=("xlam_function_calling_60k.json",),
    ),
    DatasetSpec(
        dataset_id="glaiveai/glaive-function-calling-v2",
        display_name="Glaive function calling v2",
        license_note="Hugging Face metadata reports license: apache-2.0.",
        explicit_files=("glaive-function-calling-v2.json",),
    ),
    DatasetSpec(
        dataset_id="MCPToolBench/MCPToolBenchPP",
        display_name="MCPToolBench++",
        license_note="Use public benchmark rows from the Hugging Face dataset card/repository; audit upstream license before redistribution.",
        file_prefix="data/",
        skip_demo_files=True,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/training/function_calling_stage3_v0")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument("--max-total-rows", type=int, default=0)
    parser.add_argument("--max-record-chars", type=int, default=24_000)
    parser.add_argument("--include-demo-files", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    token = args.hf_token or None

    datasets = [dataset_plan(spec, args.revision, token, include_demo_files=args.include_demo_files) for spec in DATASET_SPECS]
    (output_dir / "dataset_plan.json").write_text(json.dumps(datasets, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        manifest = manifest_base(args, datasets, train_rows=0, val_rows=0, skipped=[])
        manifest["dry_run"] = True
        write_manifest(output_dir, manifest)
        print(json.dumps(manifest, indent=2))
        return

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    skipped: list[dict[str, object]] = []
    rows_by_dataset: dict[str, int] = {}
    files_by_dataset: dict[str, list[str]] = {}
    seen_hashes: set[str] = set()
    train_rows = 0
    val_rows = 0
    total_rows = 0

    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open("w", encoding="utf-8") as val_handle:
        for plan in datasets:
            dataset_id = str(plan["dataset_id"])
            rows_by_dataset[dataset_id] = 0
            files_by_dataset[dataset_id] = []
            for filename in plan["files"]:
                if args.max_total_rows and total_rows >= args.max_total_rows:
                    break
                if args.max_rows_per_dataset and rows_by_dataset[dataset_id] >= args.max_rows_per_dataset:
                    break
                try:
                    raw_path = download_dataset_file(
                        dataset_id,
                        str(filename),
                        raw_dir,
                        revision=args.revision,
                        token=token,
                        sleep=args.sleep,
                    )
                    file_rows = read_json_records(raw_path)
                    files_by_dataset[dataset_id].append(str(filename))
                except Exception as error:  # noqa: BLE001 - record and keep ingesting other datasets
                    skipped.append({"dataset": dataset_id, "file": filename, "reason": "download_or_parse_error", "error": str(error)})
                    continue
                for index, record in enumerate(file_rows):
                    if args.max_total_rows and total_rows >= args.max_total_rows:
                        break
                    if args.max_rows_per_dataset and rows_by_dataset[dataset_id] >= args.max_rows_per_dataset:
                        break
                    row = make_training_row(dataset_id, str(filename), index, record, args.max_record_chars)
                    if row is None:
                        skipped.append({"dataset": dataset_id, "file": filename, "row": index, "reason": "filtered_record"})
                        continue
                    text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                    if text_hash in seen_hashes:
                        continue
                    seen_hashes.add(text_hash)
                    is_val = stable_fraction(row["id"]) < args.val_fraction
                    handle = val_handle if is_val else train_handle
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                    val_rows += 1 if is_val else 0
                    train_rows += 0 if is_val else 1
                    rows_by_dataset[dataset_id] += 1
                    total_rows += 1
            if args.max_total_rows and total_rows >= args.max_total_rows:
                break

    write_holdout(output_dir)
    if not args.keep_raw:
        for path in raw_dir.glob("*.json"):
            path.unlink(missing_ok=True)
        try:
            raw_dir.rmdir()
        except OSError:
            pass

    manifest = manifest_base(args, datasets, train_rows=train_rows, val_rows=val_rows, skipped=skipped)
    manifest["rows_by_dataset"] = rows_by_dataset
    manifest["files_by_dataset"] = files_by_dataset
    manifest["sha256"] = {
        "train": sha256_file(train_path),
        "val": sha256_file(val_path),
        "eval_holdout": sha256_file(output_dir / "eval_holdout.jsonl"),
    }
    write_manifest(output_dir, manifest)
    print(json.dumps(manifest, indent=2))


def dataset_plan(spec: DatasetSpec, revision: str, token: str | None, include_demo_files: bool) -> dict[str, object]:
    metadata = hf_dataset_metadata(spec.dataset_id, token=token)
    siblings = [item.get("rfilename", "") for item in metadata.get("siblings", [])]
    if spec.explicit_files:
        files = [filename for filename in spec.explicit_files if filename in siblings or filename.endswith(".json")]
    else:
        files = [
            filename
            for filename in siblings
            if filename.startswith(spec.file_prefix)
            and filename.endswith(".json")
            and (include_demo_files or not should_skip_demo_file(filename, spec))
        ]
    return {
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "revision": revision,
        "gated": metadata.get("gated", spec.gated),
        "private": metadata.get("private", False),
        "sha": metadata.get("sha", ""),
        "license_note": spec.license_note,
        "tags": metadata.get("tags", []),
        "files": sorted(files),
    }


def should_skip_demo_file(filename: str, spec: DatasetSpec) -> bool:
    return spec.skip_demo_files and "demo" in Path(filename).name.lower()


def hf_dataset_metadata(dataset_id: str, token: str | None) -> dict[str, object]:
    return hf_json(f"{HF_API}/{dataset_id}", token=token)


def hf_json(url: str, token: str | None) -> dict[str, object]:
    request = urllib.request.Request(url, headers=hf_headers(token))
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Hugging Face API request failed {error.code}: {detail}") from error


def download_dataset_file(
    dataset_id: str,
    filename: str,
    raw_dir: Path,
    revision: str,
    token: str | None,
    sleep: float,
) -> Path:
    safe_name = hashlib.sha256(f"{dataset_id}:{filename}:{revision}".encode("utf-8")).hexdigest()[:16]
    raw_path = raw_dir / f"{safe_name}.json"
    if raw_path.exists() and raw_path.stat().st_size > 0:
        return raw_path
    quoted_file = urllib.parse.quote(filename)
    url = f"{HF_RESOLVE}/{dataset_id}/resolve/{revision}/{quoted_file}"
    request = urllib.request.Request(url, headers=hf_headers(token))
    try:
        with urllib.request.urlopen(request, timeout=240) as response, raw_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"dataset file download failed {error.code}: {detail}") from error
    time.sleep(sleep)
    return raw_path


def hf_headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "TinyCore-LM-stage3-ingestor",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def read_json_records(path: Path) -> list[object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "rows", "examples", "instances"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise ValueError(f"Unsupported JSON payload in {path}: {type(payload).__name__}")


def make_training_row(
    dataset_id: str,
    filename: str,
    row_index: int,
    record: object,
    max_record_chars: int,
) -> dict[str, str] | None:
    record_text = record_to_training_text(record)
    if not record_text or len(record_text.strip()) < 40:
        return None
    if len(record_text) > max_record_chars:
        record_text = record_text[:max_record_chars].rstrip() + "\n[TRUNCATED]\n"
    if any(pattern.search(record_text) for pattern in SECRET_PATTERNS):
        return None
    row_id = hashlib.sha256(f"{dataset_id}:{filename}:{row_index}:{record_text}".encode("utf-8")).hexdigest()[:24]
    prompt = (
        "### Stage\nfunction_calling_stage3_v0\n"
        f"### Dataset\n{dataset_id}\n"
        f"### Source File\n{filename}\n"
        "### Record\n"
    )
    response = record_text.strip() + "\n"
    return {
        "id": f"function_calling_stage3_{row_id}",
        "split": "train_or_val",
        "category": "function_calling_tool_use",
        "source_kind": "huggingface_dataset",
        "dataset": dataset_id,
        "source_file": filename,
        "prompt": prompt,
        "response": response,
        "text": prompt + response,
    }


def record_to_training_text(record: object) -> str:
    if isinstance(record, dict):
        if isinstance(record.get("system"), str) and isinstance(record.get("chat"), str):
            return f"{record['system'].strip()}\n\n{record['chat'].strip()}"
        if "tools" in record and ("messages" in record or "conversations" in record):
            return canonical_json(record)
        if "query" in record and ("tools" in record or "answers" in record or "answer" in record):
            return canonical_json(record)
        if "instruction" in record and ("output" in record or "response" in record):
            return canonical_json(record)
        return canonical_json(record)
    if isinstance(record, str):
        return record
    return canonical_json(record)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2)


def stable_fraction(value: str) -> float:
    number = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)
    return number / float(0xFFFFFFFFFFFF)


def write_holdout(output_dir: Path) -> None:
    rows = [
        {
            "id": "function_stage3_holdout_weather_call",
            "split": "eval_holdout",
            "category": "function_calling_tool_use",
            "prompt": "Q:function call weather|A:",
            "response": '<functioncall> {"name":"get_weather","arguments":{"location":"London"}}\n',
            "text": 'Q:function call weather|A:<functioncall> {"name":"get_weather","arguments":{"location":"London"}}\n',
        },
        {
            "id": "function_stage3_holdout_tool_schema",
            "split": "eval_holdout",
            "category": "function_calling_tool_use",
            "prompt": "Q:tool schema rule|A:",
            "response": "Use the provided tool name and required JSON arguments; do not invent unavailable tools.\n",
            "text": "Q:tool schema rule|A:Use the provided tool name and required JSON arguments; do not invent unavailable tools.\n",
        },
        {
            "id": "function_stage3_holdout_mcp",
            "split": "eval_holdout",
            "category": "function_calling_tool_use",
            "prompt": "Q:mcp tool call|A:",
            "response": "Choose the MCP server tool whose input_schema matches the task, then emit valid arguments.\n",
            "text": "Q:mcp tool call|A:Choose the MCP server tool whose input_schema matches the task, then emit valid arguments.\n",
        },
    ]
    with (output_dir / "eval_holdout.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def manifest_base(
    args: argparse.Namespace,
    datasets: list[dict[str, object]],
    train_rows: int,
    val_rows: int,
    skipped: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "name": "function_calling_stage3_v0",
        "source": "Hugging Face public/gated function-calling and MCP tool-use datasets",
        "datasets": datasets,
        "license_notes": [
            "Audit upstream dataset cards before redistribution.",
            "xLAM is gated/auto-approved on Hugging Face and requires accepted terms plus HF_TOKEN for raw downloads.",
            "MCPToolBench++ metadata does not expose a standard license in the API response at setup time.",
        ],
        "train_rows": train_rows,
        "val_rows": val_rows,
        "num_documents": train_rows + val_rows,
        "filters": {
            "val_fraction": args.val_fraction,
            "max_rows_per_dataset": args.max_rows_per_dataset,
            "max_total_rows": args.max_total_rows,
            "max_record_chars": args.max_record_chars,
            "include_demo_files": args.include_demo_files,
            "secret_patterns": len(SECRET_PATTERNS),
        },
        "skipped": skipped[:500],
        "num_skipped": len(skipped),
        "format": "jsonl with id, split, category, source_kind, dataset, source_file, prompt, response, text",
    }


def write_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
