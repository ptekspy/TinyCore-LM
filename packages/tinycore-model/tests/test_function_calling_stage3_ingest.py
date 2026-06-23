from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "ingest_function_calling_stage3.py"
spec = importlib.util.spec_from_file_location("ingest_function_calling_stage3", SCRIPT)
assert spec is not None and spec.loader is not None
ingest = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ingest
spec.loader.exec_module(ingest)


def test_glaive_record_preserves_system_and_chat() -> None:
    record = {
        "system": "SYSTEM: Use this function.\n{\"name\":\"search\"}",
        "chat": "USER: Search docs\n\nASSISTANT: <functioncall> {\"name\":\"search\"}",
    }

    text = ingest.record_to_training_text(record)

    assert "SYSTEM: Use this function" in text
    assert "USER: Search docs" in text
    assert "<functioncall>" in text


def test_generic_records_are_canonical_json() -> None:
    record = {"tools": [{"name": "read_file"}], "messages": [{"role": "user", "content": "read package"}]}

    text = ingest.record_to_training_text(record)

    assert '"messages": [' in text
    assert '"name": "read_file"' in text
    assert '"tools": [' in text


def test_make_training_row_wraps_dataset_metadata() -> None:
    row = ingest.make_training_row(
        "MCPToolBench/MCPToolBenchPP",
        "data/search/search_0725_single_v2.json",
        7,
        {"tools": [{"name": "web_search"}], "query": "search for docs"},
        max_record_chars=1000,
    )

    assert row is not None
    assert row["category"] == "function_calling_tool_use"
    assert row["source_kind"] == "huggingface_dataset"
    assert row["dataset"] == "MCPToolBench/MCPToolBenchPP"
    assert "### Stage\nfunction_calling_stage3_v0" in row["text"]
    assert "web_search" in row["text"]


def test_make_training_row_filters_secret_like_records() -> None:
    row = ingest.make_training_row(
        "example/dataset",
        "rows.json",
        0,
        {"chat": "token = 'ghp_abcdefghijklmnopqrstuvwxyzABCDE'"},
        max_record_chars=1000,
    )

    assert row is None


def test_demo_file_policy_skips_mcp_demos_by_default() -> None:
    spec = ingest.DatasetSpec("MCPToolBench/MCPToolBenchPP", "MCP", "license", file_prefix="data/")

    assert ingest.should_skip_demo_file("data/browser/browser_single_demo.json", spec) is True
    assert ingest.should_skip_demo_file("data/browser/browser_0724_single_v3.json", spec) is False
