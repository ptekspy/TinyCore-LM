from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
NATIVE_GENERATOR = ROOT / "runtime" / "tinycore.cpp" / "build" / "tinycore-generate"
ARTIFACT_DIR = ROOT / "reports" / "runs" / "ablation_toy" / "tinycore_recurrent_v0_extracted"
BUNDLE = ROOT / "reports" / "runs" / "ablation_toy" / "tinycore_recurrent_v0.tcmdl"
PERMUTED_TINYCORE_LR4_BUNDLE = (
    ROOT / "reports" / "runs" / "instruction_code_permuted_tinycore" / "tinycore_recurrent_lr4_state8.tcmdl"
)
EXPECTED_TEXT = "TinyCoreeeerree "
EXPECTED_NEW_TOKENS = [101, 101, 101, 114, 114, 101, 101, 32]


def test_native_generator_matches_reference_artifact_dir() -> None:
    _require_native_fixture(ARTIFACT_DIR)

    result = _run_native_generate(ARTIFACT_DIR)

    assert result["text"] == EXPECTED_TEXT
    assert result["new_tokens"] == EXPECTED_NEW_TOKENS
    assert result["runtime"] == "native"


def test_native_generator_matches_reference_bundle() -> None:
    _require_native_fixture(BUNDLE)

    result = _run_native_generate(BUNDLE)

    assert result["text"] == EXPECTED_TEXT
    assert result["new_tokens"] == EXPECTED_NEW_TOKENS
    assert result["model"]["architecture"] == "tinycore_recurrent_v0"


def test_native_generator_matches_permuted_tinycore_compact_add_bundle() -> None:
    _require_native_fixture(PERMUTED_TINYCORE_LR4_BUNDLE)

    result = _run_native_generate(PERMUTED_TINYCORE_LR4_BUNDLE, "Q:add|A:", 16)

    assert result["text"] == "Q:add|A:def add(a,b):ret"
    assert result["new_tokens"] == [100, 101, 102, 32, 97, 100, 100, 40, 97, 44, 98, 41, 58, 114, 101, 116]
    assert result["model"]["model_name"] == "tinycore_recurrent_lr4_state8"


def test_native_generator_matches_permuted_tinycore_compact_json_bundle() -> None:
    _require_native_fixture(PERMUTED_TINYCORE_LR4_BUNDLE)

    result = _run_native_generate(PERMUTED_TINYCORE_LR4_BUNDLE, "Q:json|A:", 16)

    assert result["text"] == 'Q:json|A:{"ok":true,"toke'
    assert result["new_tokens"] == [123, 34, 111, 107, 34, 58, 116, 114, 117, 101, 44, 34, 116, 111, 107, 101]
    assert result["model"]["model_name"] == "tinycore_recurrent_lr4_state8"


def test_native_generator_matches_permuted_tinycore_compact_basis_bundle() -> None:
    _require_native_fixture(PERMUTED_TINYCORE_LR4_BUNDLE)

    result = _run_native_generate(PERMUTED_TINYCORE_LR4_BUNDLE, "Q:basis|A:", 16)

    assert result["text"] == "Q:basis|A:reuse basis weig"
    assert result["new_tokens"] == [114, 101, 117, 115, 101, 32, 98, 97, 115, 105, 115, 32, 119, 101, 105, 103]
    assert result["model"]["model_name"] == "tinycore_recurrent_lr4_state8"


def _require_native_fixture(artifact: Path) -> None:
    if not NATIVE_GENERATOR.exists():
        pytest.skip("native generator has not been built")
    if not artifact.exists():
        pytest.skip(f"reference artifact is missing: {artifact}")


def _run_native_generate(artifact: Path, prompt: str = "TinyCore", max_tokens: int = 8) -> dict:
    completed = subprocess.run(
        [str(NATIVE_GENERATOR), str(artifact), prompt, str(max_tokens)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)
