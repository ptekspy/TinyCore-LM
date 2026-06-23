from __future__ import annotations

from pathlib import Path

import pytest

from tinycore_model.data import corpus_text, dataset_manifest, make_token_splits


def test_instruction_code_corpus_is_selectable() -> None:
    text = corpus_text("tinycore_instruction_code_v0", repeat=2)
    train_tokens, val_tokens, tokenizer = make_token_splits(
        train_fraction=0.9,
        corpus_name="tinycore_instruction_code_v0",
        repeat=2,
    )
    manifest = dataset_manifest(
        train_tokens,
        val_tokens,
        tokenizer,
        corpus_name="tinycore_instruction_code_v0",
        repeat=2,
    )

    assert "Q: py add" in text
    assert "def add(a, b)" in text
    assert manifest["name"] == "tinycore_instruction_code_v0"
    assert manifest["num_documents"] > 1
    assert manifest["num_tokens"] == train_tokens.numel() + val_tokens.numel()


def test_compact_instruction_code_corpus_is_selectable() -> None:
    text = corpus_text("tinycore_instruction_code_compact_v0", repeat=1)

    assert "Q:add|A:def add(a,b):return a+b" in text
    assert "Q:json|A:" in text


def test_permuted_compact_corpus_rotates_document_order() -> None:
    text = corpus_text("tinycore_instruction_code_compact_permuted_v0", repeat=2)
    docs = [doc for doc in text.strip().split("\n\n") if doc]

    assert docs[0].startswith("Q:add|A:")
    assert docs[1].startswith("Q:rev|A:")
    assert docs[8].startswith("Q:rev|A:")
    assert docs[9].startswith("Q:len|A:")


def test_5090_jsonl_corpus_loads_from_files() -> None:
    train_tokens, val_tokens, tokenizer = make_token_splits(
        corpus_name="tinycore_instruction_code_5090_v0",
        repeat=1,
    )
    manifest = dataset_manifest(
        train_tokens,
        val_tokens,
        tokenizer,
        corpus_name="tinycore_instruction_code_5090_v0",
        repeat=1,
    )

    assert train_tokens.numel() > 7_000_000
    assert val_tokens.numel() > 700_000
    assert manifest["name"] == "tinycore_instruction_code_5090_v0"
    assert manifest["num_documents"] == 66_000
    assert manifest["repeat"] == 1
    assert manifest["train_shards"]
    assert manifest["val_shards"]


def test_typescript_github_corpus_loads_when_ingested() -> None:
    corpus_dir = Path("data/training/typescript_github_top100_v0")
    if not (corpus_dir / "train.jsonl").exists():
        pytest.skip("GitHub TypeScript corpus has not been ingested yet")

    train_tokens, val_tokens, tokenizer = make_token_splits(
        corpus_name="typescript_github_top100_v0",
        repeat=1,
    )
    manifest = dataset_manifest(
        train_tokens,
        val_tokens,
        tokenizer,
        corpus_name="typescript_github_top100_v0",
        repeat=1,
    )

    assert train_tokens.numel() > 0
    assert val_tokens.numel() > 0
    assert manifest["name"] == "typescript_github_top100_v0"
    assert manifest["train_shards"]
