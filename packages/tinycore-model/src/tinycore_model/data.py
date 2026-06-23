from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[4]

TOY_CORPUS = """
TinyCore is a small research model.
It reuses basis weights across virtual depth.
The baseline transformer stores unique weights per layer.
The benchmark must measure loss, bytes, speed, and samples.
Correctness comes before claims.
Shared weights can fail, and that result is still useful.
"""

INSTRUCTION_CODE_DOCUMENTS = [
    """
Q: py add
A:
def add(a, b):
    return a + b
""",
    """
Q: py reverse
A:
def reverse_text(text):
    return text[::-1]
""",
    """
Q: py count tokens
A:
def count_tokens(tokens):
    return len(tokens)
""",
    """
Q: shared weights?
A:
TinyCore can reuse basis weights across virtual layers. A route chooses the
basis mixture, and optional low rank deltas add capacity.
""",
    """
Q: why benchmark?
A:
Report validation loss, generated samples, speed, stored bytes, and quality per
stored byte. Architecture claims need measurements.
""",
    """
Q: json ok
A:
{"ok": true, "tokens": [1, 2, 3]}
""",
    """
Q: fix off by one
A:
Use range(len(items)) only when the index is needed. Otherwise iterate over the
items directly and add a test for the boundary case.
""",
    """
Q: explain manifest
A:
A model artifact stores config, tokenizer metadata, metrics, manifest sections,
and checkpoint tensors so a run can be reproduced.
""",
]

COMPACT_INSTRUCTION_CODE_DOCUMENTS = [
    "Q:add|A:def add(a,b):return a+b",
    "Q:rev|A:def rev(s):return s[::-1]",
    "Q:len|A:def count(xs):return len(xs)",
    "Q:json|A:{\"ok\":true,\"tokens\":[1,2,3]}",
    "Q:basis|A:reuse basis weights across virtual layers",
    "Q:bytes|A:track val_loss and stored_bytes",
    "Q:manifest|A:save config tokenizer metrics checkpoint",
    "Q:test|A:add a boundary test before claiming success",
]

CORPORA = {
    "tinycore_builtin_char_corpus_v0": [TOY_CORPUS],
    "tinycore_instruction_code_v0": INSTRUCTION_CODE_DOCUMENTS,
    "tinycore_instruction_code_compact_v0": COMPACT_INSTRUCTION_CODE_DOCUMENTS,
    "tinycore_instruction_code_compact_permuted_v0": COMPACT_INSTRUCTION_CODE_DOCUMENTS,
}

FILE_CORPORA = {
    "tinycore_instruction_code_5090_v0": REPO_ROOT / "data" / "training" / "instruction_code_5090_v0",
    "typescript_github_top100_v0": REPO_ROOT / "data" / "training" / "typescript_github_top100_v0",
}


class ByteTokenizer:
    vocab_size = 128

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % self.vocab_size for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) if 9 <= i < 127 else "?" for i in ids)

    def manifest(self, dataset_name: str) -> dict[str, object]:
        return {
            "tokenizer_type": "byte_char_mvp",
            "vocab_size": self.vocab_size,
            "special_tokens": [],
            "training_corpus_manifest": dataset_name,
            "normalization": "ascii_mod_128",
        }


def corpus_text(corpus_name: str = "tinycore_builtin_char_corpus_v0", repeat: int = 32) -> str:
    if corpus_name in FILE_CORPORA:
        return _read_jsonl_text(FILE_CORPORA[corpus_name] / "train.jsonl") * max(1, repeat)
    if corpus_name not in CORPORA:
        known = ", ".join(sorted([*CORPORA, *FILE_CORPORA]))
        raise ValueError(f"Unknown corpus {corpus_name!r}; known corpora: {known}")
    docs = [doc.strip() for doc in CORPORA[corpus_name]]
    if corpus_name == "tinycore_instruction_code_compact_permuted_v0":
        chunks = []
        for index in range(max(1, repeat)):
            offset = index % len(docs)
            chunks.extend(docs[offset:] + docs[:offset])
        return "\n\n".join(chunks) + "\n"
    return ("\n\n".join(docs) + "\n") * max(1, repeat)


def make_token_splits(
    train_fraction: float = 0.9,
    corpus_name: str = "tinycore_builtin_char_corpus_v0",
    repeat: int = 32,
) -> tuple[Tensor, Tensor, ByteTokenizer]:
    tokenizer = ByteTokenizer()
    if corpus_name in FILE_CORPORA:
        corpus_dir = FILE_CORPORA[corpus_name]
        train_text = _read_jsonl_text(corpus_dir / "train.jsonl") * max(1, repeat)
        val_path = corpus_dir / "val.jsonl"
        if val_path.exists():
            val_text = _read_jsonl_text(val_path)
            return (
                torch.tensor(tokenizer.encode(train_text), dtype=torch.long),
                torch.tensor(tokenizer.encode(val_text), dtype=torch.long),
                tokenizer,
            )
        tokens = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
        split = max(64, int(tokens.numel() * train_fraction))
        return tokens[:split], tokens[split:], tokenizer
    text = corpus_text(corpus_name, repeat)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = max(64, int(tokens.numel() * train_fraction))
    return tokens[:split], tokens[split:], tokenizer


def dataset_manifest(
    train_tokens: Tensor,
    val_tokens: Tensor,
    tokenizer: ByteTokenizer,
    corpus_name: str = "tinycore_builtin_char_corpus_v0",
    repeat: int = 32,
) -> dict[str, object]:
    if corpus_name in FILE_CORPORA:
        corpus_dir = FILE_CORPORA[corpus_name]
        manifest_path = corpus_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return {
            "name": corpus_name,
            "source": manifest.get("source", "generated_jsonl_corpus"),
            "license_notes": manifest.get("license_notes", "Generated local corpus."),
            "num_documents": int(manifest.get("num_documents", 0)),
            "num_tokens": int(train_tokens.numel() + val_tokens.numel()),
            "tokenizer": "byte_char_mvp",
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "repeat": max(1, repeat),
            "train_shards": [str(corpus_dir / "train.jsonl")],
            "val_shards": [str(corpus_dir / "val.jsonl")] if (corpus_dir / "val.jsonl").exists() else [],
            "corpus_manifest": str(manifest_path) if manifest_path.exists() else "",
        }
    docs = CORPORA.get(corpus_name, [TOY_CORPUS])
    return {
        "name": corpus_name,
        "source": "synthetic_builtin_text",
        "license_notes": "Generated local toy corpus for architecture and instruction/code smoke tests.",
        "num_documents": len(docs) * max(1, repeat),
        "num_tokens": int(train_tokens.numel() + val_tokens.numel()),
        "tokenizer": "byte_char_mvp",
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "repeat": max(1, repeat),
        "train_shards": [],
        "val_shards": [],
    }


def sample_batch(tokens: Tensor, batch_size: int, seq_len: int, device: torch.device) -> Tensor:
    if tokens.numel() <= seq_len + 1:
        raise ValueError("Token split is too small for requested seq_len")
    ix = torch.randint(0, tokens.numel() - seq_len - 1, (batch_size,))
    batch = torch.stack([tokens[i : i + seq_len + 1] for i in ix])
    return batch.to(device)


def _read_jsonl_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Corpus shard is missing: {path}")
    texts = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if "text" in item:
            texts.append(str(item["text"]).strip())
        elif "prompt" in item and "response" in item:
            texts.append(f"{item['prompt']}{item['response']}".strip())
        else:
            raise ValueError(f"{path}:{line_number} must contain text or prompt/response")
    return "\n\n".join(texts) + "\n"
