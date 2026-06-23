from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


PY_FUNCS = [
    ("add", "a,b", "return a+b", "sum two numbers"),
    ("sub", "a,b", "return a-b", "subtract b from a"),
    ("mul", "a,b", "return a*b", "multiply two numbers"),
    ("safe_div", "a,b", "return None if b==0 else a/b", "divide while handling zero"),
    ("clamp", "x,lo,hi", "return min(max(x,lo),hi)", "clamp a value to a range"),
    ("rev", "s", "return s[::-1]", "reverse a string"),
    ("count", "xs", "return len(xs)", "count items"),
    ("first", "xs,default=None", "return xs[0] if xs else default", "get first item safely"),
    ("last", "xs,default=None", "return xs[-1] if xs else default", "get last item safely"),
    ("is_even", "n", "return n%2==0", "check if an integer is even"),
    ("normalize", "xs", "total=sum(xs); return xs if total==0 else [x/total for x in xs]", "normalize numeric weights"),
    ("dedupe", "xs", "seen=set(); out=[]\n    for x in xs:\n        if x not in seen:\n            seen.add(x); out.append(x)\n    return out", "deduplicate while preserving order"),
]

TS_FUNCS = [
    ("add", "a: number, b: number", "return a + b;", "sum two numbers"),
    ("clamp", "x: number, lo: number, hi: number", "return Math.min(Math.max(x, lo), hi);", "clamp a value"),
    ("isEven", "n: number", "return n % 2 === 0;", "check if a number is even"),
    ("firstOrNull", "items: string[]", "return items.length === 0 ? null : items[0];", "get first item safely"),
    ("joinWords", "words: string[]", "return words.join(' ');", "join words with spaces"),
]

TINYCORE_FACTS = [
    ("basis", "TinyCore reuses basis weights across virtual layers and routes each layer through learned coefficients."),
    ("low_rank", "Low-rank deltas add capacity on top of shared basis matrices without materializing every layer."),
    ("recurrent", "The recurrent variant carries a small state across virtual depth to improve expressiveness."),
    ("metric", "Compare validation loss, generated samples, speed, stored bytes, and quality per stored byte."),
    ("artifact", "A reproducible artifact stores config, tokenizer metadata, metrics, manifest sections, and checkpoint tensors."),
    ("native", "The native runtime loads tensor sidecars or tcmdl bundles and should match Python greedy generation."),
    ("eval", "Instruction evals should report exact prompt scores, reference completion loss, and selected checkpoint step."),
]

BUG_PATTERNS = [
    ("off_by_one", "Loop uses range(len(xs)-1) and skips the final item.", "iterate over all items or test the final boundary"),
    ("mutable_default", "Function argument items=[] keeps state across calls.", "use None as the default and allocate inside"),
    ("missing_zero_guard", "Division assumes denominator is never zero.", "return None or raise a clear error when denominator is zero"),
    ("path_traversal", "Tool accepts ../ paths from user input.", "normalize paths and reject traversal outside the repo root"),
    ("unchecked_json", "Code trusts parsed JSON fields without validation.", "validate required fields and types before use"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/training/instruction_code_5090_v0")
    parser.add_argument("--train-examples", type=int, default=60_000)
    parser.add_argument("--val-examples", type=int, default=6_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = list(generate_examples(args.train_examples, args.seed, split="train"))
    val = list(generate_examples(args.val_examples, args.seed + 10_000_000, split="val"))

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "eval_holdout.jsonl", list(generate_holdouts()))
    manifest = {
        "name": "tinycore_instruction_code_5090_v0",
        "source": "deterministic_local_generator",
        "generator": "tools/generate_large_instruction_corpus.py",
        "license_notes": "Synthetic local examples generated from repository-owned templates; no scraped or third-party text.",
        "train_examples": len(train),
        "val_examples": len(val),
        "num_documents": len(train) + len(val),
        "eval_holdout_examples": 8,
        "seed": args.seed,
        "format": "jsonl with id, split, category, prompt, response, text",
        "sha256": {
            "train": sha256_file(output_dir / "train.jsonl"),
            "val": sha256_file(output_dir / "val.jsonl"),
            "eval_holdout": sha256_file(output_dir / "eval_holdout.jsonl"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def generate_examples(count: int, seed: int, split: str) -> Iterable[dict[str, str]]:
    for index in range(count):
        selector = (index * 1103515245 + seed) % 11
        if selector in {0, 1, 2}:
            yield python_example(index, split, seed + index)
        elif selector in {3, 4}:
            yield typescript_example(index, split, seed + index)
        elif selector in {5, 6}:
            yield tinycore_example(index, split, seed + index)
        elif selector in {7, 8}:
            yield json_example(index, split, seed + index)
        elif selector == 9:
            yield bugfix_example(index, split, seed + index)
        else:
            yield command_example(index, split, seed + index)


def python_example(index: int, split: str, key: int) -> dict[str, str]:
    name, args, body, description = PY_FUNCS[key % len(PY_FUNCS)]
    style = key % 4
    if style == 0:
        prompt = f"Q:py {name}|A:"
        response = f"def {name}({args}):\n    {body}\n"
    elif style == 1:
        prompt = f"### Instruction\nWrite a Python function to {description}.\n### Response\n"
        response = f"def {name}({args}):\n    {body}\n"
    elif style == 2:
        prompt = f"Q:explain py {name}|A:"
        response = f"Use def {name}({args}) and {body.replace(chr(10), ' ')}.\n"
    else:
        prompt = f"Q:test py {name}|A:"
        response = f"assert {name} is not None\n# add boundary tests for {description}\n"
    return item(index, split, "python", prompt, response)


def typescript_example(index: int, split: str, key: int) -> dict[str, str]:
    name, args, body, description = TS_FUNCS[key % len(TS_FUNCS)]
    prompt = f"### Instruction\nWrite a TypeScript function to {description}.\n### Response\n"
    response = f"export function {name}({args}) {{\n  {body}\n}}\n"
    return item(index, split, "typescript", prompt, response)


def tinycore_example(index: int, split: str, key: int) -> dict[str, str]:
    name, fact = TINYCORE_FACTS[key % len(TINYCORE_FACTS)]
    variants = [
        (f"Q:{name}|A:", fact),
        (f"### Instruction\nExplain TinyCore {name}.\n### Response\n", fact),
        ("Q:quality per byte|A:", "Prefer the model with better eval quality per stored unique byte, not just lower parameter count."),
    ]
    prompt, response = variants[key % len(variants)]
    return item(index, split, "tinycore", prompt, response + "\n")


def json_example(index: int, split: str, key: int) -> dict[str, str]:
    payloads = [
        {"ok": True, "tokens": [1, 2, 3], "runtime": "native"},
        {"run_group": "instruction_code_5090", "device": "cuda", "seq_len": 256},
        {"model_type": "tinycore_recurrent_v0", "low_rank": 16, "basis_rank": 8},
        {"eval": "instruction_code_generalization_v0", "select_best_eval_checkpoint": True},
    ]
    payload = payloads[key % len(payloads)]
    prompt = f"Q:json {key % len(payloads)}|A:"
    response = json.dumps(payload, separators=(",", ":")) + "\n"
    return item(index, split, "json", prompt, response)


def bugfix_example(index: int, split: str, key: int) -> dict[str, str]:
    name, bug, fix = BUG_PATTERNS[key % len(BUG_PATTERNS)]
    prompt = f"### Instruction\nFind the likely bug: {bug}\n### Response\n"
    response = f"The issue is {name}. Fix: {fix}. Add a regression test before claiming success.\n"
    return item(index, split, "bugfix", prompt, response)


def command_example(index: int, split: str, key: int) -> dict[str, str]:
    commands = [
        "python3 -m pytest",
        "npm test",
        "python3 benchmarks/run_instruction_code_benchmark.py --config configs/instruction_code_5090_tinycore.yaml --output reports/runs/instruction_code_5090_report.json",
        "python3 -m tinycore_format.cli export-tensors reports/runs/instruction_code_5090_tinycore/tinycore_recurrent_5090_v0",
        "runtime/tinycore.cpp/build/tinycore-generate model.tcmdl 'Q:py add|A:' 32 0 0 1337",
    ]
    command = commands[key % len(commands)]
    prompt = "Q:command|A:"
    response = f"{command}\n"
    return item(index, split, "command", prompt, response)


def generate_holdouts() -> Iterable[dict[str, str]]:
    examples = [
        ("holdout_clamp", "Q:py clamp|A:", "def clamp(x,lo,hi):\n    return min(max(x,lo),hi)\n"),
        ("holdout_json", "Q:json training|A:", '{"run_group":"instruction_code_5090","device":"cuda"}\n'),
        ("holdout_basis", "Q:basis tradeoff|A:", "Use shared basis weights, low-rank deltas, and validation quality per stored byte.\n"),
        ("holdout_bug", "Q:bug mutable default|A:", "Use None as the default and allocate a new list inside the function.\n"),
        ("holdout_ts", "Q:ts isEven|A:", "export function isEven(n: number) {\n  return n % 2 === 0;\n}\n"),
        ("holdout_native", "Q:native verify|A:", "Compare Python and native greedy tokens for the same tcmdl artifact.\n"),
        ("holdout_eval", "Q:best checkpoint|A:", "Select the checkpoint with the highest instruction eval score, then passes, then lower reference loss.\n"),
        ("holdout_manifest", "Q:manifest contents|A:", "Store config, tokenizer, metrics, checkpoint, tensor_index, and tensors.bin.\n"),
    ]
    for index, (name, prompt, response) in enumerate(examples):
        yield {
            "id": name,
            "split": "eval_holdout",
            "category": "holdout",
            "prompt": prompt,
            "response": response,
            "text": prompt + response,
        }


def item(index: int, split: str, category: str, prompt: str, response: str) -> dict[str, str]:
    text = prompt + response
    digest = hashlib.sha256(f"{split}:{category}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
    return {
        "id": f"{split}_{category}_{index:06d}_{digest}",
        "split": split,
        "category": category,
        "prompt": prompt,
        "response": response,
        "text": text,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
