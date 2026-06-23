from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .artifacts import load_model_artifact
from .data import ByteTokenizer


def generate_from_artifact(request: dict[str, Any]) -> dict[str, Any]:
    artifact_dir = Path(str(request["artifact_dir"]))
    prompt = str(request.get("prompt", ""))
    max_tokens = int(request.get("max_tokens", request.get("max_new_tokens", 32)))
    temperature = float(request.get("temperature", 0.8))
    seed = int(request.get("seed", 1337))

    torch.manual_seed(seed)
    tokenizer = ByteTokenizer()
    model, manifest = load_model_artifact(artifact_dir, map_location="cpu")
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(tokens, max_tokens, temperature)  # type: ignore[attr-defined]
    elapsed = time.perf_counter() - start
    token_ids = generated[0].tolist()
    return {
        "text": tokenizer.decode(token_ids),
        "tokens": token_ids,
        "metrics": {
            "tokens_per_sec": max_tokens / max(elapsed, 1e-9),
            "wall_clock_time_sec": elapsed,
        },
        "runtime": "python",
        "model": {
            "architecture": manifest["architecture"],
            "model_name": manifest.get("model_name"),
            "run_id": manifest.get("run_id"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    _ = argv
    try:
        request = json.loads(sys.stdin.read())
        response = generate_from_artifact(request)
        print(json.dumps(response))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
