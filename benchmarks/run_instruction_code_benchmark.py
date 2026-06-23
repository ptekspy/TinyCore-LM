from __future__ import annotations

import argparse
import json
import os
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore", message="CUDA initialization:.*")

from tinycore_model import load_benchmark_config, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/instruction_code_toy.yaml")
    parser.add_argument("--output", default="reports/runs/instruction_code_report.json")
    args = parser.parse_args()
    report = run_benchmark(load_benchmark_config(args.config), args.output)
    print(
        json.dumps(
            {
                "run_group": report["run_group"],
                "conclusion": report["conclusion"],
                "models": [
                    {
                        "name": model["name"],
                        "val_loss": model["val_loss"],
                        "stored_unique_bytes_bf16": model["stored_unique_bytes_bf16"],
                        "tokens_per_sec": model["tokens_per_sec"],
                        "compression_ratio_vs_baseline": model["compression_ratio_vs_baseline"],
                        "instruction_eval_mean_score": (
                            model["instruction_code_eval"]["mean_score"]
                            if model["instruction_code_eval"]
                            else None
                        ),
                        "instruction_eval_passed": (
                            model["instruction_code_eval"]["num_passed"]
                            if model["instruction_code_eval"]
                            else None
                        ),
                        "reference_completion_loss": (
                            model["instruction_code_eval"]["mean_reference_completion_loss"]
                            if model["instruction_code_eval"]
                            else None
                        ),
                        "instruction_eval_score_per_100kib_bf16": model.get(
                            "instruction_eval_score_per_100kib_bf16"
                        ),
                        "selected_checkpoint": model.get("selected_checkpoint"),
                    }
                    for model in report["models"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
