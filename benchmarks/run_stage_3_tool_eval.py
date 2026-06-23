from __future__ import annotations

import argparse
import json

from tinycore_model.tool_eval import run_tool_eval_for_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", default="reports/runs/stage_3_tool_eval_report.json")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    report = run_tool_eval_for_artifact(
        args.artifact_dir,
        output=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(json.dumps({"output": args.output, "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
