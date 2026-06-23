from __future__ import annotations

import argparse
import json
import sys

from .bundle import extract_tcmdl_bundle, write_tcmdl_bundle
from .manifest import estimate_size, inspect_manifest, verify_manifest
from .tensor_payload import export_tensor_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinycore-format")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("path")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("path")
    estimate_parser = sub.add_parser("estimate-size")
    estimate_parser.add_argument("config")
    convert_parser = sub.add_parser("convert")
    convert_parser.add_argument("artifact_dir")
    convert_parser.add_argument("output")
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("bundle")
    extract_parser.add_argument("output_dir")
    export_tensors_parser = sub.add_parser("export-tensors")
    export_tensors_parser.add_argument("artifact_dir")
    args = parser.parse_args(argv)

    if args.command == "inspect":
        result = inspect_manifest(args.path)
    elif args.command == "verify":
        result = verify_manifest(args.path)
    elif args.command == "estimate-size":
        result = estimate_size(args.config)
    elif args.command == "convert":
        result = write_tcmdl_bundle(args.artifact_dir, args.output)
    elif args.command == "extract":
        result = extract_tcmdl_bundle(args.bundle, args.output_dir)
    elif args.command == "export-tensors":
        result = export_tensor_payload(args.artifact_dir)
    else:
        parser.error(f"unknown command: {args.command}")

    print(json.dumps(result, indent=2))
    if args.command == "verify" and not result["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
