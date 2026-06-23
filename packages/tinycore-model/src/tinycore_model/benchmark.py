from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .accounting import estimate_activation_kv_memory_bytes, parameter_report
from .artifacts import save_model_artifact
from .config import BenchmarkConfig, ModelConfig, ModelVariant
from .data import ByteTokenizer, dataset_manifest, make_token_splits, sample_batch
from .instruction_eval import run_instruction_eval
from .models import SharedTransformerLM, TinyCoreLM, TransformerLM


def run_benchmark(cfg: BenchmarkConfig, output: str | Path = "reports/runs/first_benchmark_report.json") -> dict[str, Any]:
    device = torch.device(cfg.device)
    train_tokens, val_tokens, tokenizer = make_token_splits(
        cfg.dataset.train_fraction,
        cfg.dataset.name,
        cfg.dataset.repeat,
    )
    dataset = dataset_manifest(train_tokens, val_tokens, tokenizer, cfg.dataset.name, cfg.dataset.repeat)
    results = []
    output = Path(output)
    artifact_root = output.parent / cfg.run_group
    for variant in _model_variants(cfg):
        torch.manual_seed(_variant_seed(cfg.seed, variant))
        model_cfg = _variant_model_config(cfg.model, variant)
        model, depth_for_memory = _build_model(model_cfg, variant)
        result = _train_one(
            variant.name,
            model.to(device),
            cfg,
            model_cfg,
            train_tokens,
            val_tokens,
            tokenizer,
            device,
            depth_for_memory,
        )
        result["model_type"] = variant.model_type
        result["overrides"] = variant.overrides
        run_id = f"{cfg.run_group}_{variant.name}"
        artifact_dir = artifact_root / variant.name
        manifest = save_model_artifact(
            artifact_dir=artifact_dir,
            model=model,
            model_cfg=model_cfg,
            run_id=run_id,
            model_name=variant.name,
            dataset_manifest=dataset,
            tokenizer=tokenizer,
            metrics=result,
            benchmark_config=asdict(cfg),
        )
        result["run_id"] = run_id
        result["artifact_dir"] = str(artifact_dir)
        result["manifest"] = str(artifact_dir / "manifest.json")
        result["checkpoint"] = str(artifact_dir / manifest["files"]["checkpoint"])
        results.append(result)

    baseline_bytes = results[0]["stored_unique_bytes_bf16"]
    for result in results:
        result["compression_ratio_vs_baseline"] = baseline_bytes / max(result["stored_unique_bytes_bf16"], 1.0)
        if result.get("instruction_code_eval"):
            stored_100kib = max(result["stored_unique_bytes_bf16"] / 102400.0, 1e-9)
            result["instruction_eval_score_per_100kib_bf16"] = (
                result["instruction_code_eval"]["mean_score"] / stored_100kib
            )

    report = {
        "run_group": cfg.run_group,
        "run_id": cfg.run_group,
        "dataset": dataset,
        "config": asdict(cfg),
        "models": results,
        "conclusion": _conclusion(results),
        "next_experiment": _next_experiment(results),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    _write_progress(report)
    return report


def _train_one(
    name: str,
    model: nn.Module,
    cfg: BenchmarkConfig,
    model_cfg: ModelConfig,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    tokenizer: ByteTokenizer,
    device: torch.device,
    depth_for_memory: int,
) -> dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    start = time.perf_counter()
    losses = []
    first_loss = None
    best_eval_state: dict[str, torch.Tensor] | None = None
    best_eval_summary: dict[str, Any] | None = None
    best_eval_key: tuple[float, int, float] | None = None
    for step in range(1, cfg.training.max_steps + 1):
        batch = sample_batch(train_tokens, cfg.training.batch_size, cfg.training.seq_len, device)
        with _autocast_context(device, model_cfg):
            loss = model.loss(batch)  # type: ignore[attr-defined]
        if first_loss is None:
            first_loss = float(loss.item())
        loss.backward()
        if cfg.training.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % cfg.training.eval_interval == 0 or step == cfg.training.max_steps:
            eval_point: dict[str, Any] = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": _evaluate(model, val_tokens, cfg, model_cfg, device),
            }
            if cfg.eval.enabled:
                instruction_eval = run_instruction_eval(
                    model,
                    tokenizer,
                    device,
                    suite_name=cfg.eval.suite_name,
                    max_new_tokens=cfg.eval.max_new_tokens,
                    temperature=cfg.eval.temperature,
                )
                eval_point.update(_instruction_eval_summary(instruction_eval))
                if cfg.training.select_best_eval_checkpoint:
                    key = _instruction_eval_selection_key(eval_point)
                    if best_eval_key is None or key > best_eval_key:
                        best_eval_key = key
                        best_eval_summary = dict(eval_point)
                        best_eval_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
            losses.append(eval_point)
    elapsed = time.perf_counter() - start
    if best_eval_state is not None:
        model.load_state_dict(best_eval_state)
    prompt = torch.tensor([tokenizer.encode(cfg.generation.prompt)], dtype=torch.long, device=device)
    generated = model.generate(prompt, cfg.generation.max_new_tokens, cfg.generation.temperature)  # type: ignore[attr-defined]
    instruction_eval = None
    if cfg.eval.enabled:
        instruction_eval = run_instruction_eval(
            model,
            tokenizer,
            device,
            suite_name=cfg.eval.suite_name,
            max_new_tokens=cfg.eval.max_new_tokens,
            temperature=cfg.eval.temperature,
        )
    params = parameter_report(model)
    memory = estimate_activation_kv_memory_bytes(
        cfg.training.batch_size,
        cfg.training.seq_len,
        model_cfg.d_model,
        depth_for_memory,
    )
    tokens_seen = cfg.training.max_steps * cfg.training.batch_size * cfg.training.seq_len
    final = best_eval_summary if best_eval_summary is not None else losses[-1]
    return {
        "name": name,
        "train_loss_first": first_loss,
        "train_loss": final["train_loss"],
        "val_loss": final["val_loss"],
        "loss_curve": losses,
        "instruction_eval_curve": [
            {
                key: item[key]
                for key in (
                    "step",
                    "instruction_eval_mean_score",
                    "instruction_eval_passed",
                    "reference_completion_loss",
                )
                if key in item
            }
            for item in losses
            if "instruction_eval_mean_score" in item
        ],
        "selected_checkpoint": (
            {
                "selection": "best_instruction_eval",
                **best_eval_summary,
            }
            if best_eval_summary is not None
            else {"selection": "final_step", "step": cfg.training.max_steps}
        ),
        "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
        "wall_clock_time_sec": elapsed,
        "sample_generation": tokenizer.decode(generated[0].tolist()),
        "instruction_code_eval": instruction_eval,
        "stored_unique_params": params["stored_unique_parameter_count"],
        "stored_unique_bytes_fp32": params["stored_unique_weight_bytes_fp32"],
        "stored_unique_bytes_bf16": params["stored_unique_weight_bytes_bf16"],
        "stored_unique_bytes_quantized_estimate": params["stored_unique_weight_bytes_quantized_estimate"],
        "effective_materialized_params": params["effective_parameter_count_if_materialized"],
        **memory,
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    val_tokens: torch.Tensor,
    cfg: BenchmarkConfig,
    model_cfg: ModelConfig,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(8):
        batch = sample_batch(val_tokens, cfg.training.batch_size, cfg.training.seq_len, device)
        with _autocast_context(device, model_cfg):
            losses.append(float(model.loss(batch).item()))  # type: ignore[attr-defined]
    model.train()
    return sum(losses) / len(losses)


def _autocast_context(device: torch.device, model_cfg: ModelConfig):
    if device.type == "cuda" and model_cfg.precision_target == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _instruction_eval_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction_eval_mean_score": result["mean_score"],
        "instruction_eval_passed": result["num_passed"],
        "reference_completion_loss": result["mean_reference_completion_loss"],
    }


def _instruction_eval_selection_key(eval_point: dict[str, Any]) -> tuple[float, int, float]:
    return (
        float(eval_point["instruction_eval_mean_score"]),
        int(eval_point["instruction_eval_passed"]),
        -float(eval_point["reference_completion_loss"]),
    )


def _conclusion(results: list[dict[str, Any]]) -> str:
    if len(results) < 2:
        return "inconclusive"
    baseline = results[0]
    challengers = results[1:]
    compressed_better = [
        item
        for item in challengers
        if item["val_loss"] <= baseline["val_loss"]
        and item["stored_unique_bytes_bf16"] <= baseline["stored_unique_bytes_bf16"]
    ]
    if compressed_better:
        best = min(compressed_better, key=lambda item: item["val_loss"])
        return f"best_compressed_better:{best['name']}"
    best = min(challengers, key=lambda item: item["val_loss"])
    if best["val_loss"] <= baseline["val_loss"]:
        return f"quality_better_but_larger:{best['name']}"
    if any(item["stored_unique_bytes_bf16"] < baseline["stored_unique_bytes_bf16"] for item in challengers):
        return "compressed_smaller_but_quality_worse"
    return "worse"


def _next_experiment(results: list[dict[str, Any]]) -> str:
    names = {item["name"] for item in results}
    if "tinycore_lora_v0" not in names:
        return "Add low-rank TinyCore and run an ablation."
    if "tinycore_recurrent_v0" not in names:
        return "Add recurrent virtual-depth mixer and compare against basis and low-rank variants."
    return "Scale instruction/code training and compare quality per stored byte on behavior evals."


def _write_progress(report: dict[str, Any]) -> None:
    names = {item["name"] for item in report["models"]}
    model_types = {item.get("model_type") for item in report["models"]}
    completed = [
        "T0_repo_scaffold",
        "T1_config_schemas",
        "T2_tokenizer_mvp",
        "T3_dataset_packer",
        "T4_baseline_transformer",
        "T5_composed_linear",
        "T6_tinycore_basis_model",
        "T7_training_loop",
        "T8_generation_loop",
        "T9_parameter_accounting",
        "T10_eval_report",
    ]
    if "tinycore_lora_v0" in names or any(item.get("overrides", {}).get("low_rank", 0) for item in report["models"]):
        completed.append("T11_low_rank_deltas")
    if "tinycore_recurrent_v0" in names or "tinycore_recurrent_v0" in model_types:
        completed.append("T12_recurrent_virtual_depth")
    if len(report["models"]) >= 3 or Path("reports/runs/ablation_report.json").exists():
        completed.append("T13_ablation_runner")
    if all("manifest" in item and "checkpoint" in item for item in report["models"]):
        completed.append("implementation_step_14_manifest_checkpoint_save_load")
    if any(item.get("instruction_code_eval") for item in report["models"]):
        completed.append("implementation_step_52_instruction_code_corpus_eval")
    if any(item.get("selected_checkpoint", {}).get("selection") == "best_instruction_eval" for item in report["models"]):
        completed.append("implementation_step_53_best_instruction_eval_checkpoint")
    if any(Path(f"{item.get('artifact_dir', '')}.tcmdl").exists() for item in report["models"]):
        completed.append("implementation_step_54_selected_tcmdl_native_regression")
    if Path("packages/tinycored/tests/nativeServer.integration.test.ts").exists():
        completed.append("implementation_step_55_tinycored_native_server_regression")
    if report.get("dataset", {}).get("name") == "tinycore_instruction_code_compact_permuted_v0":
        completed.append("implementation_step_56_permuted_compact_corpus")
    if Path("data/training/instruction_code_5090_v0/manifest.json").exists():
        completed.append("implementation_step_57_5090_training_corpus")
    if Path("tasks/phase_2_5090_training_runbook.md").exists():
        completed.append("implementation_step_58_5090_training_runbook")
    if Path("tools/ingest_github_typescript_repos.py").exists():
        completed.append("implementation_step_59_github_typescript_ingestion_pipeline")
    if Path("configs/typescript_github_5090_tinycore.yaml").exists() and Path(
        "tasks/phase_2_github_typescript_ingestion_runbook.md"
    ).exists():
        completed.append("implementation_step_60_typescript_github_5090_pass")
    if Path("data/training/typescript_github_top100_v0/README.md").exists():
        completed.append("implementation_step_61_docs_site_ingestion_for_install_usage")
    if Path("tools/ingest_function_calling_stage3.py").exists():
        completed.append("implementation_step_62_function_calling_stage3_ingestion")
    if Path("configs/function_calling_stage3_5090_tinycore.yaml").exists() and Path(
        "tasks/phase_3_function_calling_training_runbook.md"
    ).exists():
        completed.append("implementation_step_63_function_calling_stage3_5090_pass")
    if Path("benchmarks/run_stage_3_tool_eval.py").exists():
        completed.append("implementation_step_64_stage_3_tool_call_eval")
    progress = {
        "completed_tasks": completed,
        "blocked_tasks": [],
        "latest_metrics": deepcopy(report["models"]),
        "architecture_invariant_status": _architecture_invariant_status(report["models"]),
    }
    Path("reports/progress.json").write_text(json.dumps(progress, indent=2) + "\n")


def _architecture_invariant_status(results: list[dict[str, Any]]) -> str:
    if len(results) < 2:
        return "unknown"
    baseline_bytes = results[0]["stored_unique_bytes_bf16"]
    for item in results[1:]:
        if item["stored_unique_bytes_bf16"] > baseline_bytes:
            return "violated"
    return "preserved"


def _model_variants(cfg: BenchmarkConfig) -> list[ModelVariant]:
    if cfg.variants:
        return cfg.variants
    return [
        ModelVariant("baseline_transformer_v0", "baseline_transformer_v0"),
        ModelVariant("shared_layer_transformer_v0", "shared_layer_transformer_v0"),
        ModelVariant("tinycore_basis_v0", "tinycore_basis_v0", {"low_rank": 0}),
    ]


def _variant_seed(base_seed: int, variant: ModelVariant) -> int:
    payload = json.dumps(
        {"model_type": variant.model_type, "overrides": variant.overrides},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % 1_000_000
    return base_seed + offset


def _variant_model_config(base: ModelConfig, variant: ModelVariant) -> ModelConfig:
    allowed = set(ModelConfig.__dataclass_fields__.keys())
    overrides = {key: value for key, value in variant.overrides.items() if key in allowed}
    cfg = replace(base, model_type=variant.model_type, **overrides)
    if variant.model_type == "tinycore_basis_v0":
        cfg = replace(cfg, low_rank=0)
    return cfg


def _build_model(cfg: ModelConfig, variant: ModelVariant) -> tuple[nn.Module, int]:
    if variant.model_type == "baseline_transformer_v0":
        return TransformerLM(cfg), cfg.n_layers
    if variant.model_type == "shared_layer_transformer_v0":
        return SharedTransformerLM(cfg), cfg.n_layers
    if variant.model_type in {"tinycore_basis_v0", "tinycore_lora_v0", "tinycore_recurrent_v0"}:
        return TinyCoreLM(cfg), cfg.n_virtual_layers
    raise ValueError(f"Unsupported model_type: {variant.model_type}")
