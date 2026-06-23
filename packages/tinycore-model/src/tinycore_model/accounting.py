from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _bytes(params: int) -> dict[str, float]:
    return {
        "stored_unique_weight_bytes_fp32": float(params * 4),
        "stored_unique_weight_bytes_bf16": float(params * 2),
        "stored_unique_weight_bytes_quantized_estimate": float(params * 0.5),
    }


def parameter_report(model: nn.Module) -> dict[str, Any]:
    if hasattr(model, "parameter_report"):
        report = dict(model.parameter_report())  # type: ignore[operator]
    else:
        stored = sum(p.numel() for p in model.parameters())
        report = {
            "stored_unique_parameter_count": stored,
            "effective_parameter_count_if_materialized": stored,
        }
    report.update(_bytes(int(report["stored_unique_parameter_count"])))
    return report


@torch.no_grad()
def estimate_activation_kv_memory_bytes(batch_size: int, seq_len: int, d_model: int, n_layers: int) -> dict[str, int]:
    fp32 = 4
    activation = batch_size * seq_len * d_model * fp32
    kv = batch_size * seq_len * d_model * 2 * n_layers * fp32
    return {
        "estimated_activation_bytes_fp32": activation,
        "estimated_kv_cache_bytes_fp32": kv,
        "estimated_kv_cache_bytes_per_token_fp32": batch_size * d_model * 2 * n_layers * fp32,
    }
