from __future__ import annotations

import time
import torch
from torch.optim import AdamW


def train(model, train_loader, val_loader, config, device):
    model.to(device)
    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    metrics = []
    start = time.time()

    for step, batch in enumerate(train_loader, start=1):
        tokens = batch.to(device)
        out = model(tokens, targets=tokens)
        loss = out["loss"]
        loss.backward()
        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            elapsed = time.time() - start
            report = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
                "elapsed_sec": elapsed,
                "tokens_per_sec": (step * config.batch_size * config.seq_len) / max(elapsed, 1e-9),
                "params": parameter_report(model),
            }
            print(report)
            metrics.append(report)

        if step >= config.max_steps:
            break
    return metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= 20:
            break
        tokens = batch.to(device)
        losses.append(model(tokens, targets=tokens)["loss"].item())
    model.train()
    return sum(losses) / len(losses)


def parameter_report(model):
    stored = sum(p.numel() for p in model.parameters())
    # Extend with module-specific reports for TinyCore.
    return {
        "stored_trainable_params": stored,
        "stored_bytes_fp32": stored * 4,
        "stored_bytes_bf16": stored * 2,
        "stored_bytes_int4_estimate": stored * 0.5,
        "stored_bytes_ternary_1_58bit_estimate": stored * 1.58 / 8,
    }
