from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .tinycore_block import TinyCoreBlock, RMSNorm


class TinyCoreLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)  # replace with RoPE later
        self.block = TinyCoreBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_basis=config.basis_rank,
            n_virtual_layers=config.n_virtual_layers,
            mlp_ratio=config.mlp_ratio,
            low_rank=config.low_rank,
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if getattr(config, "tie_embeddings", True):
            self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: Tensor, targets: Tensor | None = None) -> dict:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None, :, :]
        for layer in range(self.config.n_virtual_layers):
            x = self.block(x, layer)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1))
        return {"logits": logits, "loss": loss}

    def generate(self, tokens: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
        for _ in range(max_new_tokens):
            idx = tokens[:, -self.config.max_seq_len:]
            logits = self(idx)["logits"][:, -1, :]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
