from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class ComposedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_basis: int,
        n_virtual_layers: int,
        low_rank: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_basis = n_basis
        self.n_virtual_layers = n_virtual_layers
        self.low_rank = low_rank
        self.basis = nn.Parameter(torch.empty(n_basis, in_features, out_features))
        self.coeff = nn.Parameter(torch.empty(n_virtual_layers, n_basis))
        if low_rank > 0:
            self.u = nn.Parameter(torch.empty(n_virtual_layers, in_features, low_rank))
            self.v = nn.Parameter(torch.empty(n_virtual_layers, low_rank, out_features))
        else:
            self.register_parameter("u", None)
            self.register_parameter("v", None)
        self.bias = nn.Parameter(torch.zeros(n_virtual_layers, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for matrix in self.basis:
            nn.init.xavier_uniform_(matrix)
        nn.init.normal_(self.coeff, std=0.02)
        if self.low_rank > 0:
            nn.init.normal_(self.u, std=0.02)
            nn.init.zeros_(self.v)

    def effective_weight(self, virtual_layer: int) -> Tensor:
        alpha = torch.softmax(self.coeff[virtual_layer], dim=-1)
        weight = torch.einsum("b,bij->ij", alpha, self.basis)
        if self.low_rank > 0:
            weight = weight + self.u[virtual_layer] @ self.v[virtual_layer]
        return weight

    def forward(self, x: Tensor, virtual_layer: int) -> Tensor:
        y = x @ self.effective_weight(virtual_layer)
        if self.bias is not None:
            y = y + self.bias[virtual_layer]
        return y

    def parameter_report(self) -> dict[str, int]:
        low_rank = 0 if self.low_rank == 0 else self.u.numel() + self.v.numel()
        bias = 0 if self.bias is None else self.bias.numel()
        stored = self.basis.numel() + self.coeff.numel() + low_rank + bias
        effective = self.n_virtual_layers * self.in_features * self.out_features
        return {
            "stored_unique_parameter_count": stored,
            "effective_parameter_count_if_materialized": effective,
        }


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.d_model * cfg.mlp_ratio
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: Tensor) -> Tensor:
        batch, seq = tokens.shape
        if seq > self.cfg.max_seq_len:
            tokens = tokens[:, -self.cfg.max_seq_len :]
            seq = tokens.size(1)
        pos = torch.arange(seq, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def loss(self, batch: Tensor) -> Tensor:
        logits = self(batch[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))

    @torch.no_grad()
    def generate(self, tokens: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
        for _ in range(max_new_tokens):
            logits = self(tokens[:, -self.cfg.max_seq_len :])[:, -1, :]
            next_token = _sample_next(logits, temperature)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

    def parameter_report(self) -> dict[str, int]:
        stored = sum(p.numel() for p in self.parameters())
        return {
            "stored_unique_parameter_count": stored,
            "effective_parameter_count_if_materialized": stored,
        }


class SharedTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.block = TransformerBlock(cfg)
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: Tensor) -> Tensor:
        batch, seq = tokens.shape
        if seq > self.cfg.max_seq_len:
            tokens = tokens[:, -self.cfg.max_seq_len :]
            seq = tokens.size(1)
        pos = torch.arange(seq, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None, :, :]
        for _ in range(self.cfg.n_layers):
            x = self.block(x)
        return self.lm_head(self.norm(x))

    def loss(self, batch: Tensor) -> Tensor:
        logits = self(batch[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))

    @torch.no_grad()
    def generate(self, tokens: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
        for _ in range(max_new_tokens):
            logits = self(tokens[:, -self.cfg.max_seq_len :])[:, -1, :]
            next_token = _sample_next(logits, temperature)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

    def parameter_report(self) -> dict[str, int]:
        stored = sum(p.numel() for p in self.parameters())
        shared_block_params = sum(p.numel() for p in self.block.parameters())
        effective = stored - shared_block_params + self.cfg.n_layers * shared_block_params
        return {
            "stored_unique_parameter_count": stored,
            "effective_parameter_count_if_materialized": effective,
        }


class TinyCoreBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.use_recurrent = cfg.model_type == "tinycore_recurrent_v0"
        self.q = ComposedLinear(cfg.d_model, cfg.d_model, cfg.basis_rank, cfg.n_virtual_layers, cfg.low_rank)
        self.k = ComposedLinear(cfg.d_model, cfg.d_model, cfg.basis_rank, cfg.n_virtual_layers, cfg.low_rank)
        self.v = ComposedLinear(cfg.d_model, cfg.d_model, cfg.basis_rank, cfg.n_virtual_layers, cfg.low_rank)
        self.o = ComposedLinear(cfg.d_model, cfg.d_model, cfg.basis_rank, cfg.n_virtual_layers, cfg.low_rank)
        hidden = cfg.d_model * cfg.mlp_ratio
        self.up = ComposedLinear(cfg.d_model, hidden, cfg.basis_rank, cfg.n_virtual_layers, cfg.low_rank)
        self.gate = ComposedLinear(cfg.d_model, hidden, cfg.basis_rank, cfg.n_virtual_layers, cfg.low_rank)
        self.down = ComposedLinear(hidden, cfg.d_model, cfg.basis_rank, cfg.n_virtual_layers, cfg.low_rank)
        if self.use_recurrent:
            self.state_dim = cfg.recurrent_state_dim
            self.state_cell = nn.GRUCell(cfg.d_model, self.state_dim)
            self.state_proj = nn.Linear(self.state_dim, cfg.d_model, bias=False)
            self.state_gate = nn.Parameter(torch.zeros(cfg.n_virtual_layers))

    def _heads(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: Tensor) -> Tensor:
        batch, heads, seq, dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, heads * dim)

    def forward(self, x: Tensor, virtual_layer: int, state: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        h = self.norm1(x)
        q = self._heads(self.q(h, virtual_layer))
        k = self._heads(self.k(h, virtual_layer))
        v = self._heads(self.v(h, virtual_layer))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(self._merge(attn), virtual_layer)
        if self.use_recurrent:
            if state is None:
                state = x.new_zeros(x.size(0), self.state_dim)
            state = self.state_cell(x.mean(dim=1), state)
            gate = torch.sigmoid(self.state_gate[virtual_layer])
            x = x + gate * self.state_proj(state).unsqueeze(1)
        h = self.norm2(x)
        mlp = F.silu(self.gate(h, virtual_layer)) * self.up(h, virtual_layer)
        return x + self.down(mlp, virtual_layer), state


class TinyCoreLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.block = TinyCoreBlock(cfg)
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: Tensor) -> Tensor:
        batch, seq = tokens.shape
        if seq > self.cfg.max_seq_len:
            tokens = tokens[:, -self.cfg.max_seq_len :]
            seq = tokens.size(1)
        pos = torch.arange(seq, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None, :, :]
        state = None
        for layer in range(self.cfg.n_virtual_layers):
            x, state = self.block(x, layer, state)
        return self.lm_head(self.norm(x))

    def loss(self, batch: Tensor) -> Tensor:
        logits = self(batch[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))

    @torch.no_grad()
    def generate(self, tokens: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
        for _ in range(max_new_tokens):
            logits = self(tokens[:, -self.cfg.max_seq_len :])[:, -1, :]
            next_token = _sample_next(logits, temperature)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

    def parameter_report(self) -> dict[str, Any]:
        stored = sum(p.numel() for p in self.parameters())
        composed_effective = sum(
            module.parameter_report()["effective_parameter_count_if_materialized"]
            for module in self.modules()
            if isinstance(module, ComposedLinear)
        )
        composed_stored = sum(
            module.parameter_report()["stored_unique_parameter_count"]
            for module in self.modules()
            if isinstance(module, ComposedLinear)
        )
        non_composed = stored - composed_stored
        effective = non_composed + composed_effective
        return {
            "stored_unique_parameter_count": stored,
            "effective_parameter_count_if_materialized": effective,
            "composed_stored_parameter_count": composed_stored,
            "composed_effective_parameter_count": composed_effective,
        }


def _sample_next(logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
