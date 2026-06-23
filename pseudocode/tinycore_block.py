from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .composed_linear import ComposedLinear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class TinyCoreBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_basis: int, n_virtual_layers: int, mlp_ratio: int = 4, low_rank: int = 0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_virtual_layers = n_virtual_layers

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.q = ComposedLinear(d_model, d_model, n_basis, n_virtual_layers, low_rank)
        self.k = ComposedLinear(d_model, d_model, n_basis, n_virtual_layers, low_rank)
        self.v = ComposedLinear(d_model, d_model, n_basis, n_virtual_layers, low_rank)
        self.o = ComposedLinear(d_model, d_model, n_basis, n_virtual_layers, low_rank)

        hidden = d_model * mlp_ratio
        self.up = ComposedLinear(d_model, hidden, n_basis, n_virtual_layers, low_rank)
        self.gate = ComposedLinear(d_model, hidden, n_basis, n_virtual_layers, low_rank)
        self.down = ComposedLinear(hidden, d_model, n_basis, n_virtual_layers, low_rank)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def causal_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # q/k/v: [b, heads, seq, head_dim]
        scale = self.head_dim ** -0.5
        scores = q @ k.transpose(-2, -1) * scale
        t = q.size(-2)
        mask = torch.triu(torch.ones(t, t, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        probs = torch.softmax(scores, dim=-1)
        return probs @ v

    def forward(self, x: Tensor, virtual_layer: int) -> Tensor:
        h = self.norm1(x)
        q = self._split_heads(self.q(h, virtual_layer))
        k = self._split_heads(self.k(h, virtual_layer))
        v = self._split_heads(self.v(h, virtual_layer))
        attn = self._merge_heads(self.causal_attention(q, k, v))
        x = x + self.o(attn, virtual_layer)

        h = self.norm2(x)
        mlp = F.silu(self.gate(h, virtual_layer)) * self.up(h, virtual_layer)
        x = x + self.down(mlp, virtual_layer)
        return x
