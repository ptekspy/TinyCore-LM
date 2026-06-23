from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ComposedLinear(nn.Module):
    """
    Reference implementation, prioritising correctness.

    Effective matrix for virtual layer l:
        W_eff[l] = sum_i coeff[l,i] * basis[i] + U[l] @ V[l]

    Shapes:
        basis: [n_basis, in_features, out_features]
        coeff: [n_virtual_layers, n_basis]
        U: [n_virtual_layers, in_features, low_rank]
        V: [n_virtual_layers, low_rank, out_features]
    """

    def __init__(self, in_features: int, out_features: int, n_basis: int, n_virtual_layers: int, low_rank: int = 0, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_basis = n_basis
        self.n_virtual_layers = n_virtual_layers
        self.low_rank = low_rank

        self.basis = nn.Parameter(torch.empty(n_basis, in_features, out_features))
        self.coeff = nn.Parameter(torch.zeros(n_virtual_layers, n_basis))
        if low_rank > 0:
            self.U = nn.Parameter(torch.empty(n_virtual_layers, in_features, low_rank))
            self.V = nn.Parameter(torch.empty(n_virtual_layers, low_rank, out_features))
        else:
            self.register_parameter("U", None)
            self.register_parameter("V", None)
        self.bias = nn.Parameter(torch.zeros(n_virtual_layers, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.n_basis):
            nn.init.xavier_uniform_(self.basis[i])
        if self.low_rank > 0:
            nn.init.normal_(self.U, std=0.02)
            nn.init.zeros_(self.V)
        nn.init.normal_(self.coeff, std=0.02)

    def effective_weight(self, virtual_layer: int) -> Tensor:
        a = torch.softmax(self.coeff[virtual_layer], dim=-1)
        w = torch.einsum("b,bij->ij", a, self.basis)
        if self.low_rank > 0:
            w = w + self.U[virtual_layer] @ self.V[virtual_layer]
        return w

    def forward(self, x: Tensor, virtual_layer: int) -> Tensor:
        w = self.effective_weight(virtual_layer)
        y = x @ w
        if self.bias is not None:
            y = y + self.bias[virtual_layer]
        return y

    def parameter_report(self) -> dict:
        basis = self.basis.numel()
        routes = self.coeff.numel()
        low_rank = 0 if self.low_rank == 0 else self.U.numel() + self.V.numel()
        bias = 0 if self.bias is None else self.bias.numel()
        effective = self.n_virtual_layers * self.in_features * self.out_features
        return {
            "basis_params": basis,
            "route_params": routes,
            "low_rank_params": low_rank,
            "bias_params": bias,
            "stored_unique_params": basis + routes + low_rank + bias,
            "effective_materialized_params": effective,
        }
