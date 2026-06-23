# Compositional Weights Spec

## Problem

Normal LLMs store independent matrices for every layer. This is simple but storage-heavy.

TinyCore replaces independent matrices with compositional effective matrices.

## Formula

For matrix kind `m` in virtual layer `l`:

```txt
W_eff[l,m] = compose(B[m], a[l,m]) + delta[l,m]
```

Basic composition:

```txt
compose(B, a) = Σ_i softmax(a)_i * B_i
```

Optional signed composition:

```txt
compose(B, a) = Σ_i tanh(a_i) * B_i
```

Low-rank delta:

```txt
delta[l,m] = U[l,m] @ V[l,m]
```

Where:

```txt
U shape = [d_in, r]
V shape = [r, d_out]
r << min(d_in, d_out)
```

## Stored vs effective parameters

If materialized:

```txt
n_virtual_layers * n_matrix_kinds * d_in * d_out
```

TinyCore stored:

```txt
n_matrix_kinds * n_basis * d_in * d_out
+ n_virtual_layers * n_matrix_kinds * n_basis
+ n_virtual_layers * n_matrix_kinds * (d_in*r + r*d_out)
```

The point is to make:

```txt
n_basis << n_virtual_layers
r << d_model
```

## Matrix kinds

Attention:

```txt
attn_q
attn_k
attn_v
attn_o
```

MLP:

```txt
mlp_up
mlp_gate
mlp_down
```

Optional:

```txt
state_in
state_out
router
```

## Efficient implementation options

### Option A: materialize per forward

Simplest, not final.

```python
W = torch.einsum('b,bij->ij', coeffs, basis)
y = x @ W
```

Good for MVP correctness.

### Option B: compose outputs

Avoid materializing `W`:

```python
y = Σ_i coeff_i * (x @ B_i)
```

This can be batched:

```python
basis_outputs = einsum(x, basis)
y = weighted_sum(coeffs, basis_outputs)
```

### Option C: cache effective weights

During generation, routes are fixed. Cache composed matrices per virtual layer/family/kind.

### Option D: fused low-bit kernels

Future native runtime should fuse:

```txt
low-bit basis matmul
+ coefficient weighting
+ low-rank delta
```

## Regularisation

To force basis diversity and route usefulness:

```txt
basis_orthogonality_loss
route_entropy_regularisation
low_rank_l2_penalty
basis_usage_balance_loss
```

Do not over-regularise early. First prove training stability.

## Failure modes

1. All routes collapse to same basis.
2. Low-rank deltas carry everything, defeating compression.
3. Basis matrices become redundant.
4. Training unstable due to composed matrices.
5. Effective depth fails because shared weights cannot specialise.

## Required metrics

```json
{
  "basis_usage_entropy": 0.0,
  "route_collapse_score": 0.0,
  "low_rank_parameter_fraction": 0.0,
  "stored_unique_weight_bytes": 0,
  "effective_materialized_weight_bytes": 0,
  "compression_ratio_vs_baseline": 0.0
}
```
