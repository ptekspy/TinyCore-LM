# TinyCore-LM Architecture Spec

## Summary

TinyCore-LM is a decoder-only language model with compositional weights. It predicts the next token like GPT-style models, but it does not store every virtual layer as unique full matrices.

Canonical forward path:

```txt
tokens
  -> token embeddings
  -> positional encoding / RoPE
  -> virtual depth loop
       -> basis-composed local causal attention
       -> recurrent state mixer
       -> basis-composed MLP
       -> residual/norm/gating
  -> final norm
  -> tied/un-tied LM head
  -> logits
```

## Core objects

### Token embedding

A normal embedding table. Later can be factorized.

```txt
E[token_id] -> vector[d_model]
```

### Virtual layer

A computational step. Virtual layers do not necessarily have unique full weights.

```txt
for depth in range(n_virtual_layers):
    family_id = depth_to_family[depth]
    route = routes[depth]
    x = TinyCoreBlock(x, family_id, route, recurrent_state)
```

### Block family

A reusable set of basis weights and core parameters.

```txt
family.attention_basis
family.mlp_basis
family.norms
family.recurrent_params
```

### Layer route

Tiny learned control vectors that tell a family how to compose effective weights for this virtual depth.

```txt
route.attn_q_coeffs
route.attn_k_coeffs
route.attn_v_coeffs
route.attn_o_coeffs
route.mlp_up_coeffs
route.mlp_gate_coeffs
route.mlp_down_coeffs
route.low_rank_scale
route.recurrent_gate
```

## Effective attention weights

For each virtual layer `l` and matrix kind `m`:

```txt
W_eff[l,m] = Σ_i alpha[l,m,i] * B[m,i] + U[l,m] @ V[l,m]
```

Where:

- `B[m,i]` are shared basis matrices.
- `alpha[l,m,i]` are small learned coefficients.
- `U @ V` is an optional low-rank correction.

Do not materialize `W_eff` if avoidable. Efficient implementations should compute composition lazily or cache per batch/step.

## Attention

MVP attention can be standard causal attention with full context for simplicity. The architecture should support local attention windows later.

Preferred MVP:

```txt
causal self-attention
multi-head
RoPE optional but recommended
KV cache optional for generation
```

Weight-composed matrices:

```txt
Q = x @ Wq_eff
K = x @ Wk_eff
V = x @ Wv_eff
O = attention(Q,K,V) @ Wo_eff
```

## MLP

Use SwiGLU-style MLP if possible:

```txt
MLP(x) = (SiLU(x @ Wgate_eff) * (x @ Wup_eff)) @ Wdown_eff
```

MVP can use GELU MLP if simpler.

## Recurrent virtual depth

A recurrent state vector allows the same block family to behave differently over repeated virtual steps.

MVP recurrent state:

```txt
state = GRU_like_update(mean_pool(x), previous_state)
x = x + gate(route, state) * state_projection(state)
```

Purpose:

- maintain information across virtual steps
- increase compute-per-weight
- test whether repeated reuse of few weights can produce useful depth

## Norm/residual

Use RMSNorm or LayerNorm.

Recommended block order:

```txt
x = x + attn(norm1(x), route)
x = x + recurrent_mixer(norm_state(x), state, route)
x = x + mlp(norm2(x), route)
```

## Model families to implement

### `tinycore_basis_v0`

- shared basis matrices
- learned routing coefficients
- no low-rank correction
- no recurrent mixer

### `tinycore_lora_v0`

- basis matrices
- routing coefficients
- low-rank per-virtual-layer deltas

### `tinycore_recurrent_v0`

- basis matrices
- routing coefficients
- low-rank deltas
- recurrent virtual-depth state

### `baseline_transformer_v0`

- plain decoder-only Transformer
- full unique layer weights
- used for comparisons

## Configuration example

```yaml
model_type: tinycore_recurrent_v0
vocab_size: 32000
d_model: 512
n_heads: 8
n_virtual_layers: 32
n_families: 4
basis_rank: 8
low_rank: 16
mlp_ratio: 4
max_seq_len: 1024
weight_precision_target: bf16_now_low_bit_later
```
