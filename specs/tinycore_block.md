# TinyCore Block Spec

## Interface

```python
class TinyCoreBlock(nn.Module):
    def forward(
        self,
        x: Tensor,                 # [batch, seq, d_model]
        virtual_layer_id: int,
        family_id: int,
        recurrent_state: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        ...
```

## Block internals

```txt
x0 = x
x = x + attention(rms_norm(x), route[virtual_layer_id], family[family_id])
x = x + recurrent_mixer(rms_norm(x), state, route)
x = x + mlp(rms_norm(x), route[virtual_layer_id], family[family_id])
return x, new_state
```

## Attention path

```txt
Wq = composed_linear(kind='attn_q', layer=l, family=f)
Wk = composed_linear(kind='attn_k', layer=l, family=f)
Wv = composed_linear(kind='attn_v', layer=l, family=f)
Wo = composed_linear(kind='attn_o', layer=l, family=f)

q = x @ Wq
k = x @ Wk
v = x @ Wv
attn = causal_attention(q,k,v)
out = attn @ Wo
```

## MLP path

Preferred:

```txt
up = x @ Wup
gate = silu(x @ Wgate)
out = (gate * up) @ Wdown
```

## State mixer MVP

```txt
summary = mean_pool(x over seq)
state_candidate = tanh(summary @ Win + state @ Wh)
g = sigmoid(route_state_gate + summary @ Wg)
new_state = g * state_candidate + (1-g) * old_state
x = x + projection(new_state).unsqueeze(seq)
```

The recurrent state is not the same as KV cache. It is a compact virtual-depth state.

## Routing

Routes are learned embeddings indexed by virtual depth.

```python
route = route_table[virtual_layer_id]
```

Optional later:

- input-conditioned routing
- token-conditioned routing
- task-conditioned routing

MVP should use static learned virtual-layer routing to reduce instability.

## Parameter accounting

The block must expose:

```python
def parameter_report(self) -> dict:
    return {
        "stored_unique_params": ...,
        "effective_materialized_params": ...,
        "basis_params": ...,
        "route_params": ...,
        "low_rank_params": ...,
        "norm_params": ...,
        "recurrent_params": ...,
    }
```
