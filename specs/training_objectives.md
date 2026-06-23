# Training Objectives

## Base objective

Standard autoregressive next-token prediction.

Given token sequence:

```txt
x[0], x[1], ..., x[n]
```

Train model to predict:

```txt
x[t+1] from x[:t]
```

Loss:

```txt
cross_entropy(logits[:, :-1], tokens[:, 1:])
```

## Regularisation objectives

Optional, not enabled by default until base training works.

### Basis diversity

Encourage basis matrices not to collapse into identical copies.

```txt
L_basis_orthogonal = mean cosine similarity between flattened basis matrices
```

### Route usage entropy

Encourage routes to use more than one basis.

```txt
L_route_entropy = -entropy(softmax(route_coeffs))
```

Careful: too much entropy pressure can prevent specialisation.

### Low-rank budget pressure

Prevent low-rank deltas becoming the whole model.

```txt
L_low_rank = lambda * ||U||^2 + ||V||^2
```

### Effective matrix smoothness

Neighbouring virtual layers can be encouraged to vary smoothly early in training.

```txt
L_route_smooth = ||route[l] - route[l-1]||
```

## Instruction tuning later

After base model works:

```txt
messages -> serialized chat tokens -> next-token loss
```

## Tool-call tuning later

Use structured assistant messages:

```json
{"type":"tool_call","tool":"read_file","args":{"path":"src/index.ts"}}
```

The model learns to emit tool calls as text or structured tokens.

## Important

Do not start with RLHF. Do not start with agent training. First train the base architecture.
