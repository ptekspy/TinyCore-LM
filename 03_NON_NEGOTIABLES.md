# Non-Negotiables

## Research honesty

This project is ambitious and unproven. Treat it as a serious research programme. Every claim must be backed by a reproducible experiment.

## Architecture invariant

TinyCore-LM must have fewer unique stored weights than an equivalent-depth Transformer.

Allowed:

```txt
shared basis weights
per-layer routing coefficients
small low-rank deltas
shared block families
virtual recurrent depth
```

Not allowed as the main architecture:

```txt
full unique Wq/Wk/Wv/Wo per layer
full unique MLP matrices per layer
post-training quantisation only
```

## Baselines are mandatory

Every model experiment must include at least one baseline:

- plain decoder-only Transformer
- same tokenizer
- same dataset
- similar hidden size/training budget where possible

## Measure bytes, not vibes

Report:

```txt
stored_unique_parameter_count
stored_unique_weight_bytes_fp32
stored_unique_weight_bytes_bf16
stored_unique_weight_bytes_quantized_estimate
effective_parameter_count_if_materialized
compression_ratio_vs_baseline
```

## Initial model quality expectations

The first model may produce garbage. That is acceptable if:

- loss decreases
- generation works
- metrics are real
- architecture path is preserved

## No premature native runtime

Do not build `tinycore.cpp` before the PyTorch version proves the tensor graph and checkpoint format. Native runtime comes after model stabilisation.
