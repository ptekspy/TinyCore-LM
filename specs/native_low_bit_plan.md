# Native Low-Bit Plan

## Goal

Make TinyCore compatible with native low-bit weights from the beginning, without making MVP training impossible.

## Phases

### Phase A: BF16/FP32 correctness

Train with ordinary floating-point parameters. Implement parameter accounting that estimates lower-bit storage.

### Phase B: fake quantisation

Use straight-through estimator style fake quantisation on basis matrices:

```txt
forward: quantize weights to ternary/2-bit
backward: pass gradients through real-valued shadow weights
```

### Phase C: basis-only quantisation

Quantise shared basis matrices first. Leave route coefficients, norms, and low-rank deltas in higher precision.

### Phase D: low-rank delta quantisation

Quantise low-rank deltas if quality survives.

### Phase E: native runtime packing

Store packed low-bit basis weights in TCMDL.

## Candidate precisions

```txt
fp32: debugging only
bf16/fp16: baseline training/inference
int8: easy quant target
int4: strong practical target
ternary: aggressive research target (-1,0,+1)
1.58-bit: conceptual target similar to ternary storage
```

## Policy

Do not block MVP on ternary training. Build the architecture so low-bit can be introduced cleanly.

## Storage estimate

A model report must provide:

```json
{
  "storage_estimates": {
    "fp32_bytes": 0,
    "bf16_bytes": 0,
    "int8_bytes": 0,
    "int4_bytes": 0,
    "ternary_1_58bit_bytes": 0
  }
}
```
