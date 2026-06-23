# ADR-0002: Weight Composition First

## Decision

Every TinyCore virtual layer obtains major matrices through shared basis composition and optional low-rank deltas.

## Formula

```txt
W_eff = Σ alpha_i * B_i + U @ V
```

## Consequences

- The implementation must distinguish stored unique parameters from effective materialized parameters.
- The runtime may cache composed matrices but should not store them as canonical weights.
