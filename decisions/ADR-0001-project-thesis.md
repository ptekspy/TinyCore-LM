# ADR-0001: Project Thesis

## Decision

TinyCore-LM will pursue architecture-native weight reduction rather than post-training compression.

## Rationale

Post-training quantisation reduces bytes but does not change the fact that the architecture learned huge independent matrices per layer. TinyCore tests whether fewer unique matrices can be learned from the start.

## Consequences

- Baseline comparisons are mandatory.
- Parameter accounting is part of the model API.
- Some quality loss is expected initially.
