# ADR-0003: Native Low-Bit Readiness

## Decision

TinyCore will be designed for low-bit basis weights but will not block MVP on low-bit training.

## Rationale

BF16 correctness is easier. Once the architecture trains, fake quantisation and packed basis storage can be introduced.

## Consequences

- Storage estimates must include low-bit projections.
- Basis matrices are the first quantisation target.
