# ADR-0004: Shared Blocks and Virtual Depth

## Decision

TinyCore should allow more computational depth than unique stored block depth.

## Rationale

The goal is more thinking per weight. Reusing a small number of block families across many virtual steps tests this directly.

## Consequences

- `n_virtual_layers` is separate from `n_families`.
- Routes are indexed by virtual layer.
- Recurrent mixer can help repeated blocks behave differently over depth.
