# Memory Benchmark Plan

## Compare

- baseline Transformer
- shared-layer Transformer
- TinyCore basis-only
- TinyCore basis+low-rank
- TinyCore recurrent

## Report

```txt
stored unique params
stored unique bytes by precision
effective params if materialized
activation peak memory
KV cache bytes/token if generation implemented
tokens/sec train
tokens/sec generate
validation loss
```

## Critical graph

X axis:

```txt
stored unique bytes
```

Y axis:

```txt
validation loss
```

TinyCore is interesting only if it moves the Pareto frontier.
