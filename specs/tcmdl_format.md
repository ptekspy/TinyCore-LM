# TCMDL Model Format Spec

## Purpose

TCMDL is a future file format for TinyCore models. It must store compositional model metadata, tokenizer data, basis weights, route coefficients, low-rank deltas, quantisation metadata, and runtime compatibility info.

## MVP format

For Python phase, use:

```txt
checkpoint.pt
manifest.json
config.yaml
tokenizer.json
metrics.json
tensor_index.json  # optional native-friendly sidecar
tensors.bin        # optional native-friendly raw tensor bytes
```

Native TCMDL comes later.

The first bridge to native runtime is a deterministic `.tcmdl` bundle:

```txt
magic: TCMDL\0
header_length: uint64 little-endian
header_json: canonical UTF-8 JSON
payload: manifest.json + files from manifest.files
```

The bundle is still a packaging format, not a native inference format. It keeps
the Python checkpoint payload intact and records file offsets, lengths, and
SHA-256 hashes so format tools can verify and extract the artifact, and native
inspectors can read the header and embedded manifest metadata before tensor
loading or kernels exist.

The tensor sidecar bridge is:

```txt
tensor_index.json: tensor names, dtype, shape, offset, length, sha256
tensors.bin: contiguous raw tensor bytes in tensor_index order
```

This sidecar is generated from `checkpoint.pt`; it does not replace the training
checkpoint yet. Native code may inspect the sidecar table, check byte totals,
probe selected float32 tensor bytes, and run small reference math probes before
implementing actual inference. Current probes include embedding/head dot
products, one TinyCore composed-linear projection, and a single-token RMSNorm
plus Q-projection block probe. The native spike also probes the one-token
attention sublayer where causal attention reduces to the V projection, then the
recurrent-state injection and MLP sublayer for that token. A final norm plus
lm_head probe produces one-layer logits, and the reference path can now run
across all virtual layers for one token. A two-token sequence probe exercises
causal attention with a non-trivial attention softmax over previous positions.
A prompt-level greedy generation probe for `TinyCore` now emits
`TinyCoreeeerree ` from native reference logits, matching the Python runtime
for the same artifact. The same reference path is exposed as the
`tinycore-generate` native CLI for both packed `.tcmdl` bundles and extracted
artifact directories, with a regression test that checks the generated text and
token IDs when the native binary is built. The low-level native parsing and
reference math helpers live in `native_runtime.hpp` and `native_runtime.cpp`,
along with the shared artifact loading, prompt logits, and greedy generation
path used by both native executables. Native generation also supports seeded
temperature/top-k sampling while preserving greedy generation as the pinned
regression path. CTest covers native tensor parsing, composed Q projection math,
greedy generation, and sampled top-k invariants against the reference artifact.
`tinycored` can route `/generate` and message `/chat` through this native
backend for packed `.tcmdl` artifacts.

## TCMDL conceptual layout

```txt
magic: TCMDL\0
version
header_length
header_json
section_table
binary_sections
checksum
```

## Header JSON

```json
{
  "format": "TCMDL",
  "format_version": "0.1.0",
  "architecture": "tinycore_recurrent_v0",
  "tokenizer": {
    "type": "bpe",
    "vocab_size": 32000,
    "section": "tokenizer_json"
  },
  "model": {
    "d_model": 512,
    "n_heads": 8,
    "n_virtual_layers": 32,
    "n_families": 4,
    "basis_rank": 8,
    "low_rank": 16
  },
  "quantization": {
    "basis": "bf16",
    "routes": "fp16",
    "low_rank": "fp16"
  },
  "sections": [
    {"name":"basis.attn_q", "dtype":"bf16", "shape":[4,8,512,512], "offset":0, "length":0}
  ]
}
```

## Required inspection commands

```bash
tinycore-format inspect model.tcmdl
tinycore-format verify model.tcmdl
tinycore-format estimate-size config.yaml
tinycore-format export-tensors artifact_dir
tinycore-format convert artifact_dir model.tcmdl
tinycore-format extract model.tcmdl artifact_dir
```

## Do not implement binary TCMDL too early

Implement manifest/checkpoint discipline first. Binary format after architecture stabilises.
