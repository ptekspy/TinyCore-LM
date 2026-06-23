# Data Pipeline Spec

## MVP corpus

Use small local/public domain or synthetic corpus for first test. The pack does not include data.

MVP pipeline:

```txt
raw text files
  -> clean empty/control chars
  -> train tokenizer
  -> tokenize
  -> pack into fixed-length binary/sharded dataset
  -> train/val split
```

## Later coding corpus

For coding-agent model:

```txt
permissively licensed code
README/docs
issues/PRs if licence permits
diffs
test failures
terminal logs
instruction/edit traces
```

## Dataset manifest

Every dataset must produce:

```json
{
  "name": "toy_corpus_v0",
  "source": "local_or_public_domain",
  "license_notes": "...",
  "num_documents": 0,
  "num_tokens": 0,
  "tokenizer": "...",
  "train_shards": [],
  "val_shards": [],
  "created_at": "..."
}
```

## Safety / legality

Do not ingest private code, secrets, credentials, or copyrighted datasets without explicit permission. Build filters for:

```txt
api keys
private keys
tokens
emails where not needed
large duplicate files
binary files
minified bundles
```
