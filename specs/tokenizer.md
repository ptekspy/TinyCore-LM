# Tokenizer Spec

## MVP tokenizer

Use a proven tokenizer library rather than inventing tokenization first.

Recommended:

- Hugging Face `tokenizers`
- BPE or Unigram
- small vocab for toy experiments: 4k..16k
- larger later: 32k..100k

## Code-aware target

For a coding agent model, tokenizer should preserve useful code structure:

```txt
indentation
common operators
braces/parens
camelCase/snake_case fragments
file paths
package names
JSON/YAML punctuation
Markdown fences
```

## Special tokens

```txt
<bos>
<eos>
<pad>
<unk>
<user>
<assistant>
<system>
<tool_call>
<tool_result>
<file>
<diff>
```

## Tokenizer manifest

```json
{
  "tokenizer_type": "bpe",
  "vocab_size": 8192,
  "special_tokens": ["<bos>", "<eos>", "<pad>", "<unk>"],
  "training_corpus_manifest": "...",
  "normalization": "byte_level_or_none",
  "created_at": "iso8601"
}
```

## Do not overfit tokenization initially

First prove the architecture with a simple tokenizer. Improve tokenization after baseline comparisons exist.
