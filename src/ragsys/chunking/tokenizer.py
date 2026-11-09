"""Token counting with the BAAI/bge-small-en-v1.5 tokenizer (cached per process).

Counts exclude special tokens ([CLS]/[SEP]); offsets are character offsets into the input.
"""

from __future__ import annotations

import functools
import os
from typing import Any

DEFAULT_TOKENIZER = "BAAI/bge-small-en-v1.5"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class TokenCounter:
    def __init__(self, model_name: str = DEFAULT_TOKENIZER) -> None:
        from transformers import AutoTokenizer

        self.model_name = model_name
        hf_tok = AutoTokenizer.from_pretrained(model_name)
        backend: Any = hf_tok.backend_tokenizer  # tokenizers.Tokenizer (Rust, fast)
        backend.no_truncation()
        backend.no_padding()
        self._backend = backend

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._backend.encode(text, add_special_tokens=False).ids)

    def count_many(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        encs = self._backend.encode_batch(texts, add_special_tokens=False)
        return [len(e.ids) for e in encs]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token of ``text`` (no special tokens)."""
        if not text:
            return []
        enc = self._backend.encode(text, add_special_tokens=False)
        return [(int(a), int(b)) for a, b in enc.offsets]


@functools.lru_cache(maxsize=4)
def get_tokenizer(model_name: str = DEFAULT_TOKENIZER) -> TokenCounter:
    return TokenCounter(model_name)
