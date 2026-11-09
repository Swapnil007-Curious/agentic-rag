"""Fixed-size sliding windows over a document's cleaned text, measured in tokens.

Window edges are snapped to word boundaries (a token that starts a whitespace-delimited
word), so no word is cut in half; a window never exceeds ``size`` tokens. Consecutive
windows overlap by roughly ``overlap`` tokens and together cover every token.
"""

from __future__ import annotations

import bisect

from ragsys.chunking.base import Chunk, DocInput, make_chunk
from ragsys.chunking.tokenizer import TokenCounter


def word_start_tokens(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    """Indices of tokens that begin a new whitespace-delimited word."""
    starts = []
    for i, (a, _b) in enumerate(offsets):
        if i == 0:
            starts.append(0)
            continue
        prev_end = offsets[i - 1][1]
        if a > prev_end and text[prev_end:a].strip() == "":
            starts.append(i)
    return starts


def token_windows(
    text: str, offsets: list[tuple[int, int]], size: int, overlap: int
) -> list[tuple[int, int]]:
    """Return [token_start, token_end) windows snapped to word starts."""
    n = len(offsets)
    if n == 0:
        return []
    bounds = [*word_start_tokens(text, offsets), n]
    windows: list[tuple[int, int]] = []
    s = 0
    while True:
        limit = min(s + size, n)
        j = bisect.bisect_right(bounds, limit) - 1
        e = bounds[j] if j >= 0 and bounds[j] > s else limit  # hard split for a giant "word"
        windows.append((s, e))
        if e >= n:
            break
        target = max(e - overlap, s + 1)
        k = bisect.bisect_left(bounds, target)
        s_next = bounds[k] if k < len(bounds) else e
        s = min(s_next, e)
    return windows


class FixedChunker:
    def __init__(self, name: str, size: int, overlap: int, tokenizer: TokenCounter) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.name = name
        self.size = size
        self.overlap = overlap
        self.tok = tokenizer

    def chunk_document(self, doc: DocInput) -> list[Chunk]:
        offsets = self.tok.offsets(doc.text)
        chunks: list[Chunk] = []
        for i, (s, e) in enumerate(token_windows(doc.text, offsets, self.size, self.overlap)):
            start, end = offsets[s][0], offsets[e - 1][1]
            n_tokens = self.tok.count(doc.text[start:end])
            chunks.append(make_chunk(doc, self.name, i, start, end, n_tokens))
        return chunks
