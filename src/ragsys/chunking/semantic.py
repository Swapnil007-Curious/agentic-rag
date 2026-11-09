"""Semantic chunking: embedding-distance breakpoints between adjacent sentence/clause units.

1. split cleaned text into units with a legal-aware splitter (paragraphs, then sentences;
   never breaks on "s.", "No.", "viz.", "i.e.", initials or leading numbering like "2.")
2. embed each unit with BAAI/bge-small-en-v1.5 on CPU (batched; cached on disk by text hash)
3. break where cosine distance between adjacent units exceeds the document's
   ``breakpoint_percentile`` of distances
4. enforce ``max_tokens`` (split) and ``min_tokens`` (merge into the more similar neighbour)
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ragsys.chunking.base import Chunk, DocInput, make_chunk
from ragsys.chunking.fixed import token_windows
from ragsys.chunking.tokenizer import TokenCounter

EmbedFn = Callable[[list[str]], np.ndarray]

# fmt: off
_ABBREVIATIONS = frozenset({
    "s", "ss", "no", "nos", "viz", "i.e", "e.g", "cl", "cls", "sch", "ins", "subs", "notifn",
    "w.e.f", "dr", "mr", "mrs", "ms", "jt", "secy", "govt", "co", "ltd", "pvt", "art", "arts",
    "rs", "vol", "p", "pp", "para", "paras", "sec", "secs", "r", "rr", "sl", "st", "vs", "v",
    "fig", "ch", "ord", "s.o", "g.s.r", "u.s", "u.k", "a.d", "b.c", "dept", "ref", "regd", "cf",
})
# fmt: on
_SENT_END_RE = re.compile(r"[.?!](?=[\"')\]]*\s+[\"'(\[]?[A-Z0-9])")
_WORD_BEFORE_RE = re.compile(r"([A-Za-z0-9./]+)$")


def _is_abbreviation(para: str, dot_pos: int) -> bool:
    m = _WORD_BEFORE_RE.search(para[:dot_pos])
    if not m:
        return False
    word = m.group(1).lower().strip(".")
    if word in _ABBREVIATIONS:
        return True
    if len(word) == 1 and word.isalpha():
        return True  # initials: "D. L.", "A. K."
    # leading numbering such as "2. Definitions" or "12A. ..."
    return bool(re.fullmatch(r"\d{1,3}[a-z]{0,2}", word)) and para[: m.start()].strip(" [") == ""


def split_units(text: str) -> list[tuple[int, int]]:
    """Split into sentence/clause units; returns trimmed (start, end) spans covering all
    non-whitespace characters of ``text``."""
    spans: list[tuple[int, int]] = []

    def add(a: int, b: int) -> None:
        seg = text[a:b]
        lstrip = len(seg) - len(seg.lstrip())
        rstrip = len(seg) - len(seg.rstrip())
        if b - rstrip > a + lstrip:
            spans.append((a + lstrip, b - rstrip))

    for pm in re.finditer(r"[^\n]+", text):
        base, para = pm.start(), pm.group(0)
        last = 0
        for em in _SENT_END_RE.finditer(para):
            if _is_abbreviation(para, em.start()):
                continue
            idx = em.end()
            idx += len(re.match(r"[\"')\]]*", para[idx:]).group(0))  # type: ignore[union-attr]
            add(base + last, base + idx)
            last = idx
        add(base + last, base + len(para))
    return spans


class EmbeddingCache:
    """On-disk cache {sha256(model + text) -> float32 vector} stored as one .npz per model."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._data: dict[str, np.ndarray] = {}
        self._dirty = False
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as z:
                for k, v in zip(z["keys"].tolist(), z["vecs"], strict=True):
                    self._data[str(k)] = np.asarray(v, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> np.ndarray | None:
        return self._data.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        self._data[key] = np.asarray(vec, dtype=np.float32)
        self._dirty = True

    def save(self) -> None:
        if self.path is None or not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._data)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(keys), vecs=np.stack([self._data[k] for k in keys]))
        tmp.replace(self.path)
        self._dirty = False


class Embedder:
    """Lazy SentenceTransformer wrapper with a persistent embedding cache."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name)
        self.cache = EmbeddingCache(cache_dir / f"embeddings_{slug}.npz" if cache_dir else None)
        self._model: Any = None
        self.stats = {"cached": 0, "computed": 0}

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\n{text}".encode()).hexdigest()

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def __call__(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        out: list[np.ndarray | None] = [self.cache.get(k) for k in keys]
        missing = sorted({t for t, v in zip(texts, out, strict=True) if v is None})
        self.stats["cached"] += sum(v is not None for v in out)
        if missing:
            vecs = self._load().encode(
                missing,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for t, v in zip(missing, vecs, strict=True):
                self.cache.put(self._key(t), v)
            self.stats["computed"] += len(missing)
            self.cache.save()
            out = [self.cache.get(k) for k in keys]
        return np.stack([v for v in out if v is not None]).astype(np.float32)


class SemanticChunker:
    name = "semantic"

    def __init__(
        self,
        tokenizer: TokenCounter,
        embed_fn: EmbedFn,
        breakpoint_percentile: float = 90.0,
        min_tokens: int = 64,
        max_tokens: int = 512,
    ) -> None:
        self.tok = tokenizer
        self.embed = embed_fn
        self.percentile = breakpoint_percentile
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def _presplit(self, text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        counts = self.tok.count_many([text[a:b] for a, b in spans])
        for (a, b), n in zip(spans, counts, strict=True):
            if n <= self.max_tokens:
                out.append((a, b))
                continue
            seg = text[a:b]
            offs = self.tok.offsets(seg)
            for s, e in token_windows(seg, offs, self.max_tokens, 0):
                out.append((a + offs[s][0], a + offs[e - 1][1]))
        return out

    def units(self, text: str) -> list[tuple[int, int]]:
        return self._presplit(text, split_units(text))

    def breakpoints(self, distances: np.ndarray) -> set[int]:
        if distances.size == 0:
            return set()
        thr = float(np.percentile(distances, self.percentile))
        return {i for i, d in enumerate(distances.tolist()) if d > thr}

    def _group(self, ntok: list[int], distances: np.ndarray) -> list[list[int]]:
        n = len(ntok)
        bps = self.breakpoints(distances)
        groups: list[list[int]] = [[0]]
        for i in range(1, n):
            if (i - 1) in bps:
                groups.append([i])
            else:
                groups[-1].append(i)

        # enforce max: greedy split of oversized groups
        sized: list[list[int]] = []
        for g in groups:
            cur: list[int] = []
            total = 0
            for u in g:
                if cur and total + ntok[u] > self.max_tokens:
                    sized.append(cur)
                    cur, total = [], 0
                cur.append(u)
                total += ntok[u]
            if cur:
                sized.append(cur)
        groups = sized

        # enforce min: merge undersized groups into the more similar neighbour
        def tokens(g: list[int]) -> int:
            return sum(ntok[u] for u in g)

        changed = True
        while changed and len(groups) > 1:
            changed = False
            for gi, g in enumerate(groups):
                if tokens(g) >= self.min_tokens:
                    continue
                options: list[tuple[float, int]] = []
                if gi > 0 and tokens(groups[gi - 1]) + tokens(g) <= self.max_tokens:
                    options.append((float(distances[g[0] - 1]), gi - 1))
                if gi + 1 < len(groups) and tokens(groups[gi + 1]) + tokens(g) <= self.max_tokens:
                    options.append((float(distances[g[-1]]), gi + 1))
                if not options:
                    continue
                _, other = min(options)
                lo, hi = sorted((gi, other))
                groups[lo] = groups[lo] + groups[hi]
                del groups[hi]
                changed = True
                break
        return groups

    def chunk_document(self, doc: DocInput) -> list[Chunk]:
        spans = self.units(doc.text)
        if not spans:
            return []
        texts = [doc.text[a:b] for a, b in spans]
        ntok = self.tok.count_many(texts)
        if len(spans) > 1:
            emb = self.embed(texts)
            distances = 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)
        else:
            distances = np.zeros(0, dtype=np.float32)
        chunks: list[Chunk] = []
        for g in self._group(ntok, distances):
            start, end = spans[g[0]][0], spans[g[-1]][1]
            n_tokens = self.tok.count(doc.text[start:end])
            if n_tokens <= self.max_tokens:
                chunks.append(make_chunk(doc, self.name, len(chunks), start, end, n_tokens))
                continue
            seg = doc.text[start:end]  # defensive: token sums can differ from a joint count
            offs = self.tok.offsets(seg)
            for s, e in token_windows(seg, offs, self.max_tokens, 0):
                a, b = start + offs[s][0], start + offs[e - 1][1]
                chunks.append(
                    make_chunk(doc, self.name, len(chunks), a, b, self.tok.count(doc.text[a:b]))
                )
        return chunks
