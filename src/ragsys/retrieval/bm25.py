"""Lexical retrieval: BM25Okapi (rank-bm25) over chunk ``text`` with a legal-aware tokenizer.

Tokenizer:
* Unicode NFKC, lower case, hyphenated words joined (``sub-section`` -> ``subsection``,
  ``e-commerce`` -> ``ecommerce``), alphanumeric tokens, built-in English stopword list;
* statute references also emit compact tokens, so a query and the text match however the
  reference is written: ``section 2(7)`` -> ``s2(7)``, ``2(7)``, ``s2``; ``rule 4(3)`` ->
  ``r4(3)``, ``4(3)``, ``r4``; ``clause (zg) of sub-section (2) of section 101`` ->
  ``s101(2)(zg)``, ``101(2)(zg)``, ``s101(2)``, ``s101``; typed forms such as ``s35(1)`` too;
* optional light suffix stripping (plural / -ing / -ed), off by default.
"""

from __future__ import annotations

import itertools
import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

# fmt: off
STOPWORDS: frozenset[str] = frozenset(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves", "also", "upon", "via", "thereof", "therein", "thereto", "whereas", "within", "without", "shall", "may"])
# fmt: on
# "shall"/"may" occur in nearly every provision and carry no topical signal.

_HYPHENS = dict.fromkeys(map(ord, "‐‑‒−­"), "-")
_LABEL = r"\(\s*[a-z0-9]{1,5}\s*\)"
_LABELS = rf"{_LABEL}(?:\s*,?\s*(?:and|or|to)?\s*{_LABEL})*"
_NUM = r"\d{1,3}[a-z]{0,2}"
_UNIT = r"(?P<unit>sections?|secs?\.?|ss?\.|rules?|rr?\.)"
DIRECT_RE = re.compile(rf"\b{_UNIT}\s*(?P<num>{_NUM})\b(?P<labels>(?:\s*{_LABEL})*)")
TYPED_RE = re.compile(
    rf"(?<![a-z0-9])(?P<unit>[sr])(?P<num>\d{{1,3}}[a-z]?)(?P<labels>(?:{_LABEL})*)(?![a-z0-9])"
)
CHAIN_RE = re.compile(
    rf"(?P<chain>(?:(?:sub-?sections?|sub-?rules?|sub-?clauses?|clauses?|items?)\s*{_LABELS}"
    rf"\s+(?:of|in|under)\s+(?:the\s+)?)+)(?P<unit>sections?|rules?)\s+(?P<num>{_NUM})\b"
)
BARE_RE = re.compile(rf"(?<![a-z0-9(])(?P<num>\d{{1,3}}[a-z]?)(?P<labels>(?:{_LABEL})+)")
_COMPONENT_RE = re.compile(
    rf"(?:sub-?sections?|sub-?rules?|sub-?clauses?|clauses?|items?)\s*(?P<labels>{_LABELS})"
)
_LABEL_VALUE_RE = re.compile(r"\(\s*([a-z0-9]{1,5})\s*\)")
_UNIT_BEFORE_RE = re.compile(r"\b(?:sections?|secs?\.?|ss?\.|rules?|rr?\.)\s*$")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _prefix(unit: str) -> str:
    return "r" if unit.startswith("r") else "s"


def _ref_tokens(prefix: str, num: str, labels: list[str]) -> list[str]:
    path = "".join(f"({lab})" for lab in labels)
    out = [f"{prefix}{num}{path}"]
    if labels:
        out.append(f"{num}{path}")
        out.extend(
            f"{prefix}{num}" + "".join(f"({x})" for x in labels[:i]) for i in range(len(labels))
        )
    return out


def reference_tokens(text: str) -> list[str]:
    """Compact statute-reference tokens found in normalised lower-case ``text``."""
    out: list[str] = []
    for m in DIRECT_RE.finditer(text):
        labels = _LABEL_VALUE_RE.findall(m.group("labels"))
        out.extend(_ref_tokens(_prefix(m.group("unit")), m.group("num"), labels))
    for m in TYPED_RE.finditer(text):
        labels = _LABEL_VALUE_RE.findall(m.group("labels"))
        out.extend(_ref_tokens(m.group("unit"), m.group("num"), labels))
    for m in CHAIN_RE.finditer(text):
        comps = [
            _LABEL_VALUE_RE.findall(c.group("labels"))
            for c in _COMPONENT_RE.finditer(m.group("chain"))
        ]
        comps = [c for c in comps if c]
        # innermost component is written first: "clause (a) of sub-section (1) of section 3"
        for combo in itertools.islice(itertools.product(*reversed(comps)), 16):
            out.extend(_ref_tokens(_prefix(m.group("unit")), m.group("num"), list(combo)))
    for m in BARE_RE.finditer(text):
        if _UNIT_BEFORE_RE.search(text[max(0, m.start() - 12) : m.start()]):
            continue  # already emitted by DIRECT_RE
        labels = _LABEL_VALUE_RE.findall(m.group("labels"))
        out.append(m.group("num") + "".join(f"({x})" for x in labels))
    return out


def light_stem(word: str) -> str:
    """Very light English suffix stripping (plurals, -ing, -ed)."""
    if len(word) <= 4 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    if word.endswith("ing") and len(word) > 6:
        return word[:-3]
    if word.endswith("ed") and len(word) > 5:
        return word[:-2]
    return word


def normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_HYPHENS).lower()


def tokenize(text: str, stem: bool = False, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    norm = normalise(text)
    refs = reference_tokens(norm)
    joined = re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", "", norm)
    words = [w for w in _WORD_RE.findall(joined) if w not in stopwords]
    if stem:
        words = [light_stem(w) for w in words]
    return words + refs


class BM25Index:
    """BM25Okapi over a fixed list of documents; ``search`` returns positive scores only."""

    def __init__(
        self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75, stem: bool = False
    ) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        if not ids:
            raise ValueError("BM25 needs at least one document")
        self.ids = list(ids)
        self.stem = stem
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(t, stem) for t in texts]
        self._bm25 = BM25Okapi(self.doc_tokens, k1=k1, b=b)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def vocab_size(self) -> int:
        return len({t for doc in self.doc_tokens for t in doc})

    @property
    def avg_doc_len(self) -> float:
        return float(np.mean([len(d) for d in self.doc_tokens]))

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query, self.stem)
        if not q:
            return np.zeros(len(self.ids))
        return np.asarray(self._bm25.get_scores(q), dtype=np.float64)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        s = self.scores(query)
        order = np.lexsort((np.arange(len(s)), -s))[:k]
        return [(self.ids[i], float(s[i])) for i in order if s[i] > 0]
