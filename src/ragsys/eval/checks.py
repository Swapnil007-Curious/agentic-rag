"""Deterministic checks used by the automatic evaluation-set builder.

* word counts and provision references ("section 35(1)", "rule 4", "clause (b)", "s. 12");
* numbers written as digits or English words ("30", "thirty", "Rs. 50,00,000", "fifty lakh"),
  compared by value, so "30 days" and "thirty days" match;
* capitalised terms (named entities such as "District Commission" or "Central Government") and
  a conservative conflict test between two answers;
* near-duplicate removal with an embedding model (cosine similarity of normalised vectors).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ragsys.retrieval.bm25 import STOPWORDS
from ragsys.retrieval.embed import Encoder


def n_words(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- provision references

_REF_WORD = (
    r"(?:sub-?sections?|sections?|sub-?rules?|rules?|sub-?clauses?|clauses?|regulations?|"
    r"provisos?|articles?)"
)
_YEAR = r"(?:1[89]|20)\d\d\b"
_LABEL = r"\(\s*[0-9a-z]{1,5}\s*\)"
PROVISION_REF_RE = re.compile(
    rf"\b{_REF_WORD}\s*(?:(?!{_YEAR})\d+[a-z]{{0,2}}\b|{_LABEL})(?:\s*{_LABEL})*"
    rf"|\b(?:chapters?|schedules?)\s+(?:[ivxlc]+|\d+)\b"
    rf"|\b(?:s{{1,2}}|r{{1,2}})\.\s?\d+[a-z]?(?:\s*{_LABEL})*"
    rf"|(?<![\w(])\d{{1,3}}[a-z]?(?:\([0-9a-z]{{1,5}}\))+",
    re.IGNORECASE,
)
_REF_NUMBER_RE = re.compile(
    rf"\b(?:sections?|rules?|s{{1,2}}\.|r{{1,2}}\.)\s*(?!{_YEAR})(\d+[a-z]?)\b", re.IGNORECASE
)


def provision_ref_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in PROVISION_REF_RE.finditer(text)]


def mentions_provision_number(text: str) -> bool:
    """True when ``text`` names a section, sub-section, rule, clause, ... number."""
    return PROVISION_REF_RE.search(text) is not None


BOOK_REF_RE = re.compile(
    r"\b(?:sections?|chapters?|figs?\.|figures?|tables?|equations?|eqs?\.)\s*\(?\d+(?:\.\d+)*\)?"
    r"|§\s*\d+(?:\.\d+)*",
    re.IGNORECASE,
)
_BOOK_NUMBER_RE = re.compile(r"(?:\b(?:sections?|chapters?)|§)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)


def mentions_book_reference(text: str) -> bool:
    """True when ``text`` names a book section, chapter, figure, table or equation number."""
    return BOOK_REF_RE.search(text) is not None


def book_reference_numbers(text: str) -> set[str]:
    """Numbers written after "section"/"chapter"/"§", e.g. {"5.1", "3"}."""
    return {m.group(1) for m in _BOOK_NUMBER_RE.finditer(text)}


def section_rule_numbers(text: str) -> set[str]:
    """Numbers written after "section"/"rule" (or "s."/"r."), lower-cased, e.g. {"35", "2a"}."""
    return {m.group(1).lower() for m in _REF_NUMBER_RE.finditer(text)}


# --------------------------------------------------------------------------- numbers

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "lac": 100_000,
    "lacs": 100_000,
    "million": 1_000_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "billion": 1_000_000_000,
}
_NUM_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?|[a-z]+")


@dataclass(frozen=True)
class NumberMention:
    value: float  # e.g. 5000000.0 for "fifty lakh" or "Rs. 50,00,000"
    base: float  # value before a trailing multiplier word: 50.0 for "fifty lakh"
    text: str


def _key(v: float) -> float:
    return round(v, 4)


def extract_numbers(text: str) -> list[NumberMention]:
    """Numbers in ``text``: digits (commas and decimals allowed) and English number words,
    each optionally followed by hundred / thousand / lakh / crore / million."""
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text.lower())  # twenty-one -> twenty one
    toks = [(m.group(), m.start(), m.end()) for m in _NUM_TOKEN_RE.finditer(s)]
    out: list[NumberMention] = []
    i = 0
    while i < len(toks):
        tok, start, _end = toks[i]
        if tok[0].isdigit():
            digits = tok.rstrip(",").replace(",", "")
            try:
                value = float(digits)
            except ValueError:
                i += 1
                continue
            base, j = value, i + 1
            while j < len(toks) and toks[j][0] in _SCALES:
                value *= _SCALES[toks[j][0]]
                j += 1
            out.append(NumberMention(value, base, s[start : toks[j - 1][2]]))
            i = j
            continue
        if tok in _UNITS or tok in _TENS:
            total, current, j = 0.0, 0.0, i
            word_base: float | None = None
            while j < len(toks):
                t = toks[j][0]
                if t in _UNITS:
                    current += _UNITS[t]
                elif t in _TENS:
                    current += _TENS[t]
                elif t == "hundred":
                    current = (current or 1) * 100
                elif t in _SCALES:
                    if word_base is None:
                        word_base = total + (current or 1)
                    total += (current or 1) * _SCALES[t]
                    current = 0
                elif t == "and" and j + 1 < len(toks) and toks[j + 1][0] in (_UNITS.keys() | _TENS):
                    pass
                else:
                    break
                j += 1
            value = total + current
            out.append(
                NumberMention(
                    value, value if word_base is None else word_base, s[start : toks[j - 1][2]]
                )
            )
            i = j
            continue
        i += 1
    return out


def number_forms(texts: Iterable[str]) -> set[float]:
    """Every value (and pre-multiplier base value) of every number in ``texts``."""
    out: set[float] = set()
    for t in texts:
        for m in extract_numbers(t):
            out.add(_key(m.value))
            out.add(_key(m.base))
    return out


def numbers_missing_from_source(
    answer: str, source_texts: Sequence[str], source_ids: Sequence[str] = ()
) -> list[str]:
    """Numbers of ``answer`` that do not occur (in any written form) in the source.

    Numbers inside provision references of the answer ("section 34(1)") may also match the
    source node ids; all other numbers must occur in ``source_texts`` (provision text,
    document titles)."""
    spans = provision_ref_spans(answer)
    masked = list(answer)
    for a, b in spans:
        masked[a:b] = " " * (b - a)
    general = extract_numbers("".join(masked))
    refs = extract_numbers(" ".join(answer[a:b] for a, b in spans))
    src = number_forms(source_texts)
    with_ids = src | number_forms(source_ids)
    missing = [m.text for m in general if _key(m.value) not in src]
    missing += [m.text for m in refs if _key(m.value) not in with_ids]
    return missing


def number_conflict(gold: str, other: str) -> tuple[list[str], list[str]]:
    """(numbers of ``gold`` missing from ``other``, numbers of ``other`` not in ``gold``) when
    both are non-empty, i.e. the two answers state different numbers; else ([], [])."""
    g = {_key(m.value): m.text for m in extract_numbers(gold)}
    o = {_key(m.value): m.text for m in extract_numbers(other)}
    missing = [g[k] for k in g if k not in o]
    extra = [o[k] for k in o if k not in g]
    if missing and extra:
        return missing, extra
    return [], []


# --------------------------------------------------------------------------- capitalised terms

_CONNECTORS = frozenset({"of", "the", "and", "for", "on", "in", "to", "&"})
_NOT_ENTITY = STOPWORDS | frozenset(
    {
        "yes",
        "no",
        "according",
        "per",
        "every",
        "however",
        "also",
        "only",
        "either",
        "neither",
        "thus",
        "therefore",
        "hence",
        "unless",
        "none",
        "nothing",
        "please",
        "note",
    }
)
_CAP_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*|[^\sA-Za-z()\"'“”‘’]+")


def _norm_token(w: str) -> str:
    w = w.lower().strip("'’-")
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def capitalised_terms(text: str) -> list[tuple[str, ...]]:
    """Runs of capitalised words (connectors such as "of the" allowed inside a run), with
    leading function words removed and tokens normalised, e.g. "the State Commission" ->
    ("state", "commission")."""
    terms: list[tuple[str, ...]] = []
    run: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        words = [w.lower().strip("'’-") for w in run]
        toks = [_norm_token(w) for w in words if w not in _CONNECTORS and w not in _NOT_ENTITY]
        toks = [t for t in toks if t]
        if toks:
            term = tuple(toks)
            if term not in terms:
                terms.append(term)
        run.clear()
        pending.clear()

    for m in _CAP_TOKEN_RE.finditer(text):
        w = m.group()
        if not w[0].isalpha():  # punctuation, digits: a run ends here
            flush()
            continue
        if w[0].isupper():
            run.extend(pending)
            pending.clear()
            run.append(w)
        elif run and w.lower() in _CONNECTORS:
            pending.append(w)
        else:
            flush()
    flush()
    return terms


def entity_conflicts(gold: str, other: str) -> list[tuple[str, str]]:
    """Pairs (gold term, other term) that name different entities of the same kind.

    Terms match when one's tokens are a subset of the other's ("National Commission" ~
    "National Consumer Disputes Redressal Commission"). A conflict is a gold term without a
    match in ``other`` and an ``other`` term without a match in ``gold`` that share the head
    (last) word, e.g. "State Commission" vs "District Commission"."""
    g_terms = capitalised_terms(gold)
    o_terms = capitalised_terms(other)

    def matched(x: tuple[str, ...], ys: list[tuple[str, ...]]) -> bool:
        sx = set(x)
        return any(sx <= set(y) or set(y) <= sx for y in ys)

    missing = [g for g in g_terms if not matched(g, o_terms)]
    extra = [o for o in o_terms if not matched(o, g_terms)]
    return [(" ".join(g), " ".join(o)) for g in missing for o in extra if g[-1] == o[-1]]


# --------------------------------------------------------------------------- near duplicates


@dataclass
class DuplicateResult:
    keep: list[bool]
    max_similarity: list[float]
    nearest: list[str | None]  # text of the most similar earlier/reference question


def near_duplicates(
    texts: Sequence[str],
    encoder: Encoder,
    threshold: float = 0.90,
    reference: Sequence[str] = (),
) -> DuplicateResult:
    """Keep-first de-duplication: ``texts[i]`` is dropped when its cosine similarity to a
    reference text or to an earlier kept text is above ``threshold``."""
    if not texts:
        return DuplicateResult([], [], [])
    all_texts = [*reference, *texts]
    vecs = np.asarray(encoder.encode(all_texts, is_query=False), dtype=np.float32)
    kept = list(range(len(reference)))
    res = DuplicateResult([], [], [])
    for i in range(len(reference), len(all_texts)):
        if kept:
            sims = vecs[kept] @ vecs[i]
            j = int(np.argmax(sims))
            best, nearest = float(sims[j]), all_texts[kept[j]]
        else:
            best, nearest = 0.0, None
        ok = best <= threshold
        res.keep.append(ok)
        res.max_similarity.append(round(best, 4))
        res.nearest.append(nearest)
        if ok:
            kept.append(i)
    return res
