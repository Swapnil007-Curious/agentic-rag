"""Drafting candidate evaluation questions with an LLM, for human review.

Target nodes are sampled per question type (definitions, factual provisions, procedural
provisions, provisions with numbers / time limits / amounts, resolved cross-references, and
Act + Rule pairs), stratified by document (quota per document proportional to the square root of
its candidate count, so the short Rules are represented) and by node level (round-robin), with
a seeded RNG. Each draft records the source nodes as ``gold_node_ids``, the source text, the
model name and a lexical-overlap score that flags copy-like questions. Drafts are written to
data/eval/drafts.jsonl with status ``pending``; only scripts/review_drafts.py moves accepted
drafts into questions.jsonl. Unanswerable questions are never drafted.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragsys.llm.client import ChatModel, LLMJSONError, Message
from ragsys.retrieval.bm25 import STOPWORDS, BM25Index, normalise
from ragsys.retrieval.corpus import NodeGraph
from ragsys.structure.base import Node
from ragsys.structure.xrefs import XRef

PROMPT_VERSION = "draft-v1"
TYPE_MIX: dict[str, int] = {
    "definition": 15,
    "factual": 20,
    "procedural": 15,
    "numeric": 15,
    "cross_reference": 15,
    "multi_section": 10,
}
# scarce pools are sampled first so the general "factual" pool does not use their nodes
SAMPLING_ORDER = (
    "multi_section",
    "cross_reference",
    "numeric",
    "procedural",
    "definition",
    "factual",
)
OVERLAP_FLAG_THRESHOLD = 0.6
MAX_ANSWER_WORDS = 40
SOURCE_CHARS = 2500

_SKIP_LEVELS = {"chapter", "heading", "schedule"}
PROCEDURAL_RE = re.compile(
    r"\b(manner|procedure|shall be (?:filed|made|submitted|presented|preferred|sent|served|"
    r"issued|published|displayed|maintained)|application|apply|appeal|file|filing|notice|"
    r"hearing|submit|register|form|steps?|forward|refer)\b",
    re.IGNORECASE,
)
NUMBER_WORDS = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|"
    r"twenty-one|thirty|forty|forty-five|fifty|sixty|ninety|hundred"
)
NUMERIC_RE = re.compile(
    rf"\b(?:\d+|{NUMBER_WORDS})\s+(?:working\s+)?(?:days?|months?|years?|weeks?|hours?|"
    rf"members?|times|per\s+cent|lakh|crore|rupees)\b|\brupees\s+[\w,]+|\b(?:lakh|crore)\b|"
    rf"\bper\s+cent\b",
    re.IGNORECASE,
)
SECTION_NUMBER_RE = re.compile(
    r"\b(?:section|sub-section|rule|sub-rule|clause|regulation)s?\s*\(?\d|\b[sr]\.?\s?\d+\s*\(",
    re.IGNORECASE,
)

TYPE_GUIDANCE: dict[str, str] = {
    "definition": "Ask what a term defined in the text means or what it includes.",
    "factual": "Ask about a specific fact stated in the text (who, what, which body, what is required).",
    "procedural": "Ask how something is done: the steps, the manner, or who must do what and how.",
    "numeric": "Ask for a number, time limit, amount or count that is stated in the text.",
    "cross_reference": (
        "The first provision refers to the second one. Ask a question whose full answer needs "
        "both provisions, i.e. it requires following the reference."
    ),
    "multi_section": (
        "One provision is from the Act and one from Rules made under it. Ask a question whose "
        "full answer needs information from both provisions."
    ),
}


@dataclass(frozen=True)
class Candidate:
    qtype: str
    nodes: tuple[Node, ...]

    @property
    def doc_id(self) -> str:
        return self.nodes[0].doc_id

    @property
    def kind(self) -> str:
        return self.nodes[0].level

    @property
    def node_ids(self) -> list[str]:
        return [n.node_id for n in self.nodes]


def _n_words(n: Node) -> int:
    return len(n.text.split())


def _eligible(n: Node, lo: int = 12, hi: int = 350) -> bool:
    return n.level not in _SKIP_LEVELS and lo <= _n_words(n) <= hi


def allocate(n: int, mix: dict[str, int]) -> dict[str, int]:
    """Scale the type mix to ``n`` questions (largest remainder, deterministic)."""
    total = sum(mix.values())
    if n <= 0 or total <= 0:
        return dict.fromkeys(mix, 0)
    raw = {t: n * w / total for t, w in mix.items()}
    out = {t: math.floor(v) for t, v in raw.items()}
    order = sorted(mix, key=lambda t: (-(raw[t] - out[t]), list(mix).index(t)))
    for t in order[: n - sum(out.values())]:
        out[t] += 1
    return out


def build_pools(
    nodes: Sequence[Node],
    xrefs: Iterable[XRef],
    graph: NodeGraph,
    exclude: Iterable[str] = (),
) -> dict[str, list[Candidate]]:
    excluded = set(exclude)
    usable = [n for n in nodes if n.node_id not in excluded]
    pools: dict[str, list[Candidate]] = {t: [] for t in TYPE_MIX}
    for n in usable:
        if n.defined_term and _eligible(n, lo=6):
            pools["definition"].append(Candidate("definition", (n,)))
            continue
        if not _eligible(n) or n.defined_term:
            continue
        pools["factual"].append(Candidate("factual", (n,)))
        if PROCEDURAL_RE.search(n.text) or (n.heading and PROCEDURAL_RE.search(n.heading)):
            pools["procedural"].append(Candidate("procedural", (n,)))
        if NUMERIC_RE.search(n.text):
            pools["numeric"].append(Candidate("numeric", (n,)))

    seen_pairs: set[tuple[str, str]] = set()
    for x in xrefs:
        if x.resolution != "exact" or not x.target_node:
            continue
        s, t = x.source_node, x.target_node
        if s in excluded or s not in graph or t not in graph or s == t:
            continue
        if t in graph.ancestors(s) or t in graph.descendants(s):
            continue
        src, tgt = graph.nodes[s], graph.nodes[t]
        if not (_eligible(src, lo=8) and _eligible(tgt, lo=8, hi=400)) or (s, t) in seen_pairs:
            continue
        seen_pairs.add((s, t))
        pools["cross_reference"].append(Candidate("cross_reference", (src, tgt)))
        if src.doc_type == "rules" and tgt.doc_type == "act":
            pools["multi_section"].append(Candidate("multi_section", (src, tgt)))

    # Act + Rule pairs without an explicit reference: best BM25 match of each rule-level node
    act_nodes = [
        n for n in usable if n.doc_type == "act" and _eligible(n, lo=15) and not n.defined_term
    ]
    rule_nodes = [
        n
        for n in usable
        if n.doc_type == "rules" and n.level in ("rule", "subsection") and _eligible(n, lo=15)
    ]
    if act_nodes and rule_nodes:
        index = BM25Index([n.node_id for n in act_nodes], [n.text for n in act_nodes])
        by_id = {n.node_id: n for n in act_nodes}
        for rn in rule_nodes:
            hits = index.search(rn.text, 1)
            if hits and hits[0][1] >= 15.0 and (rn.node_id, hits[0][0]) not in seen_pairs:
                seen_pairs.add((rn.node_id, hits[0][0]))
                pools["multi_section"].append(Candidate("multi_section", (rn, by_id[hits[0][0]])))
    return pools


def _doc_quotas(sizes: dict[str, int], quota: int) -> dict[str, int]:
    """Split ``quota`` across documents proportionally to sqrt(pool size), capped by size."""
    out = dict.fromkeys(sizes, 0)
    remaining = quota
    active = {d for d, s in sizes.items() if s > 0}
    while remaining > 0 and active:
        weights = {d: math.sqrt(sizes[d] - out[d]) for d in active}
        total = sum(weights.values())
        raw = {d: remaining * w / total for d, w in weights.items()}
        give = {d: min(math.floor(v), sizes[d] - out[d]) for d, v in raw.items()}
        if sum(give.values()) == 0:
            # hand out the remainder one by one, largest fractional share first
            for d in sorted(active, key=lambda d: (-(raw[d] - math.floor(raw[d])), d)):
                if remaining == 0:
                    break
                if out[d] < sizes[d]:
                    out[d] += 1
                    remaining -= 1
        else:
            for d, g in give.items():
                out[d] += g
                remaining -= g
        active = {d for d in active if out[d] < sizes[d]}
    return out


def stratified_sample(
    pools: dict[str, list[Candidate]], alloc: dict[str, int], seed: int = 0
) -> tuple[list[Candidate], dict[str, int]]:
    """Sample candidates per type (scarce types first); nodes are never reused.
    Returns the sample and the shortfall per type."""
    used: set[str] = set()
    sample: list[Candidate] = []
    shortfall: dict[str, int] = {}
    for qtype in SAMPLING_ORDER:
        quota = alloc.get(qtype, 0)
        if quota <= 0:
            continue
        rng = random.Random(f"{seed}-{qtype}")
        by_doc: dict[str, list[Candidate]] = defaultdict(list)
        for c in pools.get(qtype, []):
            if not used.intersection(c.node_ids):
                by_doc[c.doc_id].append(c)
        quotas = _doc_quotas({d: len(v) for d, v in sorted(by_doc.items())}, quota)
        taken = 0
        for doc in sorted(by_doc):
            by_kind: dict[str, list[Candidate]] = defaultdict(list)
            for c in by_doc[doc]:
                by_kind[c.kind].append(c)
            kinds = sorted(by_kind)
            for k in kinds:
                rng.shuffle(by_kind[k])
            got = 0
            while got < quotas[doc] and any(by_kind.values()):
                for k in kinds:
                    if got >= quotas[doc]:
                        break
                    while by_kind[k]:
                        c = by_kind[k].pop()
                        if not used.intersection(c.node_ids):
                            used.update(c.node_ids)
                            sample.append(c)
                            got += 1
                            break
            taken += got
        if taken < quota:
            shortfall[qtype] = quota - taken
    return sample, shortfall


def build_messages(c: Candidate, doc_titles: dict[str, str]) -> list[Message]:
    section_rule = (
        "You may mention the section or rule number of the first provision."
        if c.qtype == "cross_reference"
        else "Do not mention section, rule, sub-section or clause numbers in the question."
    )
    blocks = []
    labels = ["PROVISION", "SECOND PROVISION"]
    for label, n in zip(labels, c.nodes, strict=False):
        title = doc_titles.get(n.doc_id, n.doc_title)
        blocks.append(f'{label} [{n.node_id}] ({title}):\n"""\n{n.text[:SOURCE_CHARS]}\n"""')
    user = "\n".join(
        [
            "Write ONE evaluation question that a member of the public might ask, "
            "answerable only from the text below.",
            "",
            f"Question type: {c.qtype}. {TYPE_GUIDANCE[c.qtype]}",
            "",
            "Rules:",
            "- The answer must be fully contained in the text; do not use outside knowledge.",
            "- Paraphrase: do not copy distinctive phrases from the text; use plain, everyday "
            "wording a citizen would use.",
            f"- {section_rule}",
            f"- gold_answer: a correct answer in at most {MAX_ANSWER_WORDS} words, based only on the text.",
            '- difficulty: "easy", "medium" or "hard".',
            "",
            "Reply with a JSON object with exactly these keys:",
            f'{{"question": "...", "gold_answer": "...", "type": "{c.qtype}", "difficulty": "..."}}',
            "",
            *blocks,
        ]
    )
    system = (
        "You write evaluation questions for a retrieval benchmark over Indian consumer "
        "protection law. Reply with a single JSON object and nothing else."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _content_words(text: str) -> list[str]:
    joined = re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", "", normalise(text))
    return [w for w in re.findall(r"[a-z0-9]+", joined) if w not in STOPWORDS]


def lexical_overlap(question: str, source_text: str) -> float:
    """Fraction of the question's content tokens that also occur in the source text."""
    q = _content_words(question)
    if not q:
        return 0.0
    src = set(_content_words(source_text))
    return sum(1 for w in q if w in src) / len(q)


def draft_flags(qtype: str, question: str, answer: str, overlap: float) -> list[str]:
    flags = []
    if overlap > OVERLAP_FLAG_THRESHOLD:
        flags.append("copy_like")
    if len(answer.split()) > MAX_ANSWER_WORDS:
        flags.append("answer_too_long")
    if qtype != "cross_reference" and SECTION_NUMBER_RE.search(question):
        flags.append("mentions_provision_number")
    return flags


def load_drafts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save_drafts(path: Path, drafts: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for d in drafts:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    tmp.replace(path)


def next_draft_number(drafts: Sequence[dict[str, Any]]) -> int:
    nums = [
        int(m.group(1))
        for d in drafts
        if (m := re.fullmatch(r"d(\d+)", str(d.get("draft_id", ""))))
    ]
    return max(nums, default=0) + 1


def draft_questions(
    llm: ChatModel,
    candidates: Sequence[Candidate],
    doc_titles: dict[str, str],
    start_number: int,
    seed: int,
    base_url: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ask the LLM for one draft per candidate. Returns (drafts, failures)."""
    drafts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    number = start_number
    for c in candidates:
        messages = build_messages(c, doc_titles)
        try:
            data, resp = llm.chat_json(messages)
        except LLMJSONError as exc:
            failures.append({"node_ids": c.node_ids, "type": c.qtype, "error": str(exc)[:200]})
            continue
        question = str(data.get("question", "")).strip()
        answer = str(data.get("gold_answer", "")).strip()
        if len(question) < 5 or not answer:
            failures.append({"node_ids": c.node_ids, "type": c.qtype, "error": "missing keys"})
            continue
        difficulty = str(data.get("difficulty", "medium")).strip().lower()
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"
        source_text = "\n\n".join(f"[{n.node_id}] {n.text[:SOURCE_CHARS]}" for n in c.nodes)
        overlap = lexical_overlap(question, "\n".join(n.text for n in c.nodes))
        flags = draft_flags(c.qtype, question, answer, overlap)
        llm_type = str(data.get("type", c.qtype))
        if llm_type != c.qtype:
            flags.append(f"model_returned_type:{llm_type}")
        drafts.append(
            {
                "draft_id": f"d{number:04d}",
                "status": "pending",
                "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                "type": c.qtype,
                "difficulty": difficulty,
                "question": question,
                "gold_answer": answer,
                "gold_node_ids": c.node_ids,
                "source_text": source_text,
                "llm_model": resp.model or llm.model,
                "llm_base_url": base_url,
                "lexical_overlap": round(overlap, 3),
                "overlap_flag": overlap > OVERLAP_FLAG_THRESHOLD,
                "flags": flags,
                "seed": seed,
                "prompt_version": PROMPT_VERSION,
            }
        )
        number += 1
    return drafts, failures


__all__ = [
    "TYPE_MIX",
    "Candidate",
    "allocate",
    "build_messages",
    "build_pools",
    "draft_questions",
    "lexical_overlap",
    "load_drafts",
    "next_draft_number",
    "save_drafts",
    "stratified_sample",
]
