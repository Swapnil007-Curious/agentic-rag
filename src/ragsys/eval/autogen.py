"""Automatic construction of evaluation questions with a local LLM and automatic filters.

The model used here (by default qwen2.5:3b through Ollama) is small and makes factual mistakes,
so an item is kept only when it passes every automatic check.

Answerable questions
--------------------
1. Candidates come from the existing drafting code (:mod:`ragsys.eval.drafting`): same type mix,
   stratified sampling of target nodes and the same drafting prompt; ``gold_node_ids`` are the
   source node(s).
2. Filters, applied in this order (each logged with pass/fail counts and reasons):

   ``format``                 valid JSON with question and answer; question 6-40 words; answer
                              at most 40 words
   ``provision_number``       no section/rule/clause numbers in the question (except
                              cross_reference questions)
   ``copy``                   lexical overlap with the source text at most 0.6; otherwise the
                              LLM paraphrases the question once and it is checked again
   ``numbers``                every number in the answer (digits or words) occurs in the source
   ``support``                LLM judge: {"supported": true|false, "corrected_answer": ...};
                              only supported=true is kept, corrections are never applied
   ``answerable_from_source`` the LLM answers from the source alone; dropped when that answer
                              states different numbers or names a different entity of the same
                              kind (deterministic comparison, :mod:`ragsys.eval.checks`)
   ``near_duplicate``         bge-small cosine > 0.90 to an existing or earlier question

3. Up to ``target`` survivors are selected, stratified by type (the drafting type mix) and as
   evenly over documents as availability allows; any shortfall is reported.

Unanswerable questions
----------------------
Candidates from two methods: out-of-scope topics (consumer-style questions on subjects this
corpus does not contain) and perturbations of accepted answerable questions (key entity or
scope moved outside the corpus). Filters: format, no section/rule number of this corpus,
near_duplicate (also against all answerable questions), and a retrieval check: the hybrid
(BM25 + bge-small, RRF) retriever's top-5 structural chunks are shown to the LLM, which must
return {"answered": false}. Selection is balanced over the two methods; the gold answer is a
one-line reason derived from the generation method.

Every LLM call goes through the client's on-disk cache, so an interrupted run resumes without
repeating calls.
"""

from __future__ import annotations

import random
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from ragsys.eval import drafting
from ragsys.eval.checks import (
    entity_conflicts,
    mentions_provision_number,
    n_words,
    near_duplicates,
    number_conflict,
    numbers_missing_from_source,
    section_rule_numbers,
)
from ragsys.eval.drafting import (
    SOURCE_CHARS,
    TYPE_MIX,
    Candidate,
    allocate,
    build_pools,
    draft_questions,
    lexical_overlap,
    stratified_sample,
)
from ragsys.eval.schema import AUTO_ORIGIN, Difficulty, EvalQuestion
from ragsys.llm.client import ChatModel, LLMJSONError, LLMResponse, Message
from ragsys.retrieval.config import RetrievalConfig
from ragsys.retrieval.corpus import NodeGraph
from ragsys.retrieval.pipeline import RetrievalResources, Retriever
from ragsys.structure.base import Node
from ragsys.structure.xrefs import XRef
from ragsys.utils.memory import fmt_duration

PROMPT_VERSION = "auto-v1"
MIN_QUESTION_WORDS = 6
MAX_QUESTION_WORDS = 40
MAX_ANSWER_WORDS = 40
COPY_THRESHOLD = 0.6
DUPLICATE_THRESHOLD = 0.90
RETRIEVAL_CHECK_K = 5
PASSAGE_CHARS = 1200
OUT_OF_SCOPE_PER_CALL = 3

OutFn = Callable[[str], None]
T = TypeVar("T")

_JSON_ONLY = "Reply with a single JSON object and nothing else."
SYSTEM_PARAPHRASE = f"You rewrite questions in plain, everyday English. {_JSON_ONLY}"
SYSTEM_SUPPORT = (
    f"You are a strict fact checker for a legal question-answering benchmark. {_JSON_ONLY}"
)
SYSTEM_ANSWER = f"You answer questions using only the legal text you are given. {_JSON_ONLY}"
SYSTEM_OUT_OF_SCOPE = (
    "You write realistic questions that people in India ask about legal and consumer matters. "
    f"{_JSON_ONLY}"
)
SYSTEM_PERTURB = f"You rewrite questions so that they ask about a different subject. {_JSON_ONLY}"
SYSTEM_RETRIEVAL = f"You judge whether retrieved passages answer a question. {_JSON_ONLY}"

CORPUS_NAME = "the Consumer Protection Act, 2019 and its four Rules"

# (topic id, description given to the LLM, short subject used in the gold reason)
OUT_OF_SCOPE_TOPICS: tuple[tuple[str, str, str], ...] = (
    (
        "state_fees",
        "court fees, stamp duty or registration charges that apply only in one particular "
        "Indian state, such as Maharashtra, Karnataka or Tamil Nadu",
        "fees that apply only in a particular state",
    ),
    (
        "state_procedures",
        "procedures run by one particular state government, such as a state's own consumer "
        "helpline, ration card or electricity complaint process in Kerala, Punjab or West Bengal",
        "procedures of a particular state government",
    ),
    (
        "ipc_bns",
        "criminal offences such as cheating, fraud or forgery and their punishments under the "
        "Indian Penal Code or the Bharatiya Nyaya Sanhita",
        "criminal law (the Indian Penal Code / Bharatiya Nyaya Sanhita)",
    ),
    (
        "motor_vehicles",
        "traffic fines, driving licences, vehicle registration and accident insurance claims "
        "under the Motor Vehicles Act",
        "the Motor Vehicles Act",
    ),
    (
        "rti",
        "filing a Right to Information (RTI) application: fees, time limits for replies and "
        "appeals under the RTI Act",
        "the Right to Information Act",
    ),
    (
        "income_tax",
        "income tax returns, TDS, deductions and tax refunds under the Income-tax Act",
        "the Income-tax Act",
    ),
    (
        "contract_act",
        "when a contract is valid or void, agreements with minors, and remedies for breach of "
        "contract under the Indian Contract Act, 1872",
        "the Indian Contract Act, 1872",
    ),
    (
        "cpa1986_limits",
        "how things worked under the old, repealed Consumer Protection Act, 1986, such as the "
        "money limits and time limits of the old District Forum",
        "the repealed Consumer Protection Act, 1986",
    ),
    (
        "cpa1986_bodies",
        "how the District Forum, State Commission and National Commission were set up and "
        "staffed under the repealed Consumer Protection Act, 1986",
        "the repealed Consumer Protection Act, 1986",
    ),
    (
        "supreme_court",
        "what the Supreme Court of India decided in specific consumer protection judgments",
        "court judgments (case law)",
    ),
    (
        "ncdrc_cases",
        "outcomes of specific landmark cases decided by the National Consumer Disputes Redressal "
        "Commission, such as medical negligence or builder delay cases",
        "decisions in specific cases (case law)",
    ),
    (
        "addresses",
        "the postal address, office hours or location of a particular District Commission or "
        "State Commission office",
        "office addresses and opening hours",
    ),
    (
        "phone_numbers",
        "phone numbers, email addresses or websites of consumer helplines and consumer "
        "commission offices",
        "phone numbers and contact details",
    ),
    (
        "case_timelines",
        "the status, dates or timeline of a specific real consumer case against a named company",
        "the timeline of a specific real case",
    ),
    (
        "banking_ombudsman",
        "complaints about banks under the Reserve Bank of India's Banking Ombudsman or "
        "Integrated Ombudsman Scheme",
        "the RBI banking ombudsman schemes",
    ),
)

# perturbation kind -> (concrete targets, template of the target description)
PERTURBATION_TARGETS: dict[str, tuple[tuple[str, ...], str]] = {
    "another_law": (
        (
            "the Right to Information Act, 2005",
            "the Motor Vehicles Act, 1988",
            "the Income-tax Act, 1961",
            "the Indian Contract Act, 1872",
            "the Bharatiya Nyaya Sanhita, 2023",
            "the repealed Consumer Protection Act, 1986",
            "the Real Estate (Regulation and Development) Act, 2016",
            "the Banking Regulation Act, 1949",
        ),
        "{x}",
    ),
    "specific_state": (
        ("Kerala", "Maharashtra", "Punjab", "Assam", "Gujarat", "Tamil Nadu", "Bihar", "Goa"),
        "a rule, fee or practice that applies only in the state of {x}",
    ),
    "specific_company": (
        (
            "Flipkart",
            "Amazon India",
            "Tata Motors",
            "Reliance Jio",
            "HDFC Bank",
            "Zomato",
            "IndiGo Airlines",
            "Maruti Suzuki",
        ),
        "the company {x}'s own policy or practice",
    ),
    "specific_year": (
        ("2017", "2022", "2015", "2023", "2018", "2021"),
        "figures, events or cases from the year {x}",
    ),
}


# --------------------------------------------------------------------------- bookkeeping


@dataclass
class FilterStat:
    name: str
    description: str
    n_in: int = 0
    n_pass: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def record(self, ok: bool, reason: str = "", example: dict[str, Any] | None = None) -> bool:
        self.n_in += 1
        if ok:
            self.n_pass += 1
        else:
            self.reasons[reason or "failed"] += 1
            if example is not None and len(self.examples) < 8:
                self.examples.append({"reason": reason, **example})
        return ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "in": self.n_in,
            "pass": self.n_pass,
            "fail": self.n_in - self.n_pass,
            "pass_rate": round(self.n_pass / self.n_in, 4) if self.n_in else None,
            "fail_reasons": dict(self.reasons.most_common()),
            "fail_examples": self.examples,
        }


class Progress:
    """Per-item progress lines with an ETA from the items that needed a real LLM call."""

    def __init__(self, phase: str, total: int, out: OutFn) -> None:
        self.phase, self.total, self.out = phase, total, out
        self.done = 0
        self.t0 = self._last = time.perf_counter()
        self.slow: list[float] = []

    def step(self, label: str) -> None:
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        self.done += 1
        if dt > 0.25:  # cache hits and deterministic steps are near-instant
            self.slow.append(dt)
        avg = statistics.fmean(self.slow) if self.slow else 0.0
        eta = fmt_duration((self.total - self.done) * avg)
        self.out(
            f"  [{self.phase} {self.done}/{self.total}] {label}  "
            f"(elapsed {fmt_duration(now - self.t0)}, ETA {eta})"
        )


class MeteredLLM:
    """Forwards to a :class:`ChatModel` and counts calls, cache hits and latency per phase."""

    def __init__(self, inner: ChatModel) -> None:
        self.inner = inner
        self.model = inner.model
        self.phase = "setup"
        self.stats: dict[str, dict[str, float]] = {}

    def _note(self, resp: LLMResponse) -> None:
        s = self.stats.setdefault(
            self.phase, {"calls": 0, "cached": 0, "latency_s": 0.0, "completion_tokens": 0}
        )
        s["calls"] += 1
        s["cached"] += int(resp.cached)
        s["latency_s"] += resp.latency_ms / 1000.0
        s["completion_tokens"] += int(resp.usage.get("completion_tokens", 0))

    def chat(self, messages: Sequence[Message], *, json_mode: bool = False) -> LLMResponse:
        resp = self.inner.chat(messages, json_mode=json_mode)
        self._note(resp)
        return resp

    def chat_json(self, messages: Sequence[Message]) -> tuple[dict[str, Any], LLMResponse]:
        try:
            data, resp = self.inner.chat_json(messages)
        except LLMJSONError:
            self._count_failed_json()
            raise
        self._note(resp)
        return data, resp

    def _count_failed_json(self) -> None:
        s = self.stats.setdefault(
            self.phase, {"calls": 0, "cached": 0, "latency_s": 0.0, "completion_tokens": 0}
        )
        s["calls"] += 1

    def summary(self) -> dict[str, Any]:
        total = {"calls": 0.0, "cached": 0.0, "latency_s": 0.0, "completion_tokens": 0.0}
        for s in self.stats.values():
            for k in total:
                total[k] += s[k]
        rnd = {k: (round(v, 1) if k == "latency_s" else int(v)) for k, v in total.items()}
        return {
            **rnd,
            "by_phase": {
                p: {k: (round(v, 1) if k == "latency_s" else int(v)) for k, v in s.items()}
                for p, s in self.stats.items()
            },
        }


# --------------------------------------------------------------------------- items


@dataclass
class AnswerableItem:
    cand: Candidate
    number: int  # draft number (d0001, ...) in candidate order
    question: str = ""
    gold_answer: str = ""
    difficulty: str = "medium"
    overlap: float = 0.0
    overlap_initial: float = 0.0
    original_question: str | None = None
    filters: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    qid: str | None = None

    @property
    def qtype(self) -> str:
        return self.cand.qtype

    @property
    def doc_id(self) -> str:
        return self.cand.doc_id

    @property
    def node_ids(self) -> list[str]:
        return self.cand.node_ids

    def brief(self) -> dict[str, Any]:
        return {
            "draft": f"d{self.number:04d}",
            "type": self.qtype,
            "nodes": self.node_ids,
            "question": self.question,
            "gold_answer": self.gold_answer,
        }


@dataclass
class UnanswerableItem:
    method: str  # "out_of_scope" | "perturbation"
    group: str  # topic id or perturbation kind (balanced at selection)
    question: str
    reason: str  # becomes gold_answer
    index: int  # generation order
    filters: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def brief(self) -> dict[str, Any]:
        return {"method": self.method, "group": self.group, "question": self.question}


# --------------------------------------------------------------------------- prompts


def source_blocks(c: Candidate, doc_titles: dict[str, str]) -> str:
    """The provision text(s) exactly as the drafting prompt shows them."""
    labels = ["PROVISION", "SECOND PROVISION"]
    blocks = []
    for label, n in zip(labels, c.nodes, strict=False):
        title = doc_titles.get(n.doc_id, n.doc_title)
        blocks.append(f'{label} [{n.node_id}] ({title}):\n"""\n{n.text[:SOURCE_CHARS]}\n"""')
    return "\n".join(blocks)


def _messages(system: str, user: str) -> list[Message]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def paraphrase_messages(question: str, qtype: str) -> list[Message]:
    numbers_rule = (
        "You may keep a section or rule number if the question needs it."
        if qtype == "cross_reference"
        else "Do not mention section, rule, sub-section or clause numbers."
    )
    user = "\n".join(
        [
            "Rewrite this question the way an ordinary citizen would ask it:",
            f'"{question}"',
            "",
            "Rules:",
            "- Keep exactly the same meaning, so that the same answer still applies.",
            "- Use plain, everyday words instead of legal terms; avoid wording copied from "
            "legal texts.",
            f"- {numbers_rule}",
            f"- Between {MIN_QUESTION_WORDS} and {MAX_QUESTION_WORDS} words.",
            "",
            'Reply with a JSON object: {"question": "..."}',
        ]
    )
    return _messages(SYSTEM_PARAPHRASE, user)


def support_messages(c: Candidate, titles: dict[str, str], q: str, a: str) -> list[Message]:
    user = "\n".join(
        [
            "Check a question-answer pair against the legal text.",
            "",
            "LEGAL TEXT:",
            source_blocks(c, titles),
            "",
            f"QUESTION: {q}",
            f"PROPOSED ANSWER: {a}",
            "",
            "Is the proposed answer correct and fully supported by the legal text? Every fact, "
            "number, authority and condition in the answer must be stated in the text, and the "
            "answer must answer the question. If the answer adds anything that is not in the "
            "text, or gets anything wrong, it is not supported.",
            "",
            "Reply with a JSON object with two keys:",
            '"supported": true or false,',
            '"corrected_answer": a short correct answer based only on the text if "supported" '
            'is false, otherwise "".',
        ]
    )
    return _messages(SYSTEM_SUPPORT, user)


def answer_messages(c: Candidate, titles: dict[str, str], q: str) -> list[Message]:
    user = "\n".join(
        [
            "Answer the question using only the legal text below.",
            "",
            "LEGAL TEXT:",
            source_blocks(c, titles),
            "",
            f"QUESTION: {q}",
            "",
            "Rules:",
            "- Use only the legal text; do not use outside knowledge.",
            f"- Answer in at most {MAX_ANSWER_WORDS} words.",
            '- If the text does not contain the answer, answer "not stated in the text".',
            "",
            'Reply with a JSON object: {"answer": "..."}',
        ]
    )
    return _messages(SYSTEM_ANSWER, user)


def out_of_scope_messages(topic: str, n: int = OUT_OF_SCOPE_PER_CALL) -> list[Message]:
    slots = ", ".join(['"..."'] * n)
    user = "\n".join(
        [
            f"Write {n} different questions that an ordinary person in India might really ask "
            "about this topic:",
            topic,
            "",
            "Rules:",
            "- Each question must be specific, self-contained and between 8 and 30 words long.",
            "- Use plain, everyday language, as a consumer or citizen would.",
            "- Do not mention section or rule numbers.",
            f"- The {n} questions must be about different aspects of the topic.",
            "",
            f'Reply with a JSON object: {{"questions": [{slots}]}}',
        ]
    )
    return _messages(SYSTEM_OUT_OF_SCOPE, user)


def perturbation_messages(question: str, target: str) -> list[Message]:
    user = "\n".join(
        [
            "Here is a question about Indian consumer protection law:",
            f'"{question}"',
            "",
            f"Rewrite it into a new question about {target}.",
            "Rules:",
            f"- The new question must clearly be about {target}, not about the Consumer "
            "Protection Act, 2019 or its Rules.",
            "- Change the subject of the question; do not just add the name to the end of the "
            "old question.",
            "- Plain, everyday language; between 8 and 30 words.",
            "- Do not mention section or rule numbers.",
            "",
            'Reply with a JSON object: {"question": "..."}',
        ]
    )
    return _messages(SYSTEM_PERTURB, user)


def retrieval_check_messages(question: str, passages: Sequence[tuple[str, str]]) -> list[Message]:
    lines = [f"QUESTION: {question}", "", "PASSAGES (from Indian consumer protection law):"]
    for i, (heading, text) in enumerate(passages, start=1):
        lines += [f"[{i}] {heading}", text[:PASSAGE_CHARS], ""]
    lines += [
        "Do these passages contain the answer to this exact question? Answer true only if the "
        "passages themselves state the answer to the question as asked, including the specific "
        "law, state, company, year, person or case it asks about. Answer false if the passages "
        "are only about a related topic or a different law, or do not give the specific "
        "information asked for.",
        "",
        'Reply with a JSON object: {"answered": true} or {"answered": false}',
    ]
    return _messages(SYSTEM_RETRIEVAL, "\n".join(lines))


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no"):
        return False
    return None


_LEAD_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|q\d*[:.)])\s*", re.IGNORECASE)


def clean_question(text: Any) -> str:
    """Strip list markers, surrounding quotes and whitespace from a generated question."""
    if isinstance(text, dict):
        text = text.get("question", "")
    s = _LEAD_RE.sub("", str(text)).strip()
    return s.strip("\"'“”‘’ ").strip()


def _refusal(answer: str) -> bool:
    return bool(
        re.search(
            r"\bnot (?:stated|mentioned|specified|provided|given)\b|\bdoes not (?:say|state|"
            r"mention|specify)\b|\bno information\b|\bcannot be determined\b",
            answer,
            re.IGNORECASE,
        )
    )


# --------------------------------------------------------------------------- selection


def type_quotas(available: dict[str, int], target: int, mix: dict[str, int]) -> dict[str, int]:
    """Quotas per type following ``mix``; a type with too few items gives its remainder to the
    other types (largest remainder by mix weight), repeated until everything fits."""
    quotas = allocate(target, mix)
    while True:
        over = {
            t: quotas[t] - available.get(t, 0) for t in quotas if quotas[t] > available.get(t, 0)
        }
        if not over:
            return quotas
        for t in over:
            quotas[t] = available.get(t, 0)
        spare = {t: mix[t] for t in quotas if available.get(t, 0) > quotas[t] and mix[t] > 0}
        if not spare:
            return quotas
        for t, k in allocate(sum(over.values()), spare).items():
            quotas[t] += k


def pick_balanced(
    groups: dict[str, list[T]], k: int, global_counts: Counter[str] | None = None
) -> list[T]:
    """Take ``k`` items, each time from the group with the fewest picks so far (ties: fewest
    picks overall in ``global_counts``, then group name); order within a group is kept."""
    counts = global_counts if global_counts is not None else Counter()
    queues = {g: list(v) for g, v in groups.items() if v}
    taken: Counter[str] = Counter()
    out: list[T] = []
    while len(out) < k and queues:
        g = min(queues, key=lambda x: (taken[x], counts[x], x))
        out.append(queues[g].pop(0))
        taken[g] += 1
        counts[g] += 1
        if not queues[g]:
            del queues[g]
    return out


def select_answerable(
    items: Sequence[AnswerableItem], target: int, mix: dict[str, int] = TYPE_MIX
) -> tuple[list[AnswerableItem], dict[str, Any]]:
    by_type: dict[str, list[AnswerableItem]] = defaultdict(list)
    for it in items:
        by_type[it.qtype].append(it)
    available = {t: len(by_type.get(t, [])) for t in mix}
    quotas = type_quotas(available, target, mix)
    doc_counts: Counter[str] = Counter()
    chosen: list[AnswerableItem] = []
    for t in mix:
        per_doc: dict[str, list[AnswerableItem]] = defaultdict(list)
        for it in by_type.get(t, []):
            per_doc[it.doc_id].append(it)
        chosen += pick_balanced(per_doc, quotas[t], doc_counts)
    chosen.sort(key=lambda it: it.number)
    mix_quota = allocate(target, mix)
    report = {
        "target": target,
        "available": len(items),
        "selected": len(chosen),
        "shortfall": max(0, target - len(chosen)),
        "by_type": {
            t: {
                "mix_target": mix_quota[t],
                "available": available[t],
                "quota_after_redistribution": quotas[t],
                "selected": sum(1 for it in chosen if it.qtype == t),
            }
            for t in mix
        },
        "by_document": dict(sorted(Counter(it.doc_id for it in chosen).items())),
    }
    return chosen, report


def select_unanswerable(
    items: Sequence[UnanswerableItem], target: int
) -> tuple[list[UnanswerableItem], dict[str, Any]]:
    methods = {"out_of_scope": 1, "perturbation": 1}
    by_method: dict[str, list[UnanswerableItem]] = defaultdict(list)
    for it in items:
        by_method[it.method].append(it)
    available = {m: len(by_method.get(m, [])) for m in methods}
    quotas = type_quotas(available, target, methods)
    chosen: list[UnanswerableItem] = []
    for m in methods:
        groups: dict[str, list[UnanswerableItem]] = defaultdict(list)
        for it in by_method.get(m, []):
            groups[it.group].append(it)
        chosen += pick_balanced(groups, quotas[m])
    chosen.sort(key=lambda it: it.index)
    report = {
        "target": target,
        "available": len(items),
        "selected": len(chosen),
        "shortfall": max(0, target - len(chosen)),
        "by_method": {
            m: {
                "available": available[m],
                "quota": quotas[m],
                "selected": sum(1 for it in chosen if it.method == m),
            }
            for m in methods
        },
        "by_group": dict(sorted(Counter(f"{it.method}:{it.group}" for it in chosen).items())),
    }
    return chosen, report


# --------------------------------------------------------------------------- builder


@dataclass
class BuildResult:
    questions: list[EvalQuestion]
    report: dict[str, Any]


def next_id_number(existing_ids: Sequence[str]) -> int:
    nums = [int(m.group(1)) for i in existing_ids if (m := re.fullmatch(r"q(\d+)", i))]
    return max(nums, default=0) + 1


class AutoEvalBuilder:
    """Runs the whole pipeline; see the module docstring."""

    def __init__(
        self,
        llm: ChatModel,
        resources: RetrievalResources,
        nodes: Sequence[Node],
        xrefs: Sequence[XRef],
        doc_titles: dict[str, str],
        existing: Sequence[EvalQuestion],
        *,
        seed: int = 0,
        dense_model: str = "bge-small",
        out: OutFn = print,
    ) -> None:
        self.llm = MeteredLLM(llm)
        self.resources = resources
        self.nodes = list(nodes)
        self.xrefs = list(xrefs)
        self.graph = NodeGraph(self.nodes)
        self.titles = doc_titles
        self.existing = [q for q in existing if q.origin != AUTO_ORIGIN]
        self.seed = seed
        self.dense_model = dense_model
        self.out = out
        self.model_name = llm.model
        self.filters: dict[str, FilterStat] = {}
        self.corpus_numbers = {
            (n.number or "").lower() for n in self.nodes if n.level in ("section", "rule")
        } - {""}
        self.paraphrase_attempted = 0
        self.paraphrase_rescued = 0
        self.answer_refusals = 0
        self.generation_failures: Counter[str] = Counter()

    # ------------------------------------------------------------ helpers

    def _stat(self, key: str, description: str) -> FilterStat:
        if key not in self.filters:
            self.filters[key] = FilterStat(key.split(":", 1)[-1], description)
        return self.filters[key]

    def _encoder(self) -> Any:
        return self.resources.encoder(self.dense_model, True)

    def _ask(self, messages: list[Message]) -> dict[str, Any] | None:
        try:
            data, _resp = self.llm.chat_json(messages)
        except LLMJSONError:
            return None
        return data

    # ------------------------------------------------------------ answerable

    def sample_candidates(self, n: int) -> tuple[list[Candidate], dict[str, Any]]:
        exclude = {g for q in self.existing for g in q.gold_node_ids}
        pools = build_pools(self.nodes, self.xrefs, self.graph, exclude=exclude)
        plan = allocate(n, TYPE_MIX)
        sample, shortfall = stratified_sample(pools, plan, seed=self.seed)
        topped_up: dict[str, int] = {}
        if shortfall:
            # give the missing targets to the general "factual" pool so n candidates are drafted
            adjusted = dict(plan)
            for t, k in shortfall.items():
                adjusted[t] -= k
            adjusted["factual"] += sum(shortfall.values())
            topped_up = {"factual": sum(shortfall.values())}
            sample, _short = stratified_sample(pools, adjusted, seed=self.seed)
        info = {
            "requested": n,
            "type_plan": plan,
            "pool_sizes": {t: len(pools[t]) for t in TYPE_MIX},
            "sampling_shortfall": shortfall,
            "topped_up": topped_up,
            "sampled": len(sample),
            "sampled_by_type": dict(Counter(c.qtype for c in sample)),
            "sampled_by_document": dict(sorted(Counter(c.doc_id for c in sample).items())),
            "excluded_nodes_of_existing_questions": len(exclude),
        }
        return sample, info

    def draft(self, candidates: Sequence[Candidate]) -> list[AnswerableItem]:
        self.llm.phase = "draft"
        stat = self._stat(
            "a:format",
            f"valid JSON with question and gold_answer; question {MIN_QUESTION_WORDS}-"
            f"{MAX_QUESTION_WORDS} words; answer <= {MAX_ANSWER_WORDS} words",
        )
        prog = Progress("draft", len(candidates), self.out)
        items: list[AnswerableItem] = []
        for i, c in enumerate(candidates, start=1):
            drafts, failures = draft_questions(self.llm, [c], self.titles, i, self.seed)
            it = AnswerableItem(c, i)
            if failures:
                err = str(failures[0].get("error", ""))
                reason = "missing_fields" if err == "missing keys" else "invalid_json"
                stat.record(False, reason, {"draft": f"d{i:04d}", "nodes": c.node_ids})
                prog.step(f"{c.qtype} {c.node_ids[0]}: {reason}")
                continue
            d = drafts[0]
            it.question, it.gold_answer = str(d["question"]), str(d["gold_answer"])
            it.difficulty = str(d["difficulty"])
            q_words, a_words = n_words(it.question), n_words(it.gold_answer)
            reason = ""
            if q_words < MIN_QUESTION_WORDS:
                reason = "question_too_short"
            elif q_words > MAX_QUESTION_WORDS:
                reason = "question_too_long"
            elif a_words > MAX_ANSWER_WORDS:
                reason = "answer_too_long"
            if stat.record(not reason, reason, it.brief()):
                it.filters.append("format")
                items.append(it)
            prog.step(f"{c.qtype} {c.node_ids[0]}: {reason or 'ok'}")
        return items

    def filter_provision_numbers(self, items: Sequence[AnswerableItem]) -> list[AnswerableItem]:
        stat = self._stat(
            "a:provision_number",
            "no section/rule/clause number in the question unless type is cross_reference",
        )
        kept = []
        for it in items:
            bad = it.qtype != "cross_reference" and mentions_provision_number(it.question)
            if stat.record(not bad, "mentions_provision_number", it.brief()):
                it.filters.append("provision_number")
                kept.append(it)
        return kept

    def filter_copy(self, items: Sequence[AnswerableItem]) -> list[AnswerableItem]:
        self.llm.phase = "paraphrase"
        stat = self._stat(
            "a:copy",
            f"lexical overlap with the source <= {COPY_THRESHOLD}; one LLM paraphrase attempt "
            "otherwise, then re-checked (also length and provision numbers)",
        )
        need = [it for it in items if self._overlap(it, it.question) > COPY_THRESHOLD]
        prog = Progress("paraphrase", len(need), self.out)
        kept = []
        for it in items:
            it.overlap_initial = it.overlap = self._overlap(it, it.question)
            if it.overlap <= COPY_THRESHOLD:
                stat.record(True)
                it.filters.append("copy")
                kept.append(it)
                continue
            self.paraphrase_attempted += 1
            data = self._ask(paraphrase_messages(it.question, it.qtype))
            new = clean_question(data.get("question", "")) if data else ""
            reason = ""
            if not data:
                reason = "paraphrase_invalid_json"
            elif not (MIN_QUESTION_WORDS <= n_words(new) <= MAX_QUESTION_WORDS):
                reason = "paraphrase_bad_length"
            elif it.qtype != "cross_reference" and mentions_provision_number(new):
                reason = "paraphrase_mentions_provision_number"
            else:
                ov = self._overlap(it, new)
                if ov > COPY_THRESHOLD:
                    reason = "copy_like_after_paraphrase"
                else:
                    it.original_question, it.question, it.overlap = it.question, new, ov
            ex = {**it.brief(), "paraphrase": new, "overlap_before": round(it.overlap_initial, 3)}
            if stat.record(not reason, reason, ex):
                self.paraphrase_rescued += 1
                it.filters.append("copy")
                kept.append(it)
            prog.step(f"{it.qtype} {it.node_ids[0]}: {reason or 'paraphrased'}")
        return kept

    def _overlap(self, it: AnswerableItem, question: str) -> float:
        return lexical_overlap(question, "\n".join(n.text for n in it.cand.nodes))

    def filter_numbers(self, items: Sequence[AnswerableItem]) -> list[AnswerableItem]:
        stat = self._stat(
            "a:numbers",
            "every number in the gold answer (digits or words) occurs in the source text, "
            "headings or document titles (numbers inside provision references may also match "
            "the source node ids)",
        )
        kept = []
        for it in items:
            src = [n.text[:SOURCE_CHARS] for n in it.cand.nodes]
            src += [n.heading or "" for n in it.cand.nodes]
            src += [self.titles.get(n.doc_id, n.doc_title) for n in it.cand.nodes]
            missing = numbers_missing_from_source(it.gold_answer, src, it.node_ids)
            if stat.record(not missing, "number_not_in_source", {**it.brief(), "missing": missing}):
                it.filters.append("numbers")
                kept.append(it)
        return kept

    def filter_support(self, items: Sequence[AnswerableItem]) -> list[AnswerableItem]:
        self.llm.phase = "support"
        stat = self._stat(
            "a:support",
            'LLM judge on source + question + answer returns {"supported": true}; corrected '
            "answers are recorded but never applied",
        )
        prog = Progress("support", len(items), self.out)
        kept = []
        for it in items:
            data = self._ask(support_messages(it.cand, self.titles, it.question, it.gold_answer))
            verdict = _as_bool(data.get("supported")) if data else None
            corrected = str(data.get("corrected_answer", "") or "") if data else ""
            it.meta["support_check"] = {"supported": verdict, "corrected_answer": corrected}
            reason = "invalid_json" if data is None else ("" if verdict else "not_supported")
            if stat.record(not reason, reason, {**it.brief(), "corrected_answer": corrected}):
                it.filters.append("support")
                kept.append(it)
            prog.step(f"{it.qtype} {it.node_ids[0]}: {reason or 'supported'}")
        return kept

    def filter_answerability(self, items: Sequence[AnswerableItem]) -> list[AnswerableItem]:
        self.llm.phase = "answer_from_source"
        stat = self._stat(
            "a:answerable_from_source",
            "the LLM answers from the source text alone; dropped when its answer states other "
            "numbers than the gold answer or names a different entity of the same kind "
            "(capitalised terms)",
        )
        prog = Progress("answer", len(items), self.out)
        kept = []
        for it in items:
            data = self._ask(answer_messages(it.cand, self.titles, it.question))
            answer = str(data.get("answer", "") or "") if data else ""
            it.meta["independent_answer"] = answer
            reason = ""
            ex = {**it.brief(), "independent_answer": answer}
            if data is None:
                reason = "invalid_json"
            else:
                missing, extra = number_conflict(it.gold_answer, answer)
                conflicts = entity_conflicts(it.gold_answer, answer)
                if missing:
                    reason = "number_conflict"
                    ex["numbers"] = {"gold_only": missing, "llm_only": extra}
                elif conflicts:
                    reason = "entity_conflict"
                    ex["entities"] = conflicts
            refused = bool(answer) and _refusal(answer)
            self.answer_refusals += int(refused)
            it.meta["independent_answer_refused"] = refused
            if stat.record(not reason, reason, ex):
                it.filters.append("answerable_from_source")
                kept.append(it)
            prog.step(f"{it.qtype} {it.node_ids[0]}: {reason or 'consistent'}")
        return kept

    def filter_duplicates_answerable(self, items: Sequence[AnswerableItem]) -> list[AnswerableItem]:
        stat = self._stat(
            "a:near_duplicate",
            f"bge-small cosine <= {DUPLICATE_THRESHOLD} to existing questions and to earlier "
            "candidates (keep the first)",
        )
        res = near_duplicates(
            [it.question for it in items],
            self._encoder(),
            DUPLICATE_THRESHOLD,
            reference=[q.question for q in self.existing],
        )
        kept = []
        for it, ok, sim, near in zip(items, res.keep, res.max_similarity, res.nearest, strict=True):
            it.meta["max_similarity"] = sim
            if stat.record(
                ok, "near_duplicate", {**it.brief(), "similarity": sim, "nearest": near}
            ):
                it.filters.append("near_duplicate")
                kept.append(it)
        return kept

    # ------------------------------------------------------------ unanswerable

    def generate_out_of_scope(self, n: int) -> list[UnanswerableItem]:
        self.llm.phase = "out_of_scope"
        n_calls = -(-n // OUT_OF_SCOPE_PER_CALL)
        topics = [OUT_OF_SCOPE_TOPICS[i % len(OUT_OF_SCOPE_TOPICS)] for i in range(n_calls)]
        prog = Progress("out-of-scope", len(topics), self.out)
        out: list[UnanswerableItem] = []
        for tid, desc, subject in topics:
            want = min(OUT_OF_SCOPE_PER_CALL, n - len(out))
            data = self._ask(out_of_scope_messages(desc, OUT_OF_SCOPE_PER_CALL))
            qs = data.get("questions") if data else None
            got = 0
            if isinstance(qs, list):
                for q in qs[:want]:
                    text = clean_question(q)
                    if text:
                        reason = f"Not in the corpus: it asks about {subject}, which {CORPUS_NAME} do not cover."
                        out.append(
                            UnanswerableItem(
                                "out_of_scope", tid, text, reason, len(out), meta={"topic": desc}
                            )
                        )
                        got += 1
            if got < want:
                self.generation_failures["out_of_scope_missing"] += want - got
            prog.step(f"{tid}: {got} question(s)")
        return out

    def generate_perturbations(
        self, sources: Sequence[AnswerableItem], n: int, start_index: int
    ) -> list[UnanswerableItem]:
        self.llm.phase = "perturbation"
        if not sources or n <= 0:
            return []
        rng = random.Random(f"{self.seed}-perturbation")
        order = list(sources)
        rng.shuffle(order)
        kinds = list(PERTURBATION_TARGETS)
        prog = Progress("perturbation", n, self.out)
        out: list[UnanswerableItem] = []
        for i in range(n):
            src = order[i % len(order)]
            kind = kinds[i % len(kinds)]
            values, template = PERTURBATION_TARGETS[kind]
            value = values[(i // len(kinds)) % len(values)]
            target = template.format(x=value)
            data = self._ask(perturbation_messages(src.question, target))
            text = clean_question(data.get("question", "")) if data else ""
            if text:
                reason = (
                    f"Not in the corpus: it asks about {target}, which {CORPUS_NAME} do not cover."
                )
                out.append(
                    UnanswerableItem(
                        "perturbation",
                        kind,
                        text,
                        reason,
                        start_index + len(out),
                        meta={
                            "target": target,
                            "source_question_id": src.qid,
                            "source_question": src.question,
                            "source_nodes": src.node_ids,
                        },
                    )
                )
            else:
                self.generation_failures["perturbation_invalid"] += 1
            prog.step(f"{kind} ({value}) from {src.qid}: {'ok' if text else 'invalid'}")
        return out

    def filter_unanswerable(
        self, items: Sequence[UnanswerableItem], answerable_questions: Sequence[str]
    ) -> list[UnanswerableItem]:
        fmt = self._stat(
            "u:format",
            f"non-empty question of {MIN_QUESTION_WORDS}-{MAX_QUESTION_WORDS} words",
        )
        corpus_num = self._stat(
            "u:corpus_provision_number",
            "the question names no section or rule number that exists in this corpus",
        )
        kept: list[UnanswerableItem] = []
        for it in items:
            w = n_words(it.question)
            reason = (
                "too_short"
                if w < MIN_QUESTION_WORDS
                else ("too_long" if w > MAX_QUESTION_WORDS else "")
            )
            if not fmt.record(not reason, reason, it.brief()):
                continue
            it.filters.append("format")
            hits = sorted(section_rule_numbers(it.question) & self.corpus_numbers)
            if corpus_num.record(
                not hits, "names_corpus_provision", {**it.brief(), "numbers": hits}
            ):
                it.filters.append("corpus_provision_number")
                kept.append(it)

        dup = self._stat(
            "u:near_duplicate",
            f"bge-small cosine <= {DUPLICATE_THRESHOLD} to every existing and selected "
            "answerable question and to earlier unanswerable candidates (keep the first)",
        )
        reference = [q.question for q in self.existing] + list(answerable_questions)
        res = near_duplicates(
            [it.question for it in kept], self._encoder(), DUPLICATE_THRESHOLD, reference
        )
        survivors = []
        for it, ok, sim, near in zip(kept, res.keep, res.max_similarity, res.nearest, strict=True):
            it.meta["max_similarity"] = sim
            if dup.record(ok, "near_duplicate", {**it.brief(), "similarity": sim, "nearest": near}):
                it.filters.append("near_duplicate")
                survivors.append(it)
        return self.retrieval_check(survivors)

    def retrieval_config(self) -> RetrievalConfig:
        data = self.resources.settings.defaults.model_dump()
        data.update(
            chunker="structural",
            retriever="hybrid_rrf",
            dense_model=self.dense_model,
            query_instruction=True,
            xref_expand=False,
            rerank=False,
            final_k=RETRIEVAL_CHECK_K,
        )
        return RetrievalConfig.model_validate(data)

    def retrieval_check(self, items: Sequence[UnanswerableItem]) -> list[UnanswerableItem]:
        self.llm.phase = "retrieval_check"
        cfg = self.retrieval_config()
        stat = self._stat(
            "u:retrieval_check",
            f"top-{RETRIEVAL_CHECK_K} chunks of {cfg.config_id()} shown to the LLM, which must "
            'return {"answered": false}',
        )
        retriever = Retriever(cfg, self.resources)
        prog = Progress("retrieval check", len(items), self.out)
        kept = []
        for it in items:
            res = retriever.search(it.question)
            top = res.results[:RETRIEVAL_CHECK_K]
            passages = [(r.chunk.heading_path, r.chunk.display_text) for r in top]
            data = self._ask(retrieval_check_messages(it.question, passages))
            verdict = _as_bool(data.get("answered")) if data else None
            it.meta["retrieved_chunk_ids"] = [r.chunk_id for r in top]
            it.meta["retrieval_config"] = cfg.config_id()
            it.meta["overlap"] = round(
                lexical_overlap(it.question, "\n".join(r.chunk.display_text for r in top)), 3
            )
            if data is None:
                reason = "invalid_json"
            elif verdict is False:
                reason = ""
            elif verdict is True:
                reason = "answered_by_retrieved_passages"
            else:
                reason = "no_verdict"
            if stat.record(
                not reason, reason, {**it.brief(), "top_chunks": it.meta["retrieved_chunk_ids"]}
            ):
                it.filters.append("retrieval_check")
                kept.append(it)
            prog.step(f"{it.method}:{it.group}: {reason or 'not answered'}")
        return kept

    # ------------------------------------------------------------ orchestration

    def build(
        self,
        n_candidates: int = 200,
        target_answerable: int = 90,
        n_unanswerable_candidates: int = 90,
        target_unanswerable: int = 30,
    ) -> BuildResult:
        t0 = time.perf_counter()
        say = self.out
        say(f"== answerable: sampling {n_candidates} candidate target(s) (seed {self.seed})")
        cands, sampling = self.sample_candidates(n_candidates)
        say(
            f"   sampled {len(cands)}: "
            + ", ".join(f"{t}={n}" for t, n in sampling["sampled_by_type"].items())
        )
        items = self.draft(cands)
        stages = [
            ("provision numbers", self.filter_provision_numbers),
            ("copy check", self.filter_copy),
            ("number check", self.filter_numbers),
            ("support check", self.filter_support),
            ("answerability from source", self.filter_answerability),
            ("near duplicates", self.filter_duplicates_answerable),
        ]
        say(f"   format: {len(items)}/{len(cands)} pass")
        for label, fn in stages:
            before = len(items)
            say(f"== answerable: {label} ({before} candidate(s))")
            items = fn(items)
            say(f"   {label}: {len(items)}/{before} pass")
        survivors = items
        selected, sel_report = select_answerable(survivors, target_answerable)
        if sel_report["shortfall"]:
            say(
                f"WARNING: only {len(selected)} answerable question(s) survived the filters "
                f"(target {target_answerable}); shortfall {sel_report['shortfall']}"
            )
        next_num = next_id_number([q.id for q in self.existing])
        for k, it in enumerate(selected):
            it.qid = f"q{next_num + k:03d}"

        n_oos = n_unanswerable_candidates // 2
        n_pert = n_unanswerable_candidates - n_oos
        say(f"== unanswerable: generating {n_oos} out-of-scope + {n_pert} perturbation candidates")
        oos = self.generate_out_of_scope(n_oos)
        pert = self.generate_perturbations(selected, n_pert, start_index=len(oos))
        unans_candidates = oos + pert
        say(f"== unanswerable: filters ({len(unans_candidates)} candidate(s))")
        unans = self.filter_unanswerable(unans_candidates, [it.question for it in selected])
        chosen_u, u_report = select_unanswerable(unans, target_unanswerable)
        if u_report["shortfall"]:
            say(
                f"WARNING: only {len(chosen_u)} unanswerable question(s) survived "
                f"(target {target_unanswerable}); shortfall {u_report['shortfall']}"
            )

        questions = [self._to_question(it) for it in selected]
        base = next_num + len(selected)
        questions += [self._to_question_u(it, f"q{base + k:03d}") for k, it in enumerate(chosen_u)]
        overlaps_sel = [it.overlap for it in selected]
        report: dict[str, Any] = {
            "model": self.model_name,
            "seed": self.seed,
            "prompt_version": PROMPT_VERSION,
            "draft_prompt_version": drafting.PROMPT_VERSION,
            "thresholds": {
                "question_words": [MIN_QUESTION_WORDS, MAX_QUESTION_WORDS],
                "max_answer_words": MAX_ANSWER_WORDS,
                "copy_overlap": COPY_THRESHOLD,
                "near_duplicate_cosine": DUPLICATE_THRESHOLD,
                "dense_model": self.resources.settings.dense_model_name(self.dense_model),
                "retrieval_check_k": RETRIEVAL_CHECK_K,
            },
            "answerable": {
                "candidates": sampling,
                "drafted": len(cands),
                "filters": [s.to_dict() for k, s in self.filters.items() if k.startswith("a:")],
                "paraphrase": {
                    "attempted": self.paraphrase_attempted,
                    "passed_copy_check": self.paraphrase_rescued,
                    "in_survivors": sum(1 for it in survivors if it.original_question),
                },
                "independent_answer_refusals": self.answer_refusals,
                "survivors": len(survivors),
                "survivors_by_type": dict(Counter(it.qtype for it in survivors)),
                "survivors_by_document": dict(
                    sorted(Counter(it.doc_id for it in survivors).items())
                ),
                "selection": sel_report,
                "mean_lexical_overlap": {
                    "selected": round(statistics.fmean(overlaps_sel), 4) if overlaps_sel else None,
                    "survivors": round(statistics.fmean(it.overlap for it in survivors), 4)
                    if survivors
                    else None,
                },
            },
            "unanswerable": {
                "candidates_generated": {"out_of_scope": len(oos), "perturbation": len(pert)},
                "candidates_requested": {"out_of_scope": n_oos, "perturbation": n_pert},
                "generation_failures": dict(self.generation_failures),
                "filters": [s.to_dict() for k, s in self.filters.items() if k.startswith("u:")],
                "survivors": len(unans),
                "survivors_by_method": dict(Counter(it.method for it in unans)),
                "selection": u_report,
                "mean_lexical_overlap_with_top5": round(
                    statistics.fmean(it.meta.get("overlap", 0.0) for it in chosen_u), 4
                )
                if chosen_u
                else None,
            },
            "llm_calls": self.llm.summary(),
            "generation_wall_time_s": round(time.perf_counter() - t0, 1),
        }
        return BuildResult(questions, report)

    def _to_question(self, it: AnswerableItem) -> EvalQuestion:
        assert it.qid is not None
        meta: dict[str, Any] = {
            "method": "draft_from_source",
            "filters_passed": list(it.filters),
            "source_nodes": it.node_ids,
            "overlap": round(it.overlap, 3),
            "paraphrased": it.original_question is not None,
            "model": self.model_name,
            "seed": self.seed,
            "prompt_version": PROMPT_VERSION,
            "draft_id": f"d{it.number:04d}",
        }
        if it.original_question is not None:
            meta["original_question"] = it.original_question
            meta["overlap_before_paraphrase"] = round(it.overlap_initial, 3)
        meta.update(it.meta)
        difficulty: Difficulty = "medium"
        if it.difficulty in ("easy", "hard"):
            difficulty = "easy" if it.difficulty == "easy" else "hard"
        return EvalQuestion(
            id=it.qid,
            question=it.question,
            type=it.qtype,  # type: ignore[arg-type]
            answerable=True,
            gold_node_ids=it.node_ids,
            gold_answer=it.gold_answer,
            difficulty=difficulty,
            author=f"auto ({self.model_name}) - filtered",
            notes="automatically generated and filtered; not individually reviewed by a person",
            origin="llm_auto_filtered",
            gen_meta=meta,
        )

    def _to_question_u(self, it: UnanswerableItem, qid: str) -> EvalQuestion:
        meta: dict[str, Any] = {
            "method": it.method,
            "filters_passed": list(it.filters),
            "source_nodes": list(it.meta.get("source_nodes", [])),
            "overlap": it.meta.get("overlap"),
            "model": self.model_name,
            "seed": self.seed,
            "prompt_version": PROMPT_VERSION,
            "group": it.group,
        }
        meta.update({k: v for k, v in it.meta.items() if k not in meta})
        return EvalQuestion(
            id=qid,
            question=it.question,
            type="unanswerable",
            answerable=False,
            gold_node_ids=[],
            gold_answer=it.reason,
            difficulty="hard" if it.method == "perturbation" else "medium",
            author=f"auto ({self.model_name}) - filtered",
            notes="automatically generated and filtered; not individually reviewed by a person",
            origin="llm_auto_filtered",
            gen_meta=meta,
        )
