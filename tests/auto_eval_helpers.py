"""A scriptable fake LLM for the automatic evaluation-set builder (no network).

``Responder`` answers every prompt type of :mod:`ragsys.eval.autogen` (and the drafting prompt)
with valid JSON by default: drafts get a question made of unrelated words (so it is neither
copy-like nor a near duplicate) and an answer copied from the start of the provision; judges
accept; the independent answer repeats the provision start; the retrieval check says "not
answered". Dictionaries override single cases.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ragsys.eval import autogen
from ragsys.llm.client import Message

VOCAB = [
    "river",
    "piano",
    "orange",
    "window",
    "pencil",
    "silver",
    "ticket",
    "bottle",
    "candle",
    "rocket",
    "forest",
    "mirror",
    "basket",
    "violin",
    "tunnel",
    "pillow",
    "marble",
    "castle",
    "jacket",
    "ladder",
    "saddle",
    "helmet",
    "anchor",
    "parrot",
    "lantern",
    "wallet",
    "kettle",
    "meadow",
    "harbor",
    "glacier",
    "comet",
    "tulip",
    "walnut",
    "pepper",
    "canyon",
    "falcon",
    "cactus",
]


def nonsense(seed: str, n: int = 8) -> str:
    rng = random.Random(seed)
    return "Could " + " ".join(rng.sample(VOCAB, n)) + " matter today?"


def provision_start(prompt: str, n: int = 8) -> str:
    m = re.search(r'"""\n(.*?)\n"""', prompt, re.DOTALL)
    words = (m.group(1) if m else "").split()
    return " ".join(words[:n]) or "unknown"


@dataclass
class Responder:
    drafts: dict[str, Any] = field(default_factory=dict)  # first node id -> dict or raw text
    paraphrases: dict[str, str] = field(default_factory=dict)  # question -> new question
    support: dict[str, Any] = field(default_factory=dict)  # first node id -> dict or raw text
    answers: dict[str, str] = field(default_factory=dict)  # question -> independent answer
    answered: dict[str, bool] = field(default_factory=dict)  # question -> retrieval verdict
    out_of_scope: dict[str, list[str]] = field(default_factory=dict)  # topic text -> questions
    perturb: dict[str, str] = field(default_factory=dict)  # source question -> new question
    seen: list[str] = field(default_factory=list)  # system prompt of every call

    def __call__(self, messages: Sequence[Message]) -> str:
        system, user = messages[0]["content"], messages[-1]["content"]
        self.seen.append(system)
        node_m = re.search(r"PROVISION \[([^\]]+)\]", user)
        node = node_m.group(1) if node_m else ""
        quoted = re.search(r'^"(.*)"$', user, re.MULTILINE)
        question = re.search(r"^QUESTION: (.*)$", user, re.MULTILINE)
        if system.startswith("You write evaluation questions"):
            d = self.drafts.get(node)
            if isinstance(d, str):
                return d
            qtype = re.search(r'"type": "([a-z_]+)"', user)
            default = {
                "question": nonsense(node),
                "gold_answer": provision_start(user),
                "type": qtype.group(1) if qtype else "factual",
                "difficulty": "medium",
            }
            return json.dumps({**default, **(d or {})})
        if system == autogen.SYSTEM_PARAPHRASE:
            q = quoted.group(1) if quoted else ""
            return json.dumps({"question": self.paraphrases.get(q, nonsense("p" + q))})
        if system == autogen.SYSTEM_SUPPORT:
            s = self.support.get(node, {"supported": True, "corrected_answer": ""})
            return s if isinstance(s, str) else json.dumps(s)
        if system == autogen.SYSTEM_ANSWER:
            q = question.group(1) if question else ""
            return json.dumps({"answer": self.answers.get(q, provision_start(user))})
        if system == autogen.SYSTEM_OUT_OF_SCOPE:
            topic = user.splitlines()[1]
            qs = self.out_of_scope.get(topic, [nonsense(topic + str(i), 9) for i in range(3)])
            return json.dumps({"questions": qs})
        if system == autogen.SYSTEM_PERTURB:
            q = quoted.group(1) if quoted else ""
            target = re.search(r"^Rewrite it into a new question about (.*)\.$", user, re.MULTILINE)
            default = nonsense("pert" + q + (target.group(1) if target else ""), 10)
            return json.dumps({"question": self.perturb.get(q, default)})
        if system == autogen.SYSTEM_RETRIEVAL:
            q = question.group(1) if question else ""
            return json.dumps({"answered": self.answered.get(q, False)})
        raise AssertionError(f"unexpected prompt: {system[:60]}")
