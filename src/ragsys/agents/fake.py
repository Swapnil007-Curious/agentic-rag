"""Deterministic offline stand-in for the generator LLM (tests and ``--fake-llm`` smoke runs).

* synthesizer prompts: answer with the first sentence of passage C1 and cite it, or abstain when
  the question shares fewer than two content words with that passage;
* critic prompts: accept;
* anything else: the default :class:`ragsys.llm.fake.FakeLLM` reply.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ragsys.llm.client import Message
from ragsys.llm.fake import FakeLLM
from ragsys.profiles import all_generation_prompts
from ragsys.retrieval.bm25 import STOPWORDS

_WORD = re.compile(r"[a-z]{3,}")
_PASSAGE = re.compile(r"<<<PASSAGE (C\d+)>>>.*?<<<TEXT>>>\n(.*?)\n<<<END \1>>>", re.DOTALL)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOPWORDS}


def agent_responder(messages: Sequence[Message]) -> str:
    system = messages[0].get("content", "") if messages else ""
    user = messages[-1].get("content", "") if messages else ""
    prompts = all_generation_prompts()
    if any(system.startswith(p.critic_system[:40]) for p in prompts):
        return json.dumps({"verdict": "accept", "unsupported_sentences": [], "feedback": ""})
    if any(system.startswith(p.synth_system[:40]) for p in prompts):
        passages = _PASSAGE.findall(user)
        q = re.search(r"^Question: (.*)$", user, re.MULTILINE)
        question = q.group(1) if q else ""
        if not passages or len(_words(question) & _words(passages[0][1])) < 2:
            return json.dumps({"abstain": True, "answer": "", "citations": []})
        label, text = passages[0]
        first = re.split(r"(?<=[.;:])\s", " ".join(text.split()), maxsplit=1)[0]
        first = first.rstrip(".;:—- ")[:300]
        return json.dumps({"abstain": False, "answer": f"{first} [{label}].", "citations": [label]})
    return FakeLLM().chat(list(messages)).content


def agent_fake_llm() -> FakeLLM:
    return FakeLLM(agent_responder, model="fake-agent-llm")
