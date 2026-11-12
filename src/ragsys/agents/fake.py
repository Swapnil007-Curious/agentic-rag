"""Deterministic offline stand-in for the generator LLM (tests and ``--fake-llm`` smoke runs).

* synthesizer prompts: answer with the first sentence of passage C1 and cite it, or abstain when
  the question shares fewer than two content words with that passage;
* critic prompts: accept;
* chat, coding and general-knowledge prompts (ragsys.agents.chat_prompts): canned Markdown;
* anything else: the default :class:`ragsys.llm.fake.FakeLLM` reply.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ragsys.agents.chat_prompts import CHAT_PREFIX, CODE_PREFIX, GENERAL_PREFIX
from ragsys.llm.client import Message
from ragsys.llm.fake import FakeLLM
from ragsys.profiles import all_generation_prompts
from ragsys.retrieval.bm25 import STOPWORDS

_WORD = re.compile(r"[a-z]{3,}")
_PASSAGE = re.compile(r"<<<PASSAGE (C\d+)>>>.*?<<<TEXT>>>\n(.*?)\n<<<END \1>>>", re.DOTALL)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOPWORDS}


def chat_responder(system: str, user: str) -> str | None:
    """Canned Markdown replies for chat, coding and general-knowledge prompts (None: other)."""
    if system.startswith(CODE_PREFIX):
        return (
            "Here is a function, assuming plain Python 3 strings.\n\n"
            "```python\n"
            "def is_palindrome(text: str) -> bool:\n"
            '    """True when text reads the same backwards (case and punctuation ignored)."""\n'
            "    chars = [c.lower() for c in text if c.isalnum()]\n"
            "    return chars == chars[::-1]\n"
            "```\n\n"
            "Complexity: O(n) time and O(n) extra space."
        )
    if system.startswith(CHAT_PREFIX):
        passages = _PASSAGE.findall(user)
        if passages:
            first = " ".join(passages[0][1].split())[:160]
            return f"According to the documents, {first} [{passages[0][0]}]."
        return "**Fake chat answer.** This reply comes from the offline stand-in model."
    if system.startswith(GENERAL_PREFIX):
        return "From general knowledge (offline stand-in): this depends on the context."
    return None


def agent_responder(messages: Sequence[Message]) -> str:
    system = messages[0].get("content", "") if messages else ""
    user = messages[-1].get("content", "") if messages else ""
    canned = chat_responder(system, user)
    if canned is not None:
        return canned
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
