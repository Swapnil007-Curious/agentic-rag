"""Recomputes the consumer_law prompt digests and system fingerprints that were recorded in
tests/data/consumer_law_prompt_snapshot.json before collections and domain profiles existed.
LLM responses are cached by prompt and generation records carry a fingerprint of the prompts,
so any change here would silently invalidate the consumer_law caches and results."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ragsys.agents import critic as cm
from ragsys.agents import synthesizer as sm
from ragsys.agents.config import default_generation_settings
from ragsys.eval import autogen, drafting
from ragsys.eval.drafting import Candidate
from ragsys.eval.generation import prompt_digest, system_fingerprint
from ragsys.structure.base import Node


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _node(i: int, doc: str = "cpa2019", dtype: str = "act", level: str = "section") -> Node:
    return Node(
        doc_id=doc,
        doc_title="The Consumer Protection Act, 2019",
        doc_type=dtype,  # type: ignore[arg-type]
        node_id=f"{doc}:s{i}",
        level=level,  # type: ignore[arg-type]
        number=str(i),
        heading=f"Heading {i}",
        text=f"{i}. Heading {i}.—(1) The District Commission shall decide every complaint "
        f"within {i} days. " * 3,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=10,
    )


def consumer_law_snapshot() -> dict[str, str]:
    titles = {"cpa2019": "Consumer Protection Act, 2019", "ecom2020": "E-Commerce Rules, 2020"}
    out: dict[str, str] = {}
    for t in drafting.TYPE_MIX:
        nodes = (
            (_node(35), _node(36, "ecom2020", "rules", "rule"))
            if t in ("cross_reference", "multi_section")
            else (_node(35),)
        )
        c = Candidate(t, nodes)
        out[f"draft:{t}"] = _h(drafting.build_messages(c, titles))
        out[f"support:{t}"] = _h(autogen.support_messages(c, titles, "Q?", "A."))
        out[f"answer:{t}"] = _h(autogen.answer_messages(c, titles, "Q?"))
        out[f"paraphrase:{t}"] = _h(
            autogen.paraphrase_messages("Within how many days must a complaint be decided?", t)
        )
    for tid, desc, _subj in autogen.OUT_OF_SCOPE_TOPICS:
        out[f"oos:{tid}"] = _h(autogen.out_of_scope_messages(desc))
    out["perturb"] = _h(
        autogen.perturbation_messages("Who decides complaints?", "the Motor Vehicles Act, 1988")
    )
    out["retrieval_check"] = _h(
        autogen.retrieval_check_messages(
            "Q?", [("Consumer Protection Act > Section 35", "text " * 400), ("H2", "t2")]
        )
    )
    out["synth_system"] = _h(sm.system_prompt(False))
    out["synth_system_plain"] = _h(sm.system_prompt(True))
    out["critic_system"] = _h(cm.SYSTEM_PROMPT)
    out["prompt_digest"] = prompt_digest()
    s = default_generation_settings()
    llm = s.llm()
    for name in s.systems:
        if s.system(name).backend is not None:
            continue  # systems added later on another backend (G2H, G4H) are not part of it
        out[f"fingerprint:{name}"] = system_fingerprint(s, name, llm)
    return out
