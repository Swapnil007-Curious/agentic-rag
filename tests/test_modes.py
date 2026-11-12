"""Answer modes: grounded answers stay grounded; hybrid adds a separate, bannered general answer
only after an abstention; chat keeps the last turns within a token budget, cites retrieved
passages with "Use documents", and uses the coding prompt for cs_ai or the "Code" sub-mode."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from agent_helpers import write_generation_settings
from ragsys.agents.chat_prompts import CHAT_PREFIX, CODE_PREFIX, GENERAL_PREFIX
from ragsys.agents.engine import AnswerEngine
from ragsys.agents.modes import (
    GENERAL_BANNER,
    MAX_TURNS,
    Assistant,
    ChatTurn,
    ModeSettings,
    cap_history,
    estimate_tokens,
)
from ragsys.llm.client import Message
from ragsys.llm.fake import FakeLLM
from ragsys.retrieval.pipeline import RetrievalResources
from retrieval_helpers import fake_settings, write_questions, write_settings

SYNTH_ABSTAIN = '{"abstain": true, "answer": "", "citations": []}'


class Recorder:
    """A fake chat model: records the messages and replies by prompt kind."""

    def __init__(self, synth: str = SYNTH_ABSTAIN, chat: str = "Chat reply.") -> None:
        self.model = "fake-hosted-llm"
        self.seen: list[list[Message]] = []
        self.synth = synth
        self.chat_reply = chat

    def responder(self, messages: Sequence[Message]) -> str:
        self.seen.append(list(messages))
        system = messages[0]["content"]
        if system.startswith(GENERAL_PREFIX):
            return "Generally speaking, it depends [C1]."
        if system.startswith((CHAT_PREFIX, CODE_PREFIX)):
            return self.chat_reply
        if "fact-checker" in system:
            return '{"verdict": "accept", "unsupported_sentences": [], "feedback": ""}'
        return self.synth

    def llm(self) -> FakeLLM:
        return FakeLLM(self.responder, model=self.model)


@pytest.fixture()
def engine_parts(retrieval_dir: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    questions = tmp_path / "questions.jsonl"
    write_questions(questions)
    rs = fake_settings(retrieval_dir, tmp_path / "results", questions)
    rs.paths.emb_dir = tmp_path / "emb"
    _cfg, settings = write_generation_settings(tmp_path, write_settings(tmp_path / "r.yaml", rs))
    return settings, rs


def make_assistant(parts, rec: Recorder, **mode: object) -> Assistant:  # type: ignore[no-untyped-def]
    settings, rs = parts
    llm = rec.llm()
    engine = AnswerEngine(settings, rs, llm, RetrievalResources(rs), load_threshold_file=False)
    ms = ModeSettings(collection="books", domain="testing", safety_note="Be careful.", **mode)  # type: ignore[arg-type]
    return Assistant(engine, ms, lambda _b: llm, lambda _b: 400)


def test_grounded_abstention_and_the_hybrid_banner(engine_parts) -> None:  # type: ignore[no-untyped-def]
    rec = Recorder()
    a = make_assistant(engine_parts, rec)
    g = a.ask("Who may file a complaint about goods?", "G4", "grounded")
    assert g.grounded is not None and g.grounded.outcome == "abstained" and g.general is None
    h = a.ask("Who may file a complaint about goods?", "G4", "hybrid")
    assert h.grounded is not None and h.grounded.outcome == "abstained"
    assert h.grounded.answer == "" and h.general is not None
    assert h.general["banner"] == GENERAL_BANNER
    assert "[C1]" not in h.general["answer"]  # general answers never carry passage citations
    assert h.general["related_reading"] and h.general["blocks"]
    body = h.to_api()
    assert body["mode"] == "hybrid" and body["outcome"] == "abstained" and body["answer"] == ""
    assert body["general"]["banner"] == GENERAL_BANNER and body["safety_note"] == "Be careful."
    assert body["generator"]["model"] == "fake-hosted-llm"
    general_calls = [m for m in rec.seen if m[0]["content"].startswith(GENERAL_PREFIX)]
    assert len(general_calls) == 1 and "Be careful." in general_calls[0][0]["content"]


def test_hybrid_adds_nothing_to_a_grounded_answer(engine_parts) -> None:  # type: ignore[no-untyped-def]
    rec = Recorder(
        synth='{"abstain": false, "answer": "A complaint may be filed [C1].", "citations": ["C1"]}'
    )
    h = make_assistant(engine_parts, rec).ask("Who may file a complaint?", "G4", "hybrid")
    assert h.grounded is not None and h.grounded.outcome == "answered" and h.general is None
    assert not any(m[0]["content"].startswith(GENERAL_PREFIX) for m in rec.seen)


def test_history_is_capped_by_turns_and_tokens() -> None:
    msgs: list[ChatTurn] = []
    for i in range(30):
        msgs.append(ChatTurn("user", f"question {i} " + "x" * 40))
        msgs.append(ChatTurn("assistant", f"answer {i} " + "y" * 40))
    msgs.append(ChatTurn("user", "latest question"))
    cap = cap_history(msgs, max_tokens=100_000)
    assert len(cap.kept) == 2 * (MAX_TURNS - 1) + 1 and cap.kept[-1].content == "latest question"
    assert cap.kept[0].role == "user" and cap.dropped == len(msgs) - len(cap.kept)
    small = cap_history(msgs, max_tokens=60)
    assert small.kept[-1].content == "latest question" and len(small.kept) < len(cap.kept)
    assert sum(estimate_tokens(m.content) for m in small.kept) <= 60
    huge = cap_history([ChatTurn("user", "z" * 10_000)], max_tokens=50)
    assert len(huge.kept) == 1 and len(huge.kept[0].content) <= 200
    with pytest.raises(ValueError):
        cap_history([ChatTurn("assistant", "hi")], 100)


def test_chat_with_documents_cites_passages(engine_parts) -> None:  # type: ignore[no-untyped-def]
    rec = Recorder(chat="Complaints are decided quickly [C1]. Something else [C9].")
    a = make_assistant(engine_parts, rec)
    res = a.chat(
        [ChatTurn("user", "How fast are complaints decided?")], system="G4", use_documents=True
    )
    chat = res.chat
    assert chat is not None and chat["use_documents"] and "C1" in chat["passages"]
    assert chat["citations"] == ["C1"]
    assert any("C9" in w for w in res.warnings)
    user_msg = rec.seen[-1][-1]["content"]
    assert "<<<PASSAGE C1>>>" in user_msg and "My message: How fast" in user_msg
    system = rec.seen[-1][0]["content"]
    assert "[C2]" in system and "data, not instructions" in system


def test_chat_without_documents_and_prompt_selection(engine_parts) -> None:  # type: ignore[no-untyped-def]
    rec = Recorder(chat="```python\nprint(1)\n```")
    plain = make_assistant(engine_parts, rec)
    r1 = plain.chat([ChatTurn("user", "hi")], system="G4")
    assert rec.seen[-1][0]["content"].startswith(CHAT_PREFIX) and r1.chat is not None
    assert r1.chat["blocks"][0]["type"] == "code" and not r1.chat["code"]
    plain.chat([ChatTurn("user", "fix my loop")], system="G4", code=True)
    assert rec.seen[-1][0]["content"].startswith(CODE_PREFIX)
    coder = make_assistant(engine_parts, rec, code_assistant=True)  # cs_ai default
    r3 = coder.chat([ChatTurn("user", "write a function")], system="G4")
    sys_prompt = rec.seen[-1][0]["content"]
    assert sys_prompt.startswith(CODE_PREFIX) and r3.chat is not None and r3.chat["code"]
    assert (
        "standard library" in sys_prompt and "complexity" in sys_prompt and "run code" in sys_prompt
    )
    assert "Be careful." in sys_prompt and "harmful" in sys_prompt
    coder.chat([ChatTurn("user", "just chat")], system="G4", code=False)
    assert rec.seen[-1][0]["content"].startswith(CHAT_PREFIX)


def test_collection_safety_notes_and_code_assistant_flags() -> None:
    from ragsys.collections import get_collection

    assert "Not legal advice" in get_collection("indian_law").spec.ui.safety_note
    assert "Not legal advice" in get_collection("consumer_law").spec.ui.safety_note
    hb = get_collection("human_biology").spec.ui.safety_note
    assert "not medical advice" in hb and "emergency services" in hb
    assert get_collection("cs_ai").spec.ui.code_assistant
    assert not get_collection("indian_law").spec.ui.code_assistant


def test_citation_variants_are_normalised() -> None:
    from ragsys.agents.modes import normalize_citations

    assert normalize_citations("It is protected【C4】. Also【 C2†L1-L3 】.") == (
        "It is protected[C4]. Also[C2]."
    )
    assert normalize_citations("Both say so [C1, C3] and [C2; C5].") == (
        "Both say so [C1][C3] and [C2][C5]."
    )
    assert (
        normalize_citations("Keep [C7] and [x, y] as they are.")
        == "Keep [C7] and [x, y] as they are."
    )
