"""Automatic evaluation-set builder: every filter on crafted pass/fail cases (FakeLLM, fake
encoder), the unanswerable retrieval-check path, the script (--rebuild never removes human
items) and resuming from the LLM cache."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx2
import pytest

from auto_eval_helpers import Responder, nonsense
from ragsys.eval import autogen
from ragsys.eval.autogen import AnswerableItem, AutoEvalBuilder, UnanswerableItem
from ragsys.eval.drafting import Candidate
from ragsys.eval.schema import EvalQuestion
from ragsys.eval.validate import load_questions
from ragsys.ingest.pipeline import load_nodes, load_xrefs
from ragsys.llm.client import LLMClient, LLMConfig
from ragsys.llm.fake import FakeLLM
from ragsys.retrieval.pipeline import RetrievalResources
from ragsys.structure.base import Node
from retrieval_helpers import fake_settings, write_settings

HUMAN = {
    "id": "q001",
    "question": "What does the word goods mean under the Act?",
    "type": "definition",
    "answerable": True,
    "gold_node_ids": ["testact:s2(2)"],
    "gold_answer": "Every kind of movable property.",
    "difficulty": "easy",
    "author": "Swapnil",
    "notes": "hand written — keep",
}
REVIEWED = {
    "id": "q002",
    "question": "Within how many days may an aggrieved person appeal?",
    "type": "numeric",
    "answerable": True,
    "gold_node_ids": ["testact:s4(1)"],
    "gold_answer": "Thirty days.",
    "difficulty": "easy",
    "author": "Swapnil",
    "origin": "llm_draft_human_reviewed",
}


@pytest.fixture()
def env(retrieval_dir: Path, tmp_path: Path) -> dict[str, Any]:
    questions = tmp_path / "questions.jsonl"
    s = fake_settings(retrieval_dir, tmp_path / "results", questions)
    s.dense_models["bge-small"] = "fake:hash-64"
    s.paths.emb_dir = tmp_path / "emb"
    nodes = load_nodes(retrieval_dir / "structure" / "nodes.jsonl")
    return {
        "settings": s,
        "cfg": write_settings(tmp_path / "retrieval.yaml", s),
        "questions": questions,
        "tmp": tmp_path,
        "nodes": nodes,
        "by_id": {n.node_id: n for n in nodes},
        "xrefs": load_xrefs(retrieval_dir / "structure" / "xrefs.jsonl"),
    }


def make_builder(
    env: dict[str, Any], responder: Responder, existing: list[EvalQuestion] | None = None
) -> AutoEvalBuilder:
    res = RetrievalResources(env["settings"])
    lines: list[str] = []
    b = AutoEvalBuilder(
        FakeLLM(responder),
        res,
        env["nodes"],
        env["xrefs"],
        {},
        existing or [],
        out=lines.append,
    )
    return b


def item(
    env: dict[str, Any], qtype: str, nodes: list[str], q: str, a: str, n: int = 1
) -> AnswerableItem:
    by_id: dict[str, Node] = env["by_id"]
    return AnswerableItem(
        Candidate(qtype, tuple(by_id[x] for x in nodes)), n, question=q, gold_answer=a
    )


def reasons(b: AutoEvalBuilder, key: str) -> dict[str, int]:
    return dict(b.filters[key].reasons)


def test_draft_and_format_filter(env: dict[str, Any]) -> None:
    by_id = env["by_id"]
    r = Responder(
        drafts={
            "testact:s1": "this is not json",
            "testact:s2(1)": {"question": "Who buys?"},
            "testact:s2(2)": {"gold_answer": "word " * 41},
            "testact:s2(3)": {"question": ""},
        }
    )
    b = make_builder(env, r)
    cands = [
        Candidate("factual", (by_id[x],))
        for x in ("testact:s1", "testact:s2(1)", "testact:s2(2)", "testact:s2(3)", "testact:s2(4)")
    ]
    kept = b.draft(cands)
    assert [it.node_ids for it in kept] == [["testact:s2(4)"]]
    assert kept[0].filters == ["format"] and kept[0].number == 5
    assert reasons(b, "a:format") == {
        "invalid_json": 1,
        "question_too_short": 1,
        "answer_too_long": 1,
        "missing_fields": 1,
    }


def test_provision_number_filter(env: dict[str, Any]) -> None:
    b = make_builder(env, Responder())
    items = [
        item(
            env,
            "factual",
            ["testact:s5"],
            "What does section 5 let the authority do?",
            "Issue guidelines.",
        ),
        item(
            env,
            "cross_reference",
            ["testact:s4(2)", "testact:s3(1)"],
            "Which manner from section 3 applies to appeals?",
            "Writing or electronic.",
        ),
        item(
            env,
            "factual",
            ["testact:s5"],
            "What may the authority issue for complaints?",
            "Guidelines.",
        ),
    ]
    kept = b.filter_provision_numbers(items)
    assert [it.qtype for it in kept] == ["cross_reference", "factual"]
    assert reasons(b, "a:provision_number") == {"mentions_provision_number": 1}


def test_copy_check_and_paraphrase_path(env: dict[str, Any]) -> None:
    copy_q = "What does goods mean, every kind of movable property?"
    stubborn = 'What does "Government" mean, the Government of Testland?'
    numbered = "Does consumer mean any person who buys goods for consideration?"
    r = Responder(
        paraphrases={
            copy_q: "In daily life which belongings count as things that people purchase?",
            stubborn: "What does Government mean, the Government of Testland?",
            numbered: "What does section 2 say about shoppers in ordinary words?",
        }
    )
    b = make_builder(env, r)
    items = [
        item(env, "definition", ["testact:s2(2)"], copy_q, "Movable property."),
        item(env, "definition", ["testact:s2(4)"], stubborn, "Government of Testland."),
        item(env, "definition", ["testact:s2(1)"], numbered, "A buyer of goods."),
        item(env, "definition", ["testact:s2(3)"], nonsense("x"), "A seller."),
    ]
    kept = b.filter_copy(items)
    assert [it.node_ids[0] for it in kept] == ["testact:s2(2)", "testact:s2(3)"]
    rescued = kept[0]
    assert rescued.original_question == copy_q and rescued.overlap <= 0.6 < rescued.overlap_initial
    assert rescued.question.startswith("In daily life")
    assert kept[1].original_question is None  # low overlap: no LLM call
    assert r.seen.count(autogen.SYSTEM_PARAPHRASE) == 3
    assert reasons(b, "a:copy") == {
        "copy_like_after_paraphrase": 1,
        "paraphrase_mentions_provision_number": 1,
    }
    assert b.paraphrase_attempted == 3 and b.paraphrase_rescued == 1


def test_number_filter_digits_and_words(env: dict[str, Any]) -> None:
    b = make_builder(env, Responder())
    items = [
        item(env, "numeric", ["testact:s4(1)"], "q", "Within 30 days."),
        item(env, "numeric", ["testact:s3(1)-prov1"], "q", "No fee below Rs. 1 lakh."),
        item(env, "numeric", ["testact:s4(1)"], "q", "Within forty-five days."),
        item(env, "factual", ["testact:s1(1)"], "q", "The Test Protection Act, 2024."),
        item(
            env,
            "cross_reference",
            ["testact:s4(2)", "testact:s3(1)"],
            "q",
            "Sub-section (1) of section 3 applies.",
        ),
    ]
    kept = b.filter_numbers(items)
    assert [it.gold_answer for it in kept] == [
        "Within 30 days.",
        "No fee below Rs. 1 lakh.",
        "The Test Protection Act, 2024.",
        "Sub-section (1) of section 3 applies.",
    ]
    assert reasons(b, "a:numbers") == {"number_not_in_source": 1}
    assert b.filters["a:numbers"].examples[0]["missing"] == ["forty five"]


def test_support_filter_never_applies_corrections(env: dict[str, Any]) -> None:
    r = Responder(
        support={
            "testact:s2(3)": {"supported": False, "corrected_answer": "A seller of goods."},
            "testact:s2(4)": "garbage",
            "testact:s2(1)": {"supported": "true", "corrected_answer": ""},
        }
    )
    b = make_builder(env, r)
    items = [
        item(env, "definition", ["testact:s2(3)"], "Who is a trader?", "A buyer."),
        item(env, "definition", ["testact:s2(4)"], "What is Government?", "Testland's government."),
        item(env, "definition", ["testact:s2(1)"], "Who is a consumer?", "A buyer of goods."),
    ]
    kept = b.filter_support(items)
    assert [it.node_ids[0] for it in kept] == ["testact:s2(1)"]
    assert items[0].gold_answer == "A buyer."  # correction recorded, not applied
    assert items[0].meta["support_check"] == {
        "supported": False,
        "corrected_answer": "A seller of goods.",
    }
    assert reasons(b, "a:support") == {"not_supported": 1, "invalid_json": 1}


def test_answerability_contradiction_detection(env: dict[str, Any]) -> None:
    r = Responder(
        answers={
            "Q1": "An appeal lies within forty-five days.",
            "Q2": "The District Commission hears it.",
            "Q3": "Within thirty days of the order.",
            "Q4": "This is not stated in the text.",
        }
    )
    b = make_builder(env, r)
    items = [
        item(env, "numeric", ["testact:s4(1)"], "Q1", "Within 30 days."),
        item(env, "factual", ["testact:s4(1)"], "Q2", "The State Commission hears it."),
        item(env, "numeric", ["testact:s4(1)"], "Q3", "Within 30 days."),
        item(env, "factual", ["testact:s4(1)"], "Q4", "Any aggrieved person."),
    ]
    kept = b.filter_answerability(items)
    assert [it.question for it in kept] == ["Q3", "Q4"]
    assert reasons(b, "a:answerable_from_source") == {"number_conflict": 1, "entity_conflict": 1}
    ex = {e["question"]: e for e in b.filters["a:answerable_from_source"].examples}
    assert ex["Q1"]["numbers"] == {"gold_only": ["30"], "llm_only": ["forty five"]}
    assert ex["Q2"]["entities"] == [["state commission", "district commission"]] or ex["Q2"][
        "entities"
    ] == [("state commission", "district commission")]
    assert b.answer_refusals == 1 and kept[1].meta["independent_answer_refused"] is True


def test_near_duplicates_against_existing_and_earlier(env: dict[str, Any]) -> None:
    existing = [EvalQuestion.model_validate(HUMAN)]
    b = make_builder(env, Responder(), existing)
    q = nonsense("dup")
    items = [
        item(env, "factual", ["testact:s5"], q, "a", 1),
        item(env, "factual", ["testact:s1(2)"], q, "b", 2),
        item(env, "definition", ["testact:s2(4)"], HUMAN["question"], "c", 3),  # type: ignore[arg-type]
        item(env, "factual", ["testact:s3(2)"], nonsense("other"), "d", 4),
    ]
    kept = b.filter_duplicates_answerable(items)
    assert [it.number for it in kept] == [1, 4]
    assert reasons(b, "a:near_duplicate") == {"near_duplicate": 2}
    assert items[1].meta["max_similarity"] == pytest.approx(1.0)


def test_unanswerable_generation_and_retrieval_check(env: dict[str, Any]) -> None:
    answered_q = "Can a consumer file a complaint in writing or electronically under this Act?"
    r = Responder(
        out_of_scope={
            autogen.OUT_OF_SCOPE_TOPICS[0][1]: [
                answered_q,
                "1. What does section 3 of the RTI Act say about fees?",
                "Too short?",
            ],
        },
        answered={answered_q: True},
    )
    b = make_builder(env, r)
    sources = [item(env, "factual", ["testact:s5"], nonsense(f"src{i}"), "a", i) for i in range(3)]
    for i, s in enumerate(sources):
        s.qid = f"q{10 + i:03d}"
    oos = b.generate_out_of_scope(6)
    pert = b.generate_perturbations(sources, 4, start_index=len(oos))
    assert [u.method for u in oos] == ["out_of_scope"] * 6 and len(pert) == 4
    assert oos[1].question.startswith("What does section 3")  # list marker stripped
    assert {u.group for u in pert} == set(autogen.PERTURBATION_TARGETS)
    assert pert[0].meta["source_question_id"] == "q010" and "Not in the corpus" in pert[0].reason
    dup = UnanswerableItem("perturbation", "another_law", sources[0].question, "r", 99)
    kept = b.filter_unanswerable([*oos, *pert, dup], [s.question for s in sources])
    assert answered_q not in {u.question for u in kept}
    assert reasons(b, "u:format") == {"too_short": 1}
    assert reasons(b, "u:corpus_provision_number") == {"names_corpus_provision": 1}
    assert reasons(b, "u:near_duplicate") == {"near_duplicate": 1}
    assert reasons(b, "u:retrieval_check") == {"answered_by_retrieved_passages": 1}
    assert len(kept) == 7
    for u in kept:
        assert len(u.meta["retrieved_chunk_ids"]) == 5 and u.meta["retrieval_config"].startswith(
            "structural|hybrid_rrf"
        )
        assert u.filters == [
            "format",
            "corpus_provision_number",
            "near_duplicate",
            "retrieval_check",
        ]
    prompts = [s for s in r.seen if s == autogen.SYSTEM_RETRIEVAL]
    assert len(prompts) == 8  # every candidate that reached the retrieval check


FULL = ("40", "12", "12", "6")  # candidates, answerable target, unanswerable candidates, target
SMALL = ("10", "4", "6", "2")


def _run(
    script: ModuleType, env: dict[str, Any], llm: Any, *extra: str, sizes: tuple[str, ...] = FULL
) -> int:
    args = [
        "--config", str(env["cfg"]),
        "--questions", str(env["questions"]),
        "--ingest-config", str(env["tmp"] / "missing.yaml"),
        "--n-candidates", sizes[0],
        "--target-answerable", sizes[1],
        "--n-unanswerable-candidates", sizes[2],
        "--target-unanswerable", sizes[3],
        *extra,
    ]  # fmt: skip
    return int(script.main(args, llm=llm))


def test_build_script_appends_and_rebuild_never_removes_human_items(
    env: dict[str, Any], script: Callable[[str], ModuleType], capsys: pytest.CaptureFixture[str]
) -> None:
    build = script("build_auto_eval")
    human_lines = json.dumps(HUMAN, ensure_ascii=False) + "\n" + json.dumps(REVIEWED) + "\n"
    env["questions"].write_text(human_lines, encoding="utf-8")
    assert _run(build, env, FakeLLM(Responder())) == 0
    out = capsys.readouterr().out
    assert "released encoder(s)" in out and "appended" in out
    qs, errors = load_questions(env["questions"])
    assert errors == []
    auto = [q for q in qs if q.origin == "llm_auto_filtered"]
    assert [q.id for q in qs[:2]] == ["q001", "q002"] and auto[0].id == "q003"
    assert len(auto) == 18 and sum(1 for q in auto if not q.answerable) == 6
    assert all(q.author == "auto (fake-llm) - filtered" and q.gen_meta for q in auto)
    a = next(q for q in auto if q.answerable)
    assert a.gen_meta is not None and {
        "method",
        "filters_passed",
        "source_nodes",
        "overlap",
        "model",
        "seed",
    } <= set(a.gen_meta)
    assert (
        a.gen_meta["filters_passed"][-1] == "near_duplicate"
        and a.gen_meta["source_nodes"] == a.gold_node_ids
    )
    assert "testact:s2(2)" not in {g for q in auto for g in q.gold_node_ids}  # used by a human item
    report = json.loads(
        (env["tmp"] / "results" / "tables" / "eval_construction.json").read_text(encoding="utf-8")
    )
    names = [f["name"] for f in report["answerable"]["filters"]]
    assert names == [
        "format",
        "provision_number",
        "copy",
        "numbers",
        "support",
        "answerable_from_source",
        "near_duplicate",
    ]
    assert [f["name"] for f in report["unanswerable"]["filters"]] == [
        "format",
        "corpus_provision_number",
        "near_duplicate",
        "retrieval_check",
    ]
    assert report["answerable"]["candidates"]["sampled"] > 12 and report["model"] == "fake-llm"
    fc = report["final_counts"]["new_items"]
    assert fc["answerable"] == 12 and set(fc["by_split"]) == {"dev", "test"}
    assert report["wall_time_s"] >= 0 and report["run_info"]["python"]
    assert report["answerable"]["selection"]["shortfall"] == 0

    first = env["questions"].read_text(encoding="utf-8")
    assert first.startswith(human_lines)
    # a second run refuses to touch the file without --rebuild
    assert _run(build, env, FakeLLM(Responder())) == 3
    assert "already contains 18" in capsys.readouterr().out
    assert env["questions"].read_text(encoding="utf-8") == first
    # --rebuild replaces only the automatic items; human lines stay byte-identical
    with env["questions"].open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({**HUMAN, "id": "q099", "question": "A later human question about goods?"})
            + "\n"
        )
    assert _run(build, env, FakeLLM(Responder()), "--rebuild") == 0
    text = env["questions"].read_text(encoding="utf-8")
    assert text.startswith(human_lines)
    qs2, _ = load_questions(env["questions"])
    assert [q.id for q in qs2 if q.origin != "llm_auto_filtered"] == ["q001", "q002", "q099"]
    auto2 = [q for q in qs2 if q.origin == "llm_auto_filtered"]
    assert auto2[0].id == "q100" and len(auto2) == 18
    assert [q.question for q in auto2] == [q.question for q in auto]  # deterministic rebuild


class Transport:
    """Fake OpenAI-compatible endpoint backed by a Responder; fails after ``fail_after``."""

    def __init__(self, responder: Responder, fail_after: int | None = None) -> None:
        self.responder = responder
        self.fail_after = fail_after
        self.requests = 0

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if self.fail_after is not None and self.requests >= self.fail_after:
            raise httpx2.ConnectError("endpoint went away")
        self.requests += 1
        body = json.loads(request.content)
        content = self.responder(body["messages"])
        return httpx2.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "cached-model",
                "choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


def client(tmp: Path, transport: Transport) -> LLMClient:
    cfg = LLMConfig(
        base_url="http://llm.test/v1",
        model="cached-model",
        cache_dir=tmp / "llm_cache",
        max_retries=0,
    )
    return LLMClient(
        cfg,
        http_client=httpx2.Client(transport=httpx2.MockTransport(transport)),
        sleep=lambda _s: None,
    )


def test_interrupted_run_resumes_from_the_llm_cache(
    env: dict[str, Any], script: Callable[[str], ModuleType], capsys: pytest.CaptureFixture[str]
) -> None:
    build = script("build_auto_eval")
    first = Transport(Responder(), fail_after=12)
    with pytest.raises(Exception, match=r"(?i)connect"):
        _run(build, env, client(env["tmp"], first), sizes=SMALL)
    assert first.requests == 12 and not env["questions"].exists()  # nothing written
    second = Transport(Responder())
    assert _run(build, env, client(env["tmp"], second), sizes=SMALL) == 0
    report = json.loads(
        (env["tmp"] / "results" / "tables" / "eval_construction.json").read_text(encoding="utf-8")
    )
    calls = report["llm_calls"]
    assert calls["calls"] > 12 and calls["cached"] == 12 and second.requests == calls["calls"] - 12
    # a full rebuild is answered entirely from the cache
    third = Transport(Responder())
    assert _run(build, env, client(env["tmp"], third), "--rebuild", sizes=SMALL) == 0
    assert third.requests == 0
    capsys.readouterr()
