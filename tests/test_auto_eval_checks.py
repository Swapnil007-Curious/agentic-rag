"""Deterministic checks of the automatic evaluation-set builder and its selection logic."""

from __future__ import annotations

from collections import Counter

import pytest

from ragsys.eval.autogen import (
    AnswerableItem,
    UnanswerableItem,
    clean_question,
    pick_balanced,
    select_answerable,
    select_unanswerable,
    type_quotas,
)
from ragsys.eval.checks import (
    capitalised_terms,
    entity_conflicts,
    extract_numbers,
    mentions_provision_number,
    near_duplicates,
    number_conflict,
    numbers_missing_from_source,
    section_rule_numbers,
)
from ragsys.eval.drafting import TYPE_MIX, Candidate
from ragsys.retrieval.fakes import HashingEncoder
from ragsys.structure.base import Node


def values(text: str) -> list[float]:
    return [m.value for m in extract_numbers(text)]


def test_numbers_in_digits_and_words() -> None:
    assert values("within thirty days") == [30]
    assert values("twenty-one days or one hundred and eighty days") == [21, 180]
    assert values("Rs. 50,00,000, i.e. fifty lakh rupees") == [5_000_000, 5_000_000]
    assert values("one crore and 10 crore") == [10_000_000, 100_000_000]
    assert values("Rs.7500 at 7.5 per cent") == [7500, 7.5]
    assert values("no numbers here") == []


def test_number_check_accepts_any_written_form_and_rejects_new_numbers() -> None:
    src = ["The appeal shall be filed within thirty days and the fee is fifty lakh rupees."]
    assert numbers_missing_from_source("Within 30 days.", src) == []
    assert numbers_missing_from_source("Rs. 50,00,000", src) == []
    assert numbers_missing_from_source("Within 45 days.", src) == ["45"]
    assert numbers_missing_from_source("forty-five days", src) == ["forty five"]
    # "50" alone matches the base of "fifty lakh"; "50 crore" does not
    assert numbers_missing_from_source("50 crore", src) == ["50 crore"]
    # numbers inside provision references may match the source node ids
    assert (
        numbers_missing_from_source("Under section 38(2), 30 days.", src, ["cpa2019:s38(2)"]) == []
    )
    assert numbers_missing_from_source("Under section 38(2), 30 days.", src) == ["38", "2"]


def test_number_and_entity_conflicts() -> None:
    assert number_conflict("within 30 days", "within forty-five days") == (["30"], ["forty five"])
    assert number_conflict("30 days", "thirty days, extendable by fifteen days") == ([], [])
    assert number_conflict("one crore rupees", "Rs. 1 crore") == ([], [])
    assert number_conflict("30 days", "The text does not say.") == ([], [])
    assert capitalised_terms(
        "The District Commission, State Commission or National Commission."
    ) == [
        ("district", "commission"),
        ("state", "commission"),
        ("national", "commission"),
    ]
    assert entity_conflicts("The State Government", "The Central Government may notify") == [
        ("state government", "central government")
    ]
    assert (
        entity_conflicts(
            "The National Commission", "the National Consumer Disputes Redressal Commission"
        )
        == []
    )
    # a partial answer (subset of the entities) is not a contradiction
    assert (
        entity_conflicts("District, State or National Commission", "The District Commission") == []
    )
    assert entity_conflicts("Yes. Consumers can file.", "The consumer may complain.") == []


def test_provision_number_detection() -> None:
    for q in (
        "What does section 2(7) say?",
        "What does Rule 3 provide?",
        "What does clause (b) cover?",
        "Is s. 35 relevant?",
        "What is 35(1) about?",
    ):
        assert mentions_provision_number(q), q
    for q in (
        "Under the E-Commerce Rules 2020, what must sellers show?",
        "Is the fee of Rs.7500 refundable?",
        "Can I appeal within 2 (two) years?",
    ):
        assert not mentions_provision_number(q), q
    assert section_rule_numbers("What do section 35 and rule 4 of the RTI Rules, 2012 say?") == {
        "35",
        "4",
    }


def test_near_duplicates_keep_first_and_use_reference() -> None:
    enc = HashingEncoder(64)
    texts = ["Who can file a complaint?", "Who can file a complaint?", "What is the appeal fee?"]
    res = near_duplicates(texts, enc, 0.9)
    assert res.keep == [True, False, True] and res.nearest[1] == texts[0]
    res = near_duplicates(
        ["What is the appeal fee?"], enc, 0.9, reference=["What is the appeal fee?"]
    )
    assert res.keep == [False]


def test_clean_question() -> None:
    assert clean_question('1. "What is the RTI fee?"') == "What is the RTI fee?"
    assert clean_question({"question": "- Who decides?"}) == "Who decides?"


def _node(nid: str) -> Node:
    doc = nid.split(":")[0]
    return Node(
        doc_id=doc, doc_title=doc, doc_type="act", node_id=nid, level="section",
        text="text", page_start=1, page_end=1, char_start=0, char_end=4,
    )  # fmt: skip


def _item(i: int, qtype: str, doc: str) -> AnswerableItem:
    return AnswerableItem(Candidate(qtype, (_node(f"{doc}:s{i}"),)), i, question=f"q{i}")


def test_type_quotas_redistribute_shortfall() -> None:
    assert type_quotas(dict.fromkeys(TYPE_MIX, 100), 90, TYPE_MIX) == TYPE_MIX
    avail = {**dict.fromkeys(TYPE_MIX, 100), "multi_section": 4}
    q = type_quotas(avail, 90, TYPE_MIX)
    assert q["multi_section"] == 4 and sum(q.values()) == 90
    assert type_quotas(dict.fromkeys(TYPE_MIX, 2), 90, TYPE_MIX) == dict.fromkeys(TYPE_MIX, 2)


def test_pick_balanced_spreads_over_groups() -> None:
    groups = {"a": [1, 2, 3, 4, 5], "b": [6], "c": [7, 8]}
    assert pick_balanced(groups, 5) == [1, 6, 7, 2, 8]
    counts: Counter[str] = Counter({"a": 3})
    assert pick_balanced({"a": [1, 2], "b": [3, 4]}, 2, counts) == [3, 1]


def test_select_answerable_is_stratified_and_reports_shortfall() -> None:
    docs = ["cpa2019", "ecom2020", "mediation2020"]
    items = []
    n = 0
    for t in TYPE_MIX:
        for k in range(12):
            n += 1
            items.append(_item(n, t, docs[0] if k < 8 else docs[k % 3]))
    chosen, rep = select_answerable(items, 30)
    assert len(chosen) == 30 and rep["shortfall"] == 0
    by_type = Counter(it.qtype for it in chosen)
    assert by_type == Counter(
        {
            "definition": 5,
            "factual": 7,
            "procedural": 5,
            "numeric": 5,
            "cross_reference": 5,
            "multi_section": 3,
        }
    )

    # each type has 9 cpa2019, 1 ecom2020 and 2 mediation2020 survivors: documents are filled
    # as evenly as that allows
    def per_doc(t: str) -> dict[str, int]:
        return dict(Counter(it.doc_id for it in chosen if it.qtype == t))

    assert per_doc("definition") == {"cpa2019": 2, "ecom2020": 1, "mediation2020": 2}
    assert per_doc("factual") == {"cpa2019": 4, "ecom2020": 1, "mediation2020": 2}
    assert per_doc("multi_section") == {"cpa2019": 1, "ecom2020": 1, "mediation2020": 1}
    assert [it.number for it in chosen] == sorted(it.number for it in chosen)
    # shortfall: only 10 survivors for a target of 90
    few, rep = select_answerable(items[:10], 90)
    assert len(few) == 10 and rep["shortfall"] == 80 and rep["selected"] == 10
    assert rep["by_type"]["definition"]["mix_target"] == 15


def test_select_unanswerable_balances_methods() -> None:
    items = [UnanswerableItem("out_of_scope", f"t{i % 5}", f"q{i}", "r", i) for i in range(20)]
    items += [UnanswerableItem("perturbation", f"k{i % 4}", f"p{i}", "r", 20 + i) for i in range(4)]
    chosen, rep = select_unanswerable(items, 10)
    m = Counter(it.method for it in chosen)
    assert m == {"out_of_scope": 6, "perturbation": 4}  # perturbation has only 4: rest from oos
    assert len({it.group for it in chosen if it.method == "out_of_scope"}) == 5
    assert rep["shortfall"] == 0 and rep["by_method"]["perturbation"]["available"] == 4
    chosen, rep = select_unanswerable(items[:3], 30)
    assert len(chosen) == 3 and rep["shortfall"] == 27


@pytest.mark.parametrize(
    "k,n,lo,hi", [(0, 20, 0.0, 0.1611), (2, 20, 0.0279, 0.3010), (10, 10, 0.7225, 1.0)]
)
def test_wilson_interval(k: int, n: int, lo: float, hi: float) -> None:
    from ragsys.eval.spot_check import wilson_interval

    ci = wilson_interval(k, n)
    assert (
        ci is not None
        and ci[0] == pytest.approx(lo, abs=1e-4)
        and ci[1] == pytest.approx(hi, abs=1e-4)
    )
    assert wilson_interval(0, 0) is None
