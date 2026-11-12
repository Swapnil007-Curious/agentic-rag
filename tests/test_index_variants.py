"""Frozen evaluation corpora and app-only documents: the cs_ai app index adds two books without
touching the evaluation corpus; every evaluation script refuses an app index and the collections
that are not evaluated; ask.py and the UI serve the app index (or, until it is built, the
evaluation index with a note)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from ragsys.agents.runtime import serving_retrieval
from ragsys.collections import (
    AppIndexRefused,
    Collection,
    all_collections,
    eval_index_refusal,
    get_collection,
    load_collections,
)
from ragsys.config import load_ingest_config
from ragsys.retrieval.config import load_retrieval_settings

APP_RS = "configs/collections/cs_ai/app/retrieval.yaml"


def test_eval_and_app_documents() -> None:
    cs = get_collection("cs_ai")
    assert cs.spec.eval_docs == ["d2l", "progit", "ods"]
    assert cs.spec.app_docs == ["d2l", "progit", "ods", "python_tutorial", "think_python_2e"]
    assert cs.has_app_index and cs.app_processed_dir.as_posix() == "collections/cs_ai/processed_app"
    assert cs.processed_dir.as_posix() == "collections/cs_ai/processed"  # frozen corpus
    ev = load_ingest_config(cs.ingest_config)
    app = load_ingest_config(cs.app_ingest_config)
    assert [d.doc_id for d in ev.documents] == cs.spec.eval_docs
    eval_entries = {d.doc_id: d.model_dump() for d in ev.documents}
    for d in app.documents[:3]:
        assert d.model_dump() == eval_entries[d.doc_id]  # copied verbatim
    tut = next(d for d in app.documents if d.doc_id == "python_tutorial")
    assert tut.file == "python_tutorial/tutorial.pdf" and tut.outline_chapters
    cl = get_collection("consumer_law")
    assert cl.spec.app_docs == cl.spec.eval_docs and not cl.has_app_index
    assert load_retrieval_settings(cs.retrieval_config).index == "eval"
    assert load_retrieval_settings(cs.app_retrieval_config).index == "app"


def test_new_collections_are_app_only() -> None:
    reg = load_collections()
    assert reg.ordered()[:2] == ["cs_ai", "consumer_law"]
    labels = {c.name: c.spec.ui.label for c in all_collections()}
    assert labels == {
        "consumer_law": "Indian Consumer Law",
        "cs_ai": "Computer Science, AI & Coding",
        "indian_law": "Indian Law, Constitution & Tax",
        "human_biology": "Human Biology",
        "physics_chemistry": "Physics & Chemistry",
    }
    for name in ("indian_law", "human_biology", "physics_chemistry"):
        c = get_collection(name)
        assert not c.evaluated and not c.spec.eval_docs and len(c.spec.ui.examples) == 6
        assert load_retrieval_settings(c.retrieval_config).index == "app"
        assert c.app_index_models == ["bge-small"]
    assert get_collection("indian_law").spec.app_docs == ["coi", "ita2025", "cgst2017", "bns2023"]


def test_refusal_rules() -> None:
    cs = get_collection("cs_ai")
    assert eval_index_refusal(cs, load_retrieval_settings(cs.retrieval_config)) is None
    msg = eval_index_refusal(cs, load_retrieval_settings(APP_RS))
    assert msg and "app index" in msg
    il = get_collection("indian_law")
    assert "not evaluated" in (
        eval_index_refusal(il, load_retrieval_settings(il.retrieval_config)) or ""
    )


EVAL_SCRIPTS: list[tuple[str, list[str]]] = [
    ("run_retrieval_grid", ["--split", "dev"]),
    ("eval_retrieval", ["--split", "test"]),
    ("select_configs", []),
    ("build_auto_eval", []),
    ("spot_check", []),
    ("tune_abstention", []),
    ("run_generation", ["--split", "dev"]),
    ("derive_systems", ["--split", "dev"]),
    ("eval_generation", ["--split", "test"]),
]


@pytest.mark.parametrize(("name", "args"), EVAL_SCRIPTS, ids=[n for n, _a in EVAL_SCRIPTS])
def test_evaluation_scripts_refuse_the_app_index(
    name: str,
    args: list[str],
    script: Callable[[str], ModuleType],
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = script(name)
    flag = (
        "--config"
        if name
        in (
            "run_retrieval_grid",
            "eval_retrieval",
            "select_configs",
            "build_auto_eval",
            "spot_check",
        )
        else "--retrieval-config"
    )
    before = Path("collections/cs_ai/results/test_access_log.json").read_bytes()
    assert mod.main([*args, "--collection", "cs_ai", flag, APP_RS]) == 2
    assert "app index" in capsys.readouterr().out
    assert mod.main([*args, "--collection", "indian_law"]) == 2
    assert "not evaluated" in capsys.readouterr().out
    assert Path("collections/cs_ai/results/test_access_log.json").read_bytes() == before


def test_library_level_refusal(retrieval_dir: Path, tmp_path: Path) -> None:
    from ragsys.eval.harness import Evaluator, run_configs
    from ragsys.retrieval.pipeline import RetrievalResources
    from retrieval_helpers import fake_settings, write_questions

    q = tmp_path / "q.jsonl"
    write_questions(q)
    rs = fake_settings(retrieval_dir, tmp_path / "res", q)
    rs.index = "app"
    ev = Evaluator(rs, RetrievalResources(rs))
    with pytest.raises(AppIndexRefused):
        run_configs(ev, rs.grid_configs()[:1], [], "dev", tmp_path / "out")


def test_serving_index(tmp_path: Path) -> None:
    cs = get_collection("cs_ai")
    rs, note = serving_retrieval(cs)
    app_npy = Path("collections/cs_ai/processed_app/emb")
    if not app_npy.exists():
        assert rs.index == "eval" and "python_tutorial" in note and "run_app_collections" in note
    else:
        assert rs.index in ("app", "eval")
    il_rs, il_note = serving_retrieval(get_collection("indian_law"))
    assert il_rs.index == "app" and il_note == ""
    assert isinstance(cs, Collection)


def test_app_sources_file_is_separate(script) -> None:  # type: ignore[no-untyped-def]
    """The cs_ai app ingest writes SOURCES_app.csv (the evaluation corpus's SOURCES.csv keeps
    its three rows), and the UI lists it only while it serves the app index."""
    from ragsys.agents.runtime import load_settings
    from ragsys.config import load_ingest_config, resolve_path

    cs = get_collection("cs_ai")
    app_ing = load_ingest_config(resolve_path(cs.app_ingest_config))
    eval_ing = load_ingest_config(resolve_path(cs.ingest_config))
    assert app_ing.paths.sources_file == "SOURCES_app.csv"
    assert eval_ing.paths.sources_file == "SOURCES.csv"
    serve = script("serve")
    settings, _ = load_settings(resolve_path(cs.generation_config))
    evaluated = serve.sources_csv(cs, settings, served_app=False)
    assert evaluated.name == "SOURCES.csv"
    app = serve.sources_csv(cs, settings, served_app=True)
    if resolve_path(app_ing.paths.sources_path()).exists():
        assert app.name == "SOURCES_app.csv"
    rows = evaluated.read_text(encoding="utf-8").strip().splitlines()
    assert [r.split(",")[1] for r in rows[1:]] == ["d2l", "progit", "ods"]
    il = get_collection("indian_law")
    il_settings, _ = load_settings(resolve_path(il.generation_config))
    assert serve.sources_csv(il, il_settings, served_app=True).name == "SOURCES.csv"
