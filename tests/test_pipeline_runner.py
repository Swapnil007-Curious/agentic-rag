"""The resumable pipeline runner with fake steps (skip, resume, fingerprints, non-fatal and
always-run steps, accepted exit codes) and the cs_ai pipeline plan."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from ragsys.collections import get_collection
from ragsys.pipeline_plan import build_steps, estimate_steps, plan_context, write_summary
from ragsys.pipeline_runner import Log, Step, read_state, run_steps


class FakeWorld:
    """Steps "produce" files; the executor returns scripted exit codes."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.ran: list[str] = []
        self.codes: dict[str, list[int]] = {}
        self.inputs = {"a": "v1"}

    def executor(self, step: Step, out: Callable[[str], None]) -> tuple[int, int]:
        self.ran.append(step.name)
        out(f"running {step.name} {' '.join(step.argv)}")
        codes = self.codes.get(step.name, [0])
        rc = codes.pop(0) if len(codes) > 1 else codes[0]
        if rc in (0, 3):
            (self.tmp / f"{step.name}.done").write_text("x", encoding="utf-8")
        return rc, 50 * 1024**2

    def step(self, name: str, **kw: object) -> Step:
        return Step(
            name,
            ["fake", name],
            done=lambda: (self.tmp / f"{name}.done").exists(),
            **kw,  # type: ignore[arg-type]
        )


def _run(world: FakeWorld, steps: list[Step], tmp: Path) -> tuple[bool, str | None]:
    log = Log(tmp / "log.txt", echo=False)
    rep = run_steps(
        steps, tmp / "state.json", log, executor=world.executor, resume_hint="resume-cmd"
    )
    log.close()
    return rep.complete, rep.failed


def test_resume_after_failure_and_skip_completed(tmp_path: Path) -> None:
    w = FakeWorld(tmp_path)
    w.codes = {"b": [1, 0]}  # b fails on the first run, succeeds on the second
    steps = [
        w.step("a", fingerprint=lambda: w.inputs["a"]),
        w.step("b"),
        Step("check", ["fake", "check"], always=True),
        w.step("c"),
    ]
    assert _run(w, steps, tmp_path) == (False, "b")
    assert w.ran == ["a", "b"]
    log = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert (
        "FAIL  b (exit 1)" in log and "resume with: resume-cmd" in log and "peak RSS 50 MB" in log
    )
    assert "running a fake a" in log
    state = read_state(tmp_path / "state.json")
    assert list(state["steps"]) == ["a"] and state["steps"]["a"]["fingerprint"] == "v1"

    w.ran.clear()
    assert _run(w, steps, tmp_path) == (True, None)
    assert w.ran == ["b", "check", "c"]  # a skipped, the always-run check ran
    assert "PIPELINE COMPLETE" in (tmp_path / "log.txt").read_text(encoding="utf-8")

    w.ran.clear()
    assert _run(w, steps, tmp_path) == (True, None)
    assert w.ran == ["check"]  # everything complete: only the always-run step

    w.ran.clear()
    w.inputs["a"] = "v2"  # changed inputs make a step run again
    assert _run(w, steps, tmp_path) == (True, None)
    assert w.ran == ["a", "check"]


def test_non_fatal_accept_codes_and_completion_check(tmp_path: Path) -> None:
    w = FakeWorld(tmp_path)
    w.codes = {"optional": [127], "exists": [3], "liar": [0]}
    liar = Step("liar", ["fake", "liar"], done=lambda: False)  # exit 0 but no output
    steps = [
        Step("optional", ["fake", "optional"], always=True, fatal=False),
        w.step("exists", accept_codes=(0, 3)),
        liar,
    ]
    assert _run(w, steps, tmp_path) == (False, "liar")
    log = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "WARN  optional failed (exit 127); continuing" in log
    assert "DONE  exists" in log and "completion check of liar failed" in log


def test_missing_command_is_reported(tmp_path: Path) -> None:
    from ragsys.pipeline_runner import subprocess_executor

    log = Log(tmp_path / "log.txt", echo=False)
    steps = [Step("ghost", ["definitely-not-a-command-xyz"], always=True, fatal=False)]
    rep = run_steps(steps, tmp_path / "state.json", log, executor=subprocess_executor)
    log.close()
    assert rep.complete and rep.outcomes[0].status == "warned"
    ok = [Step("py", [sys.executable, "-c", "print('hello from step')"], always=True)]
    log = Log(tmp_path / "log2.txt", echo=False)
    rep = run_steps(ok, tmp_path / "state2.json", log, executor=subprocess_executor)
    log.close()
    assert rep.complete and "hello from step" in (tmp_path / "log2.txt").read_text(encoding="utf-8")


def test_cs_ai_plan_order_systems_and_estimates() -> None:
    coll = get_collection("cs_ai")
    ctx = plan_context(coll, "configs/collections.yaml")
    steps = build_steps(ctx)
    assert [s.name for s in steps] == [
        "ingest",
        "chunk",
        "build_index",
        "build_auto_eval",
        "validate_eval",
        "ollama_stop",
        "retrieval_dev_grid",
        "select_configs",
        "eval_retrieval_test",
        "generation_dev",
        "derive_dev",
        "generation_test",
        "summary",
    ]
    by = {s.name: s for s in steps}
    assert all(
        "--collection" in s.argv and "cs_ai" in s.argv for s in steps if s.name != "ollama_stop"
    )
    assert by["build_index"].argv[-6:-4] == ["bge-base", "--no-chroma"]
    assert by["generation_dev"].argv[by["generation_dev"].argv.index("--systems") + 1 :][:2] == [
        "G2",
        "G4",
    ]
    tail = by["generation_test"].argv
    assert tail[tail.index("--systems") + 1 : tail.index("--systems") + 5] == [
        "G1",
        "G2",
        "G3",
        "G4",
    ]
    assert by["derive_dev"].argv[-5] == "G3"
    assert by["validate_eval"].always and by["validate_eval"].fatal
    assert not by["ollama_stop"].fatal and by["ollama_stop"].argv[:2] == ["ollama", "stop"]
    assert ctx.grid_size == 16 and ctx.chunkers == ["fixed_512", "structural"]
    est, basis, info = estimate_steps(ctx)
    assert set(est) == {s.name for s in steps} and all(v > 0 for v in est.values())
    assert est["build_auto_eval"] > 1800 and est["generation_test"] > est["generation_dev"]
    assert basis and info["n_dev"] + info["n_test"] == 80


def test_summary_writes_without_results(tmp_path: Path) -> None:
    import yaml

    reg = yaml.safe_load(Path("configs/collections.yaml").read_text(encoding="utf-8"))
    spec = reg["collections"]["cs_ai"]
    spec["roots"]["results_dir"] = str(tmp_path / "results")
    spec["roots"]["processed_dir"] = str(tmp_path / "processed")
    path = tmp_path / "collections.yaml"
    path.write_text(
        yaml.safe_dump({"collections": {"cs_ai": spec}, "default": "cs_ai"}), encoding="utf-8"
    )
    out = write_summary(get_collection("cs_ai", path))
    assert out.name == "SUMMARY.md" and "# CS / AI textbooks: pipeline summary" in out.read_text(
        encoding="utf-8"
    )
