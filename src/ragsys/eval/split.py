"""Deterministic dev/test split of the evaluation set, and the test-split guard.

The split is computed from the question id and never stored:

    split = "dev"  if int(sha256("split-v1:" + id)[:8], 16) / 2**32 < 0.30  else "test"

Adding questions never moves existing ones between splits.

Test questions can only be obtained with a :class:`TestAccess` grant, and
:func:`grant_test_access` only grants it when called from ``scripts/eval_retrieval.py``; each
grant is appended to ``results/test_access_log.json``. A test in the suite also checks the
source tree statically so no other file calls :func:`grant_test_access`.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import inspect
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

from ragsys.utils.runinfo import git_info

Split = Literal["dev", "test"]
SPLIT_SALT = "split-v1:"
DEV_FRACTION = 0.30
TEST_EVAL_SCRIPT = ("scripts", "eval_retrieval.py")  # the only caller allowed to read test

_GRANT_KEY = object()


def split_of(question_id: str) -> Split:
    digest = hashlib.sha256(f"{SPLIT_SALT}{question_id}".encode()).hexdigest()
    return "dev" if int(digest[:8], 16) / 2**32 < DEV_FRACTION else "test"


class TestSplitLocked(PermissionError):
    """Raised when test-split questions are requested without a valid grant."""

    __test__ = False  # not a pytest test class


class TestAccess:
    """Proof that the test split was opened through :func:`grant_test_access`."""

    __test__ = False  # not a pytest test class

    def __init__(self, key: object, caller: str) -> None:
        if key is not _GRANT_KEY:
            raise TestSplitLocked("TestAccess can only be created by grant_test_access()")
        self.caller = caller
        self._key = key

    @property
    def valid(self) -> bool:
        return self._key is _GRANT_KEY


def _is_test_eval_script(filename: str) -> bool:
    p = Path(filename).resolve()
    return (p.parent.name, p.name) == TEST_EVAL_SCRIPT


def read_access_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    data = json.loads(log_path.read_text(encoding="utf-8"))
    return list(data) if isinstance(data, list) else []


def grant_test_access(
    log_path: Path, configs: Sequence[str], n_questions: int, note: str = ""
) -> TestAccess:
    """Open the test split (only from scripts/eval_retrieval.py) and log the access."""
    caller = inspect.stack(context=0)[1].filename
    if not _is_test_eval_script(caller):
        raise TestSplitLocked(
            "the test split can only be evaluated by scripts/eval_retrieval.py "
            f"(called from {caller}); use --split dev for development"
        )
    entries = read_access_log(log_path)
    git = git_info()
    entries.append(
        {
            "utc_time": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "configs": list(configs),
            "n_questions": n_questions,
            "note": note,
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return TestAccess(_GRANT_KEY, caller)


Q = TypeVar("Q")


def questions_for_split(
    questions: Iterable[Q], split: Split, access: TestAccess | None = None
) -> list[Q]:
    """Questions of one split. ``split="test"`` requires a grant from grant_test_access()."""
    if split == "test" and not (isinstance(access, TestAccess) and access.valid):
        raise TestSplitLocked(
            "test-split questions requested without a TestAccess grant; "
            "only scripts/eval_retrieval.py may evaluate the test split"
        )
    if split not in ("dev", "test"):
        raise ValueError(f"unknown split {split!r}")
    return [q for q in questions if split_of(str(getattr(q, "id", q))) == split]
