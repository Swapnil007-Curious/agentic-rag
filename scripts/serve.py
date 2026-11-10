"""Start the local web UI on http://127.0.0.1:8000 (loopback only).

Usage:
    python scripts/serve.py [--port 8000] [--no-warm]
Ask box, system selector (G2, G4, G5, G6; default G5), plain-language toggle, answers with
clickable citation chips that open the cited passage (heading path, section id, pages, text with
the sentence's key terms highlighted), "Not found in these documents" with the three closest
sections when the system abstains, a collapsible step trace with timings, and thumbs up/down
stored in data/feedback.jsonl. The server always binds to 127.0.0.1; it is meant for one local
user. Needs Ollama running (configs/llm.yaml). Stop it with Ctrl+C.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from ragsys.agents.runtime import build_engine, llm_unreachable_message, load_settings, make_llm
from ragsys.app.server import create_app
from ragsys.utils.console import emit
from ragsys.utils.memory import set_torch_threads
from ragsys.utils.seed import set_seed

HOST = "127.0.0.1"  # never exposed beyond this machine


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", type=int, default=None, help="default: ui.port (8000)")
    ap.add_argument("--config", default="configs/generation.yaml")
    ap.add_argument("--retrieval-config", default=None)
    ap.add_argument("--no-warm", action="store_true", help="load models on the first question")
    ap.add_argument("--fake-llm", action="store_true", help="offline deterministic stand-in")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    settings, rsettings = load_settings(args.config, args.retrieval_config)
    set_seed(settings.seed)
    set_torch_threads(args.threads)
    llm = make_llm(settings, fake=args.fake_llm)
    problem = llm_unreachable_message(llm)
    if problem:
        emit(f"WARNING: {problem}; questions will fail until it is reachable")
    engine = build_engine(settings, rsettings, llm)
    for w in engine.warnings_for(settings.ui.default_system):
        emit(f"WARNING: {w}")
    if not args.no_warm:
        emit(f"loading the retriever of {settings.ui.default_system} ...")
        engine.orchestrator(settings.ui.default_system)
    port = args.port or settings.ui.port
    app = create_app(settings, engine)
    emit(f"serving on http://{HOST}:{port}  (Ctrl+C to stop)")
    try:
        uvicorn.run(app, host=HOST, port=port, log_level="warning")
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
