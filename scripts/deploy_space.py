"""Create or update the Hugging Face Space <username>/agentic-rag-assistant from
deploy/hf_space/build/ (assemble it first with scripts/build_space.py).

Usage (PowerShell):
    $env:HF_TOKEN = "hf_..."            # a write token; read from the environment only
    python scripts/deploy_space.py [--space-name agentic-rag-assistant] [--private]
                                   [--set-llm-secret] [--dry-run]

The username comes from the token (whoami). The Space is created as a public Docker Space when
it does not exist, then the build folder is uploaded as one commit (files that are no longer in
the build are removed from the Space). --set-llm-secret also stores the LLM_API_KEY of this
shell as the Space secret LLM_API_KEY; without it, add the secret in the Space settings. The
build is scanned again before anything is sent (no PDFs, no .env files, no token or key values).
Token and key values are never printed or written.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ragsys.config import resolve_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_space import DEFAULT_OUT, MARKER, scan, secret_values

DEFAULT_SPACE = "agentic-rag-assistant"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--build", default=str(DEFAULT_OUT))
    ap.add_argument("--space-name", default=DEFAULT_SPACE)
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument(
        "--set-llm-secret",
        action="store_true",
        help="store this shell's LLM_API_KEY as the Space secret LLM_API_KEY",
    )
    ap.add_argument("--dry-run", action="store_true", help="check everything, upload nothing")
    args = ap.parse_args(argv)

    build = resolve_path(Path(args.build))
    if not (build / MARKER).exists() or not (build / "Dockerfile").exists():
        print(f"{build} is not a Space build; run python scripts/build_space.py first")
        return 2
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print(
            'HF_TOKEN is not set (PowerShell: $env:HF_TOKEN = "hf_..."; a token with write access)'
        )
        return 2
    problems = scan(build, secret_values())
    if problems:
        print("The build contains files that must not be uploaded:")
        for p in problems:
            print(f"  {p}")
        return 1
    llm_key = os.environ.get("LLM_API_KEY", "").strip()
    if args.set_llm_secret and not llm_key:
        print("--set-llm-secret: LLM_API_KEY is not set in this shell")
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami().get("name")
    if not user:
        print("could not read the username of this token")
        return 2
    repo_id = f"{user}/{args.space_name}"
    n_files = sum(1 for p in build.rglob("*") if p.is_file())
    print(
        f"Space: {repo_id} ({'private' if args.private else 'public'}), {n_files} files from {build}"
    )
    if args.dry_run:
        print("dry run: nothing was created or uploaded")
        return 0
    api.create_repo(
        repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True
    )
    if args.set_llm_secret:
        api.add_space_secret(repo_id, "LLM_API_KEY", llm_key)
        print("Space secret LLM_API_KEY set (value not shown)")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(build),
        commit_message="Update the Space build",
        ignore_patterns=[MARKER],
        delete_patterns=["*", "**/*"],
    )
    print(f"Uploaded. The Space builds now: https://huggingface.co/spaces/{repo_id}")
    if not args.set_llm_secret:
        print(
            "Add the secret LLM_API_KEY under Settings -> Variables and secrets if it is not set."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
