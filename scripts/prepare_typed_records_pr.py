#!/usr/bin/env python
"""Stage the typed-record migration on the next branch in the isolated stack clone."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/pr-stack"
BASE = "codex/15-workflow-quality"
BRANCH = "codex/16-typed-records"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(DEST), *args], text=True).strip()


def main() -> None:
    if git("branch", "--show-current") != BASE or git("status", "--porcelain"):
        raise ValueError(f"Expected clean isolated branch {BASE}")
    files = {
        "AGENTS.md",
        "pyproject.toml",
        "docs/development.md",
        "docs/typed-records.md",
        "docs/architecture.md",
        "docs/architecture-view.html",
        "configs/experiments/typed-records-validation.json",
        "tests/fixtures/error-report-wire.json",
    }
    for folder in ("relift", "scripts", "tests", "packages/frontends", "stubs"):
        files.update(
            str(path.relative_to(ROOT))
            for path in (ROOT / folder).rglob("*")
            if path.is_file()
            and path.suffix in (".py", ".pyi")
            and not {".hatch", "__pycache__"}.intersection(path.parts)
        )
    changed = [
        name
        for name in sorted(files)
        if not (DEST / name).exists() or (DEST / name).read_bytes() != (ROOT / name).read_bytes()
    ]
    git("checkout", "-b", BRANCH)
    for name in changed:
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    git("add", *changed)
    print(git("diff", "--cached", "--stat"))


if __name__ == "__main__":
    main()
