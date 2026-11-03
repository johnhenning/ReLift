#!/usr/bin/env python
"""Stage the quality pass in the isolated stack clone without committing or pushing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/pr-stack"
BASE = "codex/11-observation-cache"
BRANCH = "codex/12-python-quality"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(DEST), *args], text=True).strip()


def main() -> None:
    if git("branch", "--show-current") != BASE or git("status", "--porcelain"):
        raise ValueError(f"Expected clean isolated branch {BASE}")
    files = {
        ".github/workflows/tests.yml",
        ".gitignore",
        "AGENTS.md",
        "pyproject.toml",
        "uv.lock",
        "docs/development.md",
        "docs/architecture-view.html",
        "configs/experiments/quality-validation.json",
    }
    for folder in ("relift", "scripts", "tests", "packages/frontends", "stubs"):
        files.update(
            str(path.relative_to(ROOT))
            for path in (ROOT / folder).rglob("*")
            if path.is_file()
            and path.suffix in (".py", ".pyi", ".md")
            and not {".hatch", "__pycache__"}.intersection(path.parts)
        )
    git("checkout", "-b", BRANCH)
    for filename in sorted(files):
        target = DEST / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / filename, target)
    git("add", *sorted(files))
    print(git("diff", "--cached", "--stat"))


if __name__ == "__main__":
    main()
