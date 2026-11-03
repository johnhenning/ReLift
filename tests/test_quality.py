"""The existing CPU CI entry point enforces repository-wide Python quality."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("ruff", ["format", "--check", "."]),
        ("ruff", ["check", "."]),
        ("pyrefly", ["check", "--min-severity", "warn"]),
    ],
)
def test_python_quality(tool: str, arguments: list[str]) -> None:
    subprocess.run([sys.executable, "-m", tool, *arguments], cwd=ROOT, check=True)
