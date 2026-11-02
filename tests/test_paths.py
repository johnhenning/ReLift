"""No machine-specific paths, user names, host names or scheduler references in the released code."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN = ["relift", "scripts", "configs", "tests", "README.md", ".python-version", "pyproject.toml", "packages", "docs"]
FORBIDDEN = [r"/home/\w", r"/datasets/", r"/tmp/claude", r"rds_phase0", r"\bassafsin\b", r"maroun\.kh", r"sbatch", r"SLURM",
             r"squeue", r"densetrack3d", r"PYTHONNOUSERSITE", r"espresso", r"nlp-pro6000", r"miniconda3"]


def test_no_machine_specific_references():
    hits = []
    for top in SCAN:
        p = REPO / top
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and f.suffix in {".py", ".yaml", ".yml", ".md", ".toml", ".sh", ".txt"}]
        for f in files:
            text = f.read_text(errors="ignore")
            for pat in FORBIDDEN:
                for m in re.finditer(pat, text):
                    line = text[:m.start()].count("\n") + 1
                    if f.name == "test_paths.py" or (f.name == "setup.sh" and pat == "PYTHONNOUSERSITE"):
                        continue                      # setup.sh sets the user-site isolation variable on purpose
                    hits.append(f"{f.relative_to(REPO)}:{line}: {pat}")
    assert not hits, "\n".join(hits)
