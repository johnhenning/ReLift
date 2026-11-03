#!/usr/bin/env python
"""Copy one completed native experiment into the next isolated review branch.

This stages files but does not commit, push or open a PR. The shared workspace's
Git index is never changed. Validate the staged diff before publishing.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/pr-stack"
BRANCHES = ["codex/05-optimization-experiments", "codex/06-native-batch-gather",
            "codex/07-native-shard-writer", "codex/08-native-cuda-rays", "codex/09-native-da3-packing"]


def git(*args):
    return subprocess.check_output(["git", "-C", str(DEST), *args], text=True).strip()


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + "/", "")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layer", type=int, choices=range(1,5))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    expected = BRANCHES[args.layer if args.refresh else args.layer-1]
    if git("branch", "--show-current") != expected or git("status", "--porcelain"):
        raise ValueError(f"Expected clean isolated branch {expected}")
    if not args.refresh:
        git("checkout", "-b", BRANCHES[args.layer])
    files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    for filename in files:
        source = ROOT / filename
        if not source.is_file() or source.suffix in (".mp4", ".so"):
            continue
        target = DEST / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if filename == "README.md":
            target.write_text(source.read_text().replace("https://github.com/ReLift3D/ReLift.git", "https://github.com/johnhenning/ReLift.git"))
        else:
            shutil.copy2(source, target)
    campaign = ROOT / "outputs/experiments/native-optimizations"
    evidence = DEST / "docs/benchmarks/native-optimizations"
    evidence.mkdir(parents=True, exist_ok=True)
    for filename in ("summary.json", "environment.json"):
        (evidence / filename).write_text(json.dumps(portable(json.loads((campaign / filename).read_text())), indent=2)+"\n")
    trials = [portable(json.loads(p.read_text())) for p in sorted(campaign.rglob("trial.json"))]
    (evidence / "trials.json").write_text(json.dumps(trials, indent=2)+"\n")
    quality = {str(p.relative_to(campaign)): portable(json.loads(p.read_text())) for p in sorted(campaign.rglob("quality.json"))}
    (evidence / "quality.json").write_text(json.dumps(quality, indent=2)+"\n")
    for image in (ROOT / "outputs/native-optimization-charts").glob("*.png"):
        shutil.copy2(image, evidence / image.name)
    git("add", "-A")
    print(git("diff", "--cached", "--stat"))


if __name__ == "__main__":
    main()
