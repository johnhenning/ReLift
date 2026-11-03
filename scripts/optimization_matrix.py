#!/usr/bin/env python
"""Expand optimization settings into a reproducible experiment spec; run separately.

uv run python scripts/optimization_matrix.py --source FROZEN_SOURCE \
    --grid configs/optimizations/example-grid.json --out outputs/matrix.json \
    --output-root outputs/experiments/combination-matrix
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from relift.optimization import OptimizationConfig


def expand(base, grid):
    if not grid or any(not isinstance(values,list) or not values for values in grid.values()):
        raise ValueError("Grid must map settings to nonempty lists")
    names = sorted(grid)
    variants = []
    for values in itertools.product(*(grid[name] for name in names)):
        settings = dict(base, **dict(zip(names, values)))
        resolved = OptimizationConfig.from_dict(settings)
        variants.append(resolved.as_dict())
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--native-build")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task", choices=("correction","pipeline"), default="correction")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    grid = json.loads(args.grid.read_text())
    base = json.loads(args.base.read_text()) if args.base else {}
    settings = expand(base,grid)
    variants = []
    for index, config in enumerate(settings):
        options = dict(optimizations=config)
        if args.task == "pipeline":
            options["video"] = "cache/demo_inputs/swing-tire.mp4"
        if args.native_build:
            options["native_build"] = args.native_build
        variants.append(dict(name=f"variant-{index:03}", source=args.source, options=options))
    spec = dict(scene="swing-tire", input="cache/demo_inputs/swing-tire.mp4", process_res=504,
                prepared_window="outputs/benchmarks/offload-primary/swing-tire/ram/trial-01/output/window",
                output_root=args.output_root, experiments={"matrix":dict(task=args.task,repeats=args.repeats,variants=variants)})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(spec,indent=2)+"\n")
    print(f"Wrote {len(variants)} explicit variants; no trials started: {args.out}")


if __name__ == "__main__":
    main()
