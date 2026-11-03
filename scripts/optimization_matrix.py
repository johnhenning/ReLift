#!/usr/bin/env python
"""Expand optimization settings into a reproducible experiment spec; run separately.

uv run python scripts/optimization_matrix.py --source FROZEN_SOURCE \
    --grid configs/optimizations/example-grid.json --out outputs/matrix.json \
    --output-root outputs/experiments/combination-matrix
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relift.configuration import InferenceConfig, load_configuration, merge
from relift.optimization import OptimizationConfig
from relift.records.benchmark import ExperimentOptions, ExperimentSpec, ExperimentVariant
from relift.records.configuration import InferenceConfigurationRecord, OptimizationRecord


def expand(
    base: Mapping[str, object], grid: Mapping[str, Sequence[object]]
) -> list[OptimizationRecord]:
    if not grid or any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("Grid must map settings to nonempty lists")
    names = sorted(grid)
    variants = []
    for values in itertools.product(*(grid[name] for name in names)):
        settings = dict(base, **dict(zip(names, values, strict=False)))
        resolved = OptimizationConfig.from_dict(settings)
        variants.append(resolved.as_dict())
    return variants


def isolate(
    base: Mapping[str, object], grid: Mapping[str, Sequence[object]]
) -> list[OptimizationRecord]:
    if not grid or any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("Grid must map settings to nonempty lists")
    variants = [OptimizationConfig.from_dict(base).as_dict()]
    for key in sorted(grid):
        for value in grid[key]:
            resolved = OptimizationConfig.from_dict(dict(base, **{key: value})).as_dict()
            if resolved not in variants:
                variants.append(resolved)
    return variants


def component_matrix(
    base: Mapping[str, object], grid: Mapping[str, Sequence[object]], *, isolated: bool = False
) -> list[InferenceConfigurationRecord]:
    """Expand dotted component fields, with the same validator as inference."""
    if not grid or any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("Grid must map component fields to nonempty lists")

    def patch(name: str, value: object) -> dict[str, dict[str, object]]:
        parts = name.split(".")
        if len(parts) != 2:
            raise ValueError("Component grid keys must be section.field")
        return {parts[0]: {parts[1]: value}}

    candidates = [base] if isolated else []
    if isolated:
        candidates.extend(
            merge(base, patch(name, value)) for name in sorted(grid) for value in grid[name]
        )
    else:
        names = sorted(grid)
        for values in itertools.product(*(grid[name] for name in names)):
            selected = base
            for name, value in zip(names, values, strict=True):
                selected = merge(selected, patch(name, value))
            candidates.append(selected)
    result = []
    for candidate in candidates:
        resolved = InferenceConfig.from_dict(candidate).as_dict()
        if resolved not in result:
            result.append(resolved)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--base", type=Path, help="legacy flat JSON base")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--profile", help="named component TOML base")
    group.add_argument("--inference-config", type=Path, help="component TOML base")
    parser.add_argument("--native-build")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--task", choices=("correction", "pipeline", "da3-configured"), default="correction"
    )
    parser.add_argument("--mode", choices=("cartesian", "one-at-a-time"), default="cartesian")
    parser.add_argument("--input", default="cache/demo_inputs/swing-tire.mp4")
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--offload", choices=("ram", "disk"))
    parser.add_argument(
        "--prepared-window",
        default="outputs/benchmarks/offload-primary/swing-tire/ram/trial-01/output/window",
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    grid = json.loads(args.grid.read_text())
    grouped = (
        args.profile is not None
        or args.inference_config is not None
        or any("." in key for key in grid)
    )
    option_sets: list[ExperimentOptions]
    if grouped:
        if args.base:
            parser.error("--base is legacy JSON; choose --profile or --inference-config")
        base = load_configuration(profile=args.profile, path=args.inference_config).config.as_dict()
        option_sets = [
            dict(configuration=config)
            for config in component_matrix(base, grid, isolated=args.mode == "one-at-a-time")
        ]
    else:
        base = json.loads(args.base.read_text()) if args.base else {}
        option_sets = [
            dict(optimizations=config)
            for config in (expand if args.mode == "cartesian" else isolate)(base, grid)
        ]
    variants: list[ExperimentVariant] = []
    for index, options in enumerate(option_sets):
        if args.task == "pipeline":
            options.update(video=args.input)
            if args.offload or not grouped:
                options["offload"] = args.offload or "ram"
        elif args.task == "da3-configured":
            options.update(clips=args.clips or [args.input])
            if args.offload or not grouped:
                options["offload"] = args.offload or "ram"
        elif args.offload or args.clips:
            raise ValueError(
                "Correction uses a prepared window; offload/clips apply to provider or pipeline tasks"
            )
        if args.native_build:
            options["native_build"] = args.native_build
        variants.append(dict(name=f"variant-{index:03}", source=args.source, options=options))
    spec: ExperimentSpec = dict(
        scene=Path(args.input).stem,
        input=args.input,
        process_res=504,
        prepared_window=args.prepared_window,
        output_root=args.output_root,
        experiments={"matrix": dict(task=args.task, repeats=args.repeats, variants=variants)},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {len(variants)} explicit variants; no trials started: {args.out}")


if __name__ == "__main__":
    main()
