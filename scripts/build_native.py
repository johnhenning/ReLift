#!/usr/bin/env python
"""Build optional CPU helpers outside inference and persist exact compiler provenance.

uv run python scripts/build_native.py --out outputs/native/gather
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument(
        "--source", type=Path, default=ROOT, help="Compile an explicit immutable source snapshot"
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite native build: {args.out}")
    args.out.mkdir(parents=True)
    source_root = args.source.resolve()
    source = source_root / "native/cpu.cpp"
    binary = args.out.resolve() / ("_relift_cpu" + sysconfig.get_config_var("EXT_SUFFIX"))
    # pybind11 headers ship in the pinned torch wheel; no extra runtime dependency.
    command = [
        args.cxx,
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-fno-fast-math",
        "-I" + sysconfig.get_path("include"),
        "-I" + str(Path(torch.__file__).parent / "include"),
        str(source),
        "-o",
        str(binary),
        "-lcrypto",
    ]
    start = time.perf_counter()
    cuda_command = None
    if args.cuda:
        obj = args.out.resolve() / "rays.o"
        cuda_command = [
            args.nvcc,
            "-O3",
            "-std=c++17",
            "-ccbin",
            args.cxx,
            "-Xcompiler",
            "-fPIC",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "-arch=" + args.arch,
            "-c",
            str(source_root / "native/rays.cu"),
            "-o",
            str(obj),
        ]
        subprocess.run(cuda_command, check=True)
        command += ["-DRELIFT_WITH_CUDA", str(obj), "-lcudart"]
    subprocess.run(command, check=True)
    record = dict(
        command=command,
        compiler=subprocess.check_output([args.cxx, "--version"], text=True),
        python=sys.version,
        torch=torch.__version__,
        platform=platform.platform(),
        openssl=subprocess.check_output(["openssl", "version"], text=True).strip(),
        sources={"native/cpu.cpp": hashlib.sha256(source.read_bytes()).hexdigest()},
        binary=binary.name,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        build_seconds=time.perf_counter() - start,
    )
    if args.cuda:
        record.update(
            cuda_command=cuda_command,
            nvcc=subprocess.check_output([args.nvcc, "--version"], text=True),
            arch=args.arch,
        )
        record["sources"]["native/rays.cu"] = hashlib.sha256(
            (source_root / "native/rays.cu").read_bytes()
        ).hexdigest()
    (args.out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(args.out / "build.json")


if __name__ == "__main__":
    main()
