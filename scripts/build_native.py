#!/usr/bin/env python
"""Build optional CPU helpers outside inference and persist exact compiler provenance.

uv run python scripts/build_native.py --out outputs/native/gather
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite native build: {args.out}")
    args.out.mkdir(parents=True)
    source = ROOT / "native/cpu.cpp"
    binary = args.out.resolve() / ("_relift_cpu" + sysconfig.get_config_var("EXT_SUFFIX"))
    # pybind11 headers ship in the pinned torch wheel; no extra runtime dependency.
    command = [args.cxx, "-O3", "-std=c++17", "-shared", "-fPIC", "-fno-fast-math",
               "-I" + sysconfig.get_path("include"), "-I" + str(Path(torch.__file__).parent / "include"),
               str(source), "-o", str(binary), "-lcrypto"]
    start = time.perf_counter()
    subprocess.run(command, check=True)
    record = dict(command=command, compiler=subprocess.check_output([args.cxx, "--version"], text=True),
                  python=sys.version, torch=torch.__version__, platform=platform.platform(),
                  openssl=subprocess.check_output(["openssl","version"],text=True).strip(),
                  sources={"native/cpu.cpp": hashlib.sha256(source.read_bytes()).hexdigest()},
                  binary=binary.name, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  build_seconds=time.perf_counter()-start)
    (args.out / "build.json").write_text(json.dumps(record, indent=2)+"\n")
    print(args.out / "build.json")


if __name__ == "__main__":
    main()
