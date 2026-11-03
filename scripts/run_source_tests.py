#!/usr/bin/env python
"""Run tests with imports pinned to a selected source, overriding editable installs."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--native-build",type=Path)
    parser.add_argument("--target",action="append",required=True)
    args=parser.parse_args()
    source=args.source.resolve()
    sys.path.insert(0,str(source))
    import relift
    if Path(relift.__file__).resolve().parents[1] != source:
        raise RuntimeError("Tests imported the wrong ReLift source")
    if args.native_build: os.environ["RELIFT_NATIVE_BUILD"]=str(args.native_build.resolve())
    import pytest
    raise SystemExit(pytest.main([*(str(source/name) for name in args.target),"--import-mode=importlib","-q"]))


if __name__ == "__main__": main()
