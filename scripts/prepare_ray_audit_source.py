#!/usr/bin/env python
"""Overlay only audited ray changes onto the recorded pre-fix experiment source.

This preserves later unpublished experiments in the working tree without including
them in the ray PR or its rerun. Existing snapshots are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
FILES=("native/rays.cu","relift/cuda_rays.py","tests/gpu/test_cuda_rays.py",
       "tests/gpu/test_ray_arithmetic.py",
       "scripts/build_native.py","scripts/audit_cuda_rays.py","scripts/prepare_ray_audit_source.py")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    shutil.copytree(args.baseline,args.out)
    for name in FILES:
        target=args.out/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    cpu=args.out/"native/cpu.cpp"
    text=cpu.read_text()
    declaration="void launch_rays(uintptr_t,uintptr_t,uintptr_t,uintptr_t,uintptr_t,uintptr_t,int,int,int,int,uintptr_t);"
    binding='    module.def("launch_rays", &launch_rays);'
    if "launch_camera_rays" not in text:
        if declaration not in text or binding not in text:
            raise ValueError("Unexpected baseline CUDA binding")
        text=text.replace(declaration,declaration+"\nvoid launch_camera_rays(uintptr_t,uintptr_t,uintptr_t,uintptr_t,uintptr_t,int,int,int,int,uintptr_t);")
        text=text.replace(binding,binding+'\n    module.def("launch_camera_rays", &launch_camera_rays);')
        cpu.write_text(text)
    manifest=json.loads((args.baseline/"snapshot.json").read_text())
    manifest["parent_snapshot_sha256"]=hashlib.sha256((args.baseline/"snapshot.json").read_bytes()).hexdigest()
    manifest["overlay_files"]=list(FILES)
    manifest["files"]={str(p.relative_to(args.out)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(args.out.rglob("*")) if p.is_file() and p.name!="snapshot.json" and "__pycache__" not in p.parts}
    (args.out/"snapshot.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(args.out)


if __name__ == "__main__": main()
