#!/usr/bin/env python
"""Stage this experiment in the existing isolated PR clone; never commit or push."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/pr-stack'
BASE='codex/10-cuda-ray-profiling'
BRANCH='codex/11-observation-cache'
FILES=['.gitattributes','relift/observation_cache.py','relift/optimization.py','relift/streaming.py',
       'scripts/experiment_worker.py','scripts/pipeline_experiment.py','scripts/profile_observation_cache.py',
       'scripts/publish_observation_cache.py','scripts/prepare_observation_cache_pr.py',
       'configs/experiments/observation-cache.json','configs/experiments/observation-cache-roles.json',
       'configs/experiments/observation-cache-revisions.json','configs/optimizations/observation-cache.json',
       'tests/test_observation_cache.py','tests/gpu/test_observation_cache.py',
       'docs/native-optimization-experiments/05-observation-cache.md',
       'docs/native-optimization-experiments/README.md','docs/optimization-configuration.md']
MEASURED=['relift/observation_cache.py','relift/optimization.py','relift/streaming.py',
          'scripts/experiment_worker.py','scripts/pipeline_experiment.py',
          'tests/test_observation_cache.py','tests/gpu/test_observation_cache.py',
          'configs/optimizations/observation-cache.json']


def git(*args):return subprocess.check_output(['git','-C',str(DEST),*args],text=True).strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['measured','final'])
    args=parser.parse_args()
    expected=BASE if args.phase=='measured' else BRANCH
    if git('branch','--show-current')!=expected or git('status','--porcelain'):
        raise ValueError(f'Expected clean isolated branch {expected}')
    source=ROOT/'outputs/experiments/observation-cache-source' if args.phase=='measured' else ROOT
    files=MEASURED if args.phase=='measured' else FILES+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'docs/benchmarks/observation-cache').rglob('*')) if p.is_file()]
    for filename in files:
        if not (source/filename).is_file():raise FileNotFoundError(filename)
    if args.phase=='measured':git('checkout','-b',BRANCH)
    for filename in files:
        target=DEST/filename;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/filename,target)
    subprocess.run([sys.executable,str(DEST/'scripts/build_architecture_view.py')],check=True)
    git('add',*files,'docs/architecture-view.html')
    print(git('diff','--cached','--stat'))


if __name__=='__main__':main()
