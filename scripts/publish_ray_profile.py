#!/usr/bin/env python
"""Publish technical ray-profile records and component charts, without analysis.

Narrative belongs in docs/native-optimization-experiments/cuda-ray-profile.md.
The large raw traces, pstats files and trajectory stores remain in the campaign.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def portable(value):
    if isinstance(value,dict): return {key:portable(item) for key,item in value.items()}
    if isinstance(value,list): return [portable(item) for item in value]
    if isinstance(value,str): return value.replace(str(ROOT)+'/', '')
    return value


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(portable(value),indent=2)+'\n')


def charts(profile,startup,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(17,5.5),layout='constrained')
    names=['current','cuda','cuda_timeline']
    labels=['Current PyTorch','Corrected CUDA','Eager timeline CUDA']
    groups=[]
    for cache in [0,256]:
        groups.append((f'Warm mapped rays · {cache} MiB depth budget',labels,
                       [[s['wall_ms'] for s in profile['results'][f'mapped_cache{cache}_{name}']['samples']] for name in names]))
    names=['torch','cuda','timeline_registered','timeline']
    labels=['Current PyTorch','Corrected CUDA','Registered timeline','Eager timeline CUDA']
    groups.append(('First resident builder call · fresh processes',labels,
                   [[r['first_call_seconds']*1000 for r in startup['records'] if r['backend']==name] for name in names]))
    for axis,(title,labels,samples) in zip(axes,groups):
        medians=[statistics.median(values) for values in samples]
        low=[m-min(v) for m,v in zip(medians,samples)]
        high=[max(v)-m for m,v in zip(medians,samples)]
        axis.barh(labels,medians,color=['#42769b']*(len(labels)-1)+['#278574'])
        axis.errorbar(medians,range(len(labels)),xerr=[low,high],fmt='none',color='#263746',capsize=3)
        for y,(m,h,v) in enumerate(zip(medians,high,samples)):
            axis.text(m+h,y,f' {m:.3f} (n={len(v)})',va='center',fontsize=9)
        axis.set_xlim(0,max(max(v) for v in samples)*1.35)
        axis.invert_yaxis();axis.set_xlabel('milliseconds');axis.set_title(title,fontsize=11)
        axis.grid(axis='x',alpha=.2)
    figure.suptitle('Recorded ray component timings',fontsize=15)
    figure.supxlabel('Medians and observed min–max. Warm rays include mapped gathering/transfers; first-call scope uses resident inputs.\nCPU/CUDA profiling traces are collected separately from these timings.',fontsize=10)
    figure.savefig(out/'components.png',dpi=150)
    figure.savefig(out/'components.svg')
    svg=out/'components.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--startup',type=Path,required=True)
    parser.add_argument('--audit',type=Path,action='append',required=True)
    parser.add_argument('--ablation-profile',type=Path)
    parser.add_argument('--ablation-startup',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    sources={}
    for name,path in [('profile',args.profile),('startup',args.startup),
                      ('ablation-profile',args.ablation_profile),('ablation-startup',args.ablation_startup)]:
        if path:
            save(args.out/(name+'.json'),json.loads(path.read_text()))
            sources[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for path in args.audit:
        save(args.out/path.name,json.loads(path.read_text()))
        sources[path.name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    records=[json.loads(path.read_text()) for path in sorted(args.campaign.rglob('trial.json'))]
    save(args.out/'trials.json',records)
    quality={str(path.relative_to(args.campaign)):json.loads(path.read_text()) for path in sorted(args.campaign.rglob('quality.json'))}
    save(args.out/'quality.json',quality)
    from scripts.summarize_optimization_experiments import summarize
    save(args.out/'summary.json',summarize(args.campaign))
    diagnostics={}
    for path in sorted(args.campaign.rglob('diagnostic.json')):
        value=json.loads(path.read_text())
        diagnostics[str(path.relative_to(args.campaign))]=dict(spans=value['spans'],caveat=value['caveat'],
            functions=value['functions'][:100],function_selection='Top 100 by self_seconds; full pstats retained locally')
    save(args.out/'diagnostics.json',diagnostics)
    save(args.out/'provenance.json',dict(sources=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    charts(json.loads(args.profile.read_text()),json.loads(args.startup.read_text()),args.out)
    print(args.out)


if __name__=='__main__': main()
