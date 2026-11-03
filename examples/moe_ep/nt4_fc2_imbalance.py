"""Synthetic all-ready FC2 routing stress; fixed contributions and tile count.

Runs the existing numerical-gated replay unchanged. This does not emulate
FC1 readiness skew or measure individual worker completion times.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import torch

p=argparse.ArgumentParser()
p.add_argument('--routing-pattern',choices=['balanced','skewed'],required=True)
a,rest=p.parse_known_args()
original_load=torch.load

def load(path,*args,**kwargs):
    saved=original_load(path,*args,**kwargs)
    if Path(path).name!='inputs.pt' or not isinstance(saved,dict) or saved.get('schema')!='nt4_expert_stage_v1':
        return saved
    ids=saved['ids'].clone()
    rank=saved['rank']
    mask=(ids>=rank*128)&(ids<(rank+1)*128)
    total=int(mask.sum())
    experts=32
    assert experts<=total<=experts*128, total
    if a.routing_pattern=='balanced':
        target=[total//experts+(i<total%experts) for i in range(experts)]
    else:
        target=[1]*experts
        left=total-experts
        for i in range(experts):
            add=min(127,left);target[i]+=add;left-=add
        assert left==0
    remaining=target.copy()
    # Bipartite degree realization: process rows in descending local degree,
    # choose distinct experts with the largest remaining column degree.
    degrees=mask.sum(1).tolist()
    for row in sorted(range(len(degrees)),key=lambda r:(-degrees[r],r)):
        slots=torch.where(mask[row])[0].tolist()
        chosen=sorted(range(experts),key=lambda e:(-remaining[e],e))[:len(slots)]
        assert all(remaining[e]>0 for e in chosen), (row,remaining)
        for slot,e in zip(slots,chosen):
            ids[row,slot]=rank*128+e;remaining[e]-=1
    assert not any(remaining)
    assert torch.equal(mask,(ids>=rank*128)&(ids<(rank+1)*128))
    for row in ids:
        valid=row[row>=0];assert valid.unique().numel()==valid.numel()
    saved=dict(saved,ids=ids)
    report=dict(pattern=a.routing_pattern,rank=rank,counts=target,
                routed_contributions=total,fc2_m128_tiles=experts,
                fc2_n128_tiles=experts*6,
                caveat='Fixed useful rows and padded MMA tiles; routing changes memory access and output values. Not readiness-delay or E2E test.')
    out=Path(rest[rest.index('--output')+1]);out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.routing.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(stage='synthetic_routing',**report)),flush=True)
    return saved

torch.load=load
script=Path(__file__).with_name('nt4_expert_stage_replay.py')
sys.argv=[str(script),*rest]
runpy.run_path(str(script),run_name='__main__')
