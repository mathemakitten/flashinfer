"""Bounded standalone lookup tuning; no kernel edits, Slurm queries, or polling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track',choices=('workers','groups'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-epoch',type=int,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    base=Path('/lustre/fsw/portfolios/nemotron/users/helenn')
    fi=base/'flashinfer-nt4-gb300-nvls'
    env=os.environ.copy()
    for key in ('NT4_EXPERT_STAGE_CAPTURE','NT4_EXPERT_INTERNAL_TIMING',
                'NT4_EXPERT_PHASE_TIMING','NT4_EXPERT_SPARSE_TIMING',
                'NT4_PIPELINE_REDUCE_SIDE','NT4_FC2_PROBE_AFTER_ISSUE',
                'NT4_LOOKAHEAD_AFTER_MMA','NT4_FC1_DIRECT_STORE','NT4_KERNEL_DUMP_DIR',
                'NT4_HARDWARE_TANH','NT4_TANH_FC1_ADVISORY','NT4_SCHED_LOOKUP'):
        env.pop(key,None)
    env.update(FLASHINFER_CHECKOUT=str(fi),FLASHINFER_WORKSPACE_BASE=str(base),
               PYTHONPATH=f'{fi}:{base/"megatron-lm-nt4-coscheduling"}',
               NT4_ALLOW_SOURCE_JIT='1',MAX_JOBS='4')
    if a.track=='workers':
        plan=[(f'rank{rank}-{index}-w{workers}',rank,workers,'1024')
              for rank in range(4) for index,workers in enumerate((152,96,128,144,152))]
    else:
        plan=[(f'rank{rank}-{order}',rank,152,groups) for rank in range(4)
              for order,groups in [('forward','128,256,512,1024,2048'),
                                   ('reverse','2048,1024,512,256,128')]]
    (a.output/'plan.json').write_text(json.dumps(dict(plan=plan,
        activation='original exponential tanh, matching current E2E',
        scope='isolated fused FC1/FC2; no mixer/NVLS; not E2E',
        numerical_policy='exact FC1 bytes/scales and FC2 versus original; poisoned graph checks'),indent=2))
    results=[]
    for label,rank,workers,groups in plan:
        remaining=int(a.deadline_epoch-time.time()-120)
        if remaining<300:
            print(json.dumps(dict(stage='deadline_stop',completed=len(results),planned=len(plan))),flush=True)
            break
        cmd=[sys.executable,'-u',str(fi/'examples/moe_ep/nt4_lookup_tune_replay.py'),
             '--artifact',str(base/f'logs/nt4-expert-stage32-20261029-144918/rank{rank}'),
             '--output',str(a.output/label),'--fc12-lookup','--workers',str(workers),
             '--lookup-groups',groups]
        print(json.dumps(dict(stage='trial_start',label=label,log=str(a.output/f'{label}.log'))),flush=True)
        with (a.output/f'{label}.log').open('w') as log:
            try:
                result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(1200,remaining))
                code=result.returncode
            except subprocess.TimeoutExpired:
                code=124
        item=dict(label=label,returncode=code)
        if code==0:
            data=json.loads((a.output/label/'summary.json').read_text())
            assert data['status']=='PASS_FC12_CONFIRM'
            item['timings']={k:v for k,v in data['timings'].items() if 'group' in k}
        results.append(item)
        (a.output/'queue-results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(item),flush=True)
        if code:raise SystemExit(code)
    print(json.dumps(dict(stage='queue_complete',completed=len(results),planned=len(plan))),flush=True)


if __name__=='__main__':main()
