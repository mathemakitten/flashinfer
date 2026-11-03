"""Bracket the 768 hardware-tanh and isolated TRT comparison with exponential controls."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

base = Path('/lustre/fsw/portfolios/nemotron/users/helenn')
fi = base / 'flashinfer-nt4-gb300-nvls'
mcore = base / 'megatron-lm-nt4-coscheduling'
source = base / 'logs/nt4-tanh-trt768-trt-first-20261102-071603'
root = base / ('logs/nt4-tanh-trt768-matched-isolated-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
root.mkdir()
print('Results:', root, flush=True)
env = {k:v for k,v in os.environ.items() if not k.startswith('NT4_')}
env.update(PYTHONPATH=f'{fi}:{mcore}', FLASHINFER_WORKSPACE_BASE=str(base),
    FLASHINFER_TRTLLM_NT4_EXPORT=str(base/'logs/nt4-trtllm-clamped-sweep-20261020-134327/export'), MAX_JOBS='4')
for k,v in dict(ALLOW_SOURCE_JIT=1, DETACHED_TAIL=1, TE_STREAM_WORKSPACES=1,
    LOOKAHEAD_AFTER_MMA=0, PACKED_FC1_ACTIVATION=0, MXFP8_GROUP_HINT=1024,
    SCHED_LOOKUP=1, E2E_MODE='direct', E2E_PRECISION='mxfp8', E2E_REQUEST_COUNT=768,
    E2E_ROUNDS=2, CONTROLLED_GC=1, SHORTCUT_OVERLAP=1, E2E_MODEL_GRAPHS=1,
    E2E_ROUTED_GRAPHS=0, MXFP8_NVLS_HYBRID=1, HYBRID_MCORE_ROOT=str(mcore),
    HARDWARE_TANH=0, TANH_FC1_ADVISORY=0, TANH_EXPERT_ADVISORY=0,
    TANH_OUTPUT_ADVISORY=0).items(): env['NT4_'+k]=str(v)
for key in ('CUDA_LAUNCH_BLOCKING','MEGA_USE_NCU'): env.pop(key,None)
records=[]

def run(name, trt, isolate, capture=False, hardware=False):
    output=root/name
    output.mkdir()
    arm='trtllm-clamped' if trt else 'hybrid-control-start'
    command=json.loads((source/(arm+'-command.json')).read_text())
    command[0]=sys.executable
    command[5]=str(fi/'examples/moe_ep/nt4_trt_validation_isolation.py')
    current=env.copy()
    current.update(NT4_E2E_OUTPUT=str(output), NT4_TRTLLM_CLAMPED=str(int(trt)),
        NT4_TRT_EXPLORATORY_TIMING=str(int(trt)), NT4_VALIDATION_ISOLATE_TRT=str(int(isolate)))
    for key in ('HARDWARE_TANH','TANH_FC1_ADVISORY','TANH_EXPERT_ADVISORY','TANH_OUTPUT_ADVISORY'):
        current['NT4_'+key]=str(int(hardware))
    if capture:
        current.update(NT4_FC1_FAILURE_DIR=str(output/'fc1-failure'),
                       NT4_EXPERT_FAILURE_DIR=str(output/'expert-failure'))
    (output/'command.json').write_text(json.dumps(command,indent=2))
    (output/'experiment-env.json').write_text(json.dumps({k:v for k,v in current.items() if k.startswith(('NT4_','FLASHINFER_')) or k=='PYTHONPATH'},indent=2))
    log=root/(name+'.log')
    print('Starting',name,'log=',log,flush=True)
    with log.open('w') as stream:
        process=subprocess.Popen(command,env=current,cwd=mcore,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try: rc=process.wait(timeout=2400)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            rc=124
    record=dict(arm=name,returncode=rc,log=str(log),results=str(output/'results.json'))
    records.append(record)
    (root/'status.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(record),flush=True)
    if capture:
        content=log.read_text()
        snapshots=list((output/'fc1-failure').glob('layer*/*.pt'))
        reproduced=rc!=0 and 'FC1 intermediate failed independent block-variant/boundary validation' in content and bool(snapshots)
        (output/'capture-status.json').write_text(json.dumps(dict(reproduced=reproduced,snapshots=[str(p) for p in snapshots]),indent=2))
        if not reproduced: raise RuntimeError('Original failure not captured; inspect log before proceeding')
    elif rc or not (output/'results.json').exists():
        raise RuntimeError('Arm did not complete; inspect '+str(log))

run('megamoe-control-start',False,False)
run('megamoe-hardware-tanh',False,False,hardware=True)
run('trtllm-isolated',True,True)
run('megamoe-control-end',False,False)
print('Completed:',root,flush=True)
