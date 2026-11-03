"""Replay the accuracy batch with all-layer activation changes; diagnostic only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mcore-root',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    checkout=Path(__file__).resolve().parents[2]
    env={k:v for k,v in os.environ.items() if not k.startswith('NT4_')}
    env.update(PYTHONPATH=f'{checkout}:{a.mcore_root}',NT4_ALLOW_SOURCE_JIT='1',MAX_JOBS='4',
        NT4_DETACHED_TAIL='1',NT4_LOOKAHEAD_AFTER_MMA='0',NT4_PACKED_FC1_ACTIVATION='0',
        NT4_E2E_MODE='direct',NT4_E2E_PRECISION='mxfp8',NT4_E2E_MODEL_GRAPHS='1',
        NT4_E2E_ROUTED_GRAPHS='0',NT4_SHORTCUT_OVERLAP='1',NT4_MXFP8_NVLS_HYBRID='1',
        NT4_HYBRID_MCORE_ROOT=str(a.mcore_root),NT4_E2E_REQUEST_COUNT='128',
        NT4_SCHED_LOOKUP='1',NT4_MXFP8_GROUP_HINT='1024',NT4_TE_STREAM_WORKSPACES='1',NT4_TANH_LOGIT_SOURCE=str(a.source),
        NT4_CONTROLLED_GC='1')
    for arm in ('control','hardware'):
        env.update(NT4_HARDWARE_TANH=str(int(arm=='hardware')),NT4_TANH_FC1_ADVISORY=str(int(arm=='hardware')),
                   NT4_E2E_OUTPUT=str(a.output/arm), NT4_TANH_EXPERT_ADVISORY=str(int(arm=='hardware')), NT4_TANH_OUTPUT_ADVISORY=str(int(arm=='hardware')))
        command=json.loads((a.source/'hybrid-hardware-tanh-command.json').read_text())
        command[0]=sys.executable
        command[5]=str(checkout/'examples/moe_ep/nt4_checkpoint_generate.py')
        (a.output/f'{arm}-command.json').write_text(json.dumps(command,indent=2))
        print(json.dumps(dict(stage='logit_arm_start',arm=arm,log=str(a.output/f'{arm}.log'))),flush=True)
        with (a.output/f'{arm}.log').open('w') as f:
            process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise RuntimeError(f'{arm} timed out')
        if code:raise RuntimeError(f'{arm} failed ({code}); inspect {a.output/arm}')
    import torch
    control=torch.load(a.output/'control/logits-round1.pt',weights_only=True)
    candidate=torch.load(a.output/'hardware/logits-round1.pt',weights_only=True)
    result=[]
    for rid,left in control.items():
        right=candidate[rid]
        same=all(left[k]==right[k] for k in ('prompt_token_ids','common_generated_prefix_ids','schedule_sha256_at_capture'))
        if not same:raise AssertionError('Prefix or batch schedule changed; do not attribute logits')
        l,r=left['logits'],right['logits'];li,ri=int(l.argmax()),int(r.argmax())
        result.append(dict(request=rid,position=left['generated_position'],same_prefix_and_schedule=True,
            control_top_id=li,hardware_top_id=ri,control_margin=float(l.topk(2).values.diff().neg()[0]),
            hardware_margin=float(r.topk(2).values.diff().neg()[0]),
            control_logits_for_pair=[float(l[li]),float(l[ri])],hardware_logits_for_pair=[float(r[li]),float(r[ri])],
            logits_relative_l2=float((r.double()-l.double()).norm()/l.double().norm()),
            max_abs=float((r-l).abs().max()),control_top_tokens=left['top_tokens'],hardware_top_tokens=right['top_tokens']))
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(stage='logit_pair_complete',comparison=str(a.output/'comparison.json'))),flush=True)


if __name__=='__main__':main()
