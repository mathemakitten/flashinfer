"""Replay identical prefixes with complementary layer activation substitutions."""
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
    layers=[7,9,11,14,16,18,20,23,25,27,29,32,34,36,38,41,43,45,47,50,52,54,56,59,61,63,66,68]
    arms={'control':[], 'quarter1':layers[:7], 'quarter2':layers[7:14], 'hardware':layers}
    (a.output/'layer-selections.json').write_text(json.dumps(arms,indent=2))
    for arm,selected in arms.items():
        env['NT4_HARDWARE_TANH_LAYERS']=','.join(map(str,selected))
        (a.output/f'{arm}-selection.json').write_text(json.dumps(dict(layers=selected),indent=2))
        env.update(NT4_HARDWARE_TANH=str(int(arm!='control')),NT4_TANH_FC1_ADVISORY=str(int(arm!='control')),
                   NT4_E2E_OUTPUT=str(a.output/arm), NT4_TANH_EXPERT_ADVISORY=str(int(arm!='control')), NT4_TANH_OUTPUT_ADVISORY=str(int(arm!='control')))
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
    full=torch.load(a.output/'hardware/logits-round1.pt',weights_only=True)
    result=[]
    for arm in arms:
        checks=json.loads((a.output/arm/'logit-replay-check.json').read_text())
        if not all(v['eager_graph_exact'] for v in checks.values()):
            raise AssertionError('Eager/graph disagreement: '+arm)
        candidate=torch.load(a.output/arm/'logits-round1.pt',weights_only=True)
        for rid,left in control.items():
            right=candidate[rid]
            same=all(left[k]==right[k] for k in ('prompt_token_ids','common_generated_prefix_ids','schedule_sha256_at_capture'))
            if not same:raise AssertionError('Prefix or batch schedule changed')
            l,r=left['logits'],right['logits']
            li,fi,ri=int(l.argmax()),int(full[rid]['logits'].argmax()),int(r.argmax())
            result.append(dict(arm=arm,layers=arms[arm],request=rid,position=left['generated_position'],
                same_prefix_and_schedule=True,eager_graph_exact=True,
                control_top_id=li,full_hardware_top_id=fi,candidate_top_id=ri,
                candidate_top_tokens=right['top_tokens'],
                fixed_pair_logits=[float(r[li]),float(r[fi])],
                fixed_pair_margin=float(r[li]-r[fi]),
                candidate_top_margin=float(r.topk(2).values[0]-r.topk(2).values[1]),
                logits_relative_l2=float((r.double()-l.double()).norm()/l.double().norm()),
                max_abs=float((r-l).abs().max())))
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(stage='layer_ablation_complete',comparison=str(a.output/'comparison.json'))),flush=True)


if __name__=='__main__':main()
