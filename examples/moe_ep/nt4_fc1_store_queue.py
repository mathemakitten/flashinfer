"""Bounded FC1 store-path research queue; no Slurm queries or polling.

Native CUTLASS output continues after each measured branch. These are matched
block diagnostics, not candidate-driven E2E throughput or deployment acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def qualified(root):
    for rank in range(4):
        d = json.loads((root / f'rank{rank}/summary.json').read_text())
        if not (d.get('status') == 'PASS_FC12_CONFIRM'
                and d.get('fc1_store_sweep') and d.get('poison_intermediates')):
            raise RuntimeError(f'FC1 store qualification missing for rank{rank}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline-epoch', type=int, required=True)
    p.add_argument('--workers', type=int, choices=(128, 152), required=True)
    a = p.parse_args()
    qualified(a.qualification)
    a.output.mkdir(parents=True, exist_ok=False)
    base = Path('/lustre/fsw/portfolios/nemotron/users/helenn')
    fi = base / 'flashinfer-nt4-gb300-nvls'
    mc = base / 'megatron-lm-nt4-coscheduling'
    env = os.environ.copy()
    for name in ('NT4_HARDWARE_TANH_LAYERS', 'NT4_TANH_LOGIT_SOURCE',
                 'NT4_FC1_FAILURE_DIR', 'NT4_EXPERT_FAILURE_DIR', 'NT4_KERNEL_DUMP_DIR',
                 'NT4_TRT_DEBUG_INTERMEDIATES', 'CUDA_LAUNCH_BLOCKING', 'MEGA_USE_NCU',
                 'NT4_MODEL_GRAPH_SPLIT_AFTER', 'NT4_MODEL_SYNC_BOUNDARIES',
                 'NT4_CUTLASS_MAX_BLOCKS', 'NT4_MXFP8_AB_STAGE_REDUCTION',
                 'NT4_EXPERT_STAGE_CAPTURE', 'NT4_CUBIN_TRACE'):
        env.pop(name, None)
    env.update(FLASHINFER_CHECKOUT=str(fi), MCORE_CHECKOUT=str(mc),
               NT4_LOG_DIR=str(base/'logs'), FLASHINFER_WORKSPACE_BASE=str(base),
               PYTHONPATH=f'{fi}:{mc}', MAX_JOBS='4', NT4_ALLOW_SOURCE_JIT='1',
               FLASHINFER_TRTLLM_NT4_EXPORT=str(base/'logs/nt4-trtllm-clamped-sweep-20261020-134327/export'),
               NT4_DETACHED_TAIL='1', NT4_LOOKAHEAD_AFTER_MMA='0',
               NT4_PACKED_FC1_ACTIVATION='0', NT4_BRANCH_SAMPLES='16',
               NT4_BRANCH_CLOCK='device', NT4_BRANCH_TAIL='1', NT4_EXPERT_PHASE_TIMING='1',
               NT4_EXPERT_INTERNAL_TIMING='0', NT4_HARDWARE_TANH='1',
               NT4_TANH_FC1_ADVISORY='1', NT4_MXFP8_GROUP_HINT='1024')
    plan = []
    # Opposite initial schedule order on the two nodes. Every comparison is
    # bracketed on its own node; cross-node timing differences are not gains.
    schedules = ('serial','overlap') if a.workers == 128 else ('overlap','serial')
    for workers in (a.workers, 280-a.workers):
        for schedule in schedules:
            for arm in ('tma-start','stg','tma-end'):
                plan.append((workers, schedule, arm, False))
    # Current-hardware-tanh turnover attribution after the first worker's
    # bracketed comparisons, before the optional worker-count crossover.
    plan[6:6] = [(a.workers, 'serial', arm, True) for arm in ('tma','stg')]
    results = []
    (a.output/'plan.json').write_text(json.dumps(dict(
        scope=__doc__, qualification=str(a.qualification), deadline_epoch=a.deadline_epoch,
        plan=plan, numerical_policy='Existing hardware-tanh FC1 advisory retained; not deployment qualified'), indent=2))
    for workers, schedule, arm, internal in plan:
        remaining = int(a.deadline_epoch - time.time() - 120)
        if remaining < 600:
            print(json.dumps(dict(stage='queue_deadline_stop', remaining_seconds=remaining)), flush=True)
            break
        label = f'w{workers}-{schedule}-{arm}' + ('-internal' if internal else '')
        env['NT4_FC1_DIRECT_STORE'] = '1' if arm == 'stg' else '0'
        env['NT4_EXPERT_INTERNAL_TIMING'] = '1' if internal else '0'
        env['NT4_MXFP8_WORKER_CLUSTERS'] = str(workers)
        cmd = [sys.executable, '-u', str(fi/'examples/moe_ep/nt4_e2e.py'),
               '--source-artifact', str(base/'logs/nt4-checkpoint-layer-20261012-080901'),
               '--mcore-root', str(mc), '--prompt-file', str(base/'research/routing/openmath2_unique2048.jsonl'),
               '--output', str(a.output/label), '--precision', 'mxfp8', '--request-count', '128',
               '--num-tokens-to-generate', '64', '--model-token-budget', '2048',
               '--model-graphs', '--shortcut-overlap', '--controlled-gc', '--mxfp8-nvls-hybrid',
               '--mxfp8-worker-clusters', str(workers), '--branch-native-output',
               '--branch-contention', schedule, '--branch-backend', 'direct',
               '--branch-input-mode', 'replay', '--branch-inputs', str(base/'logs/nt4-trtllm-matched128-20261023-112340/inputs'),
               '--rounds', '2', '--timeout-seconds', str(min(1800, remaining))]
        print(json.dumps(dict(stage='queue_arm_start', label=label, remaining_seconds=remaining)), flush=True)
        with (a.output/f'{label}.log').open('w') as log:
            # nt4_e2e owns child process groups and its timeout/cleanup.
            completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = dict(label=label, returncode=completed.returncode, summary=str(a.output/label/'summary.json'))
        results.append(result)
        (a.output/'queue-results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)
    print(json.dumps(dict(stage='queue_complete', completed=len(results), planned=len(plan))), flush=True)


if __name__ == '__main__':
    main()
