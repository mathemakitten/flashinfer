"""Replay the known failing full-model command with one diagnostic graph split."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-diagnostic', type=Path, required=True)
    p.add_argument('--mcore-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split-after', default='mid', help='executed block ordinal (1-based), or mid')
    p.add_argument('--timeout-seconds', type=int, default=1800)
    a = p.parse_args()
    if a.split_after != 'mid' and (not a.split_after.isdigit() or int(a.split_after) < 1):
        p.error('split-after must be mid or a positive block ordinal')
    if a.timeout_seconds <= 0:
        p.error('timeout must be positive')
    if '_diagnostic_graph_factory' not in (a.mcore_root / 'megatron/core/transformer/cuda_graphs.py').read_text():
        p.error('sync the MCore diagnostic graph factory hook first')
    source = json.loads(a.source_diagnostic.read_text())
    command = json.loads(Path(source['source']).read_text())
    checkout = Path(__file__).resolve().parents[2]
    command[0] = sys.executable
    command[command.index('--nproc-per-node=4') + 1] = str(checkout / 'examples/moe_ep/nt4_checkpoint_generate.py')
    if command[command.index('--num-tokens-to-generate') + 1] != '512':
        p.error('expected the saved 512-output-token failure command')
    env = {k:v for k,v in os.environ.items() if not k.startswith('NT4_')}
    env.update(source['env'])
    env.update(NT4_MODEL_GRAPH_SPLIT_AFTER=a.split_after, NT4_MODEL_SYNC_BOUNDARIES='1',
               NT4_HYBRID_MCORE_ROOT=str(a.mcore_root), NT4_E2E_OUTPUT=str(a.output / 'hybrid'),
               PYTHONPATH=str(checkout) + os.pathsep + str(a.mcore_root))
    env.pop('CUDA_LAUNCH_BLOCKING', None)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'command.json').write_text(json.dumps(command, indent=2))
    (a.output / 'diagnostic.json').write_text(json.dumps(dict(
        source=str(a.source_diagnostic), split_after=a.split_after, performance_use=False,
        env={k:v for k,v in env.items() if k.startswith('NT4_')},
    ), indent=2))
    print(f'Full-model graph bisection: {a.output}', flush=True)
    with (a.output / 'hybrid.log').open('w') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=a.timeout_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            rc = 124
    (a.output / 'exit.json').write_text(json.dumps(dict(returncode=rc)))
    print(f'Diagnostic exit: {rc}', flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
