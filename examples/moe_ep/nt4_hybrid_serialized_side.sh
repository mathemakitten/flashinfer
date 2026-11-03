#!/usr/bin/env bash
# Run only the previously failing hybrid arm, retaining a distinct side stream.
set -euo pipefail
: "${PY:?Set PY}"
: "${FLASHINFER_CHECKOUT:?Set FLASHINFER_CHECKOUT}"
: "${MCORE_CHECKOUT:?Set MCORE_CHECKOUT}"
: "${NT4_LOG_DIR:?Set NT4_LOG_DIR}"
export NT4_MODEL_SYNC_BOUNDARIES=0
export NT4_HYBRID_MEMCHECK=0
case "${1:-control}" in
  control) export NT4_MODEL_PRIVATE_CAPTURE_POOLS=0 ;;
  private-pool) export NT4_MODEL_PRIVATE_CAPTURE_POOLS=1 ;;
  sync-boundaries) export NT4_MODEL_PRIVATE_CAPTURE_POOLS=1 NT4_MODEL_SYNC_BOUNDARIES=1 ;;
  memcheck) export NT4_MODEL_PRIVATE_CAPTURE_POOLS=1 NT4_MODEL_SYNC_BOUNDARIES=1 NT4_HYBRID_MEMCHECK=1 ;;
  *) echo "Usage: $0 [control|private-pool|sync-boundaries|memcheck]" >&2; exit 2 ;;
esac
export NT4_SERIALIZED_OUTPUT="${NT4_SERIALIZED_OUTPUT:-$NT4_LOG_DIR/nt4-hybrid-serialized-side32-${1:-control}-$(date +%Y%m%d-%H%M%S)}"
"$PY" - <<'PY'
import json, os, pathlib, signal, subprocess, sys, shutil
root = pathlib.Path(os.environ['NT4_SERIALIZED_OUTPUT'])
memcheck = os.environ['NT4_HYBRID_MEMCHECK'] == '1'
sanitizer = shutil.which('compute-sanitizer') if memcheck else None
if memcheck and sanitizer is None:
    raise SystemExit('compute-sanitizer is not on PATH; no model process launched')
timeout = 7200 if memcheck else 1200
base = pathlib.Path(os.environ['NT4_LOG_DIR'])
source = base / 'nt4-hybrid-staged-single-stream32-20261021-121251/overlap-command.json'
command = json.loads(source.read_text())
command[0] = sys.executable
script = command.index('--nproc-per-node=4') + 1
command[script] = str(pathlib.Path(os.environ['FLASHINFER_CHECKOUT']) / 'examples/moe_ep/nt4_checkpoint_generate.py')
assert '--inference-shortcut-moe-overlap' in command
command[command.index('--num-tokens-to-generate') + 1] = '16'
env = {k: v for k, v in os.environ.items() if not k.startswith('NT4_')}
env.update(
    NT4_MODEL_SYNC_BOUNDARIES=os.environ['NT4_MODEL_SYNC_BOUNDARIES'],
    NT4_MODEL_PRIVATE_CAPTURE_POOLS=os.environ['NT4_MODEL_PRIVATE_CAPTURE_POOLS'],
    NT4_ALLOW_SOURCE_JIT='1', NT4_SERIALIZE_SHORTCUT_SIDE_STREAM='1',
    NT4_SHORTCUT_OVERLAP='1', NT4_TE_STREAM_WORKSPACES='1',
    NT4_E2E_MODE='direct', NT4_E2E_PRECISION='mxfp8',
    NT4_E2E_REQUEST_COUNT='32', NT4_E2E_ROUNDS='1', NT4_CONTROLLED_GC='1',
    NT4_E2E_MODEL_GRAPHS='1', NT4_E2E_ROUTED_GRAPHS='0',
    NT4_EAGER_RANK_AUDIT='1', NT4_MXFP8_NVLS_HYBRID='1',
    NT4_CUTLASS_DECODE_PROFILE_IDS='[39,428]',
    NT4_TRTLLM_CLAMPED='0', NT4_TRTLLM_RELU2_SURROGATE='0',
    NT4_HYBRID_MCORE_ROOT=os.environ['MCORE_CHECKOUT'],
    NT4_E2E_OUTPUT=str(root / 'overlap'),
)
env.pop('CUDA_LAUNCH_BLOCKING', None)
env['PYTHONPATH'] = os.pathsep.join((env['FLASHINFER_CHECKOUT'], env['MCORE_CHECKOUT']))
root.mkdir(parents=True, exist_ok=False)
if memcheck:
    command = [sanitizer, '--tool', 'memcheck', '--target-processes', 'all',
               '--error-exitcode', '86', '--log-file', str(root / 'memcheck-%p.log')] + command
(root / 'command.json').write_text(json.dumps(command, indent=2))
(root / 'diagnostic.json').write_text(json.dumps(dict(
    source=str(source), performance_use=False, memcheck=memcheck, timeout_seconds=timeout,
    settings={k:v for k,v in env.items() if k.startswith('NT4_')},
), indent=2))
print(f'Serialized side-stream diagnostic: {root}', flush=True)
with (root / 'overlap.log').open('w') as log:
    process = subprocess.Popen(command, env=env, cwd=env['FLASHINFER_CHECKOUT'],
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
(root / 'exit.json').write_text(json.dumps(dict(returncode=code)))
print(f'Diagnostic exit: {code}; log: {root / "overlap.log"}', flush=True)
sys.exit(code)
PY
