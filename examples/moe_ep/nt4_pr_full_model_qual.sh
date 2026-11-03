#!/usr/bin/env bash
set -euo pipefail

# Remove diagnostic overrides left by earlier experiments in this shell.
while IFS= read -r name; do unset "$name"; done < <(compgen -v NT4_)

export PY="${PY:-$(command -v python3)}"
export FLASHINFER_CHECKOUT=/lustre/fsw/portfolios/nemotron/users/helenn/flashinfer-nt4-megamoe-pr
export MCORE_CHECKOUT=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/helenn/megatron-lm
export NT4_LOG_DIR=/lustre/fsw/portfolios/nemotron/users/helenn/logs
export FLASHINFER_WORKSPACE_BASE=/lustre/fsw/portfolios/nemotron/users/helenn
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT"
export NT4_ALLOW_SOURCE_JIT=1
export MAX_JOBS=4 OMP_NUM_THREADS=1 FLASHINFER_JIT_DEBUG=0 FLASHINFER_JIT_VERBOSE=0
unset FLASHINFER_TRTLLM_NT4_EXPORT CUDA_LAUNCH_BLOCKING FLASHINFER_DISABLE_JIT

qual_root="$NT4_LOG_DIR/nt4-megamoe-pr-full-model-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$qual_root"
printf 'Qualification results: %s\n' "$qual_root"
"$PY" - <<'PY'
import os
from pathlib import Path
from packaging.version import Version
import flashinfer
expected = Path(os.environ['FLASHINFER_CHECKOUT']).resolve()
actual = Path(flashinfer.__file__).resolve()
assert actual.parent.parent == expected, (actual, expected)
assert Version(flashinfer.__version__) >= Version('0.6.4'), (
    'Generate checkout build metadata before running qualification',
    flashinfer.__version__,
)
print('FlashInfer library:', actual, 'version:', flashinfer.__version__, flush=True)
PY

harness="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/nt4_e2e.py"
for case in mxfp8-hybrid128 mxfp8-hybrid256 bf16-direct128; do
    extra=()
    precision=mxfp8
    requests=128
    if [[ "$case" == bf16-* ]]; then
        precision=bf16
    else
        extra+=(--mxfp8-nvls-hybrid --cutlass-decode-profile-ids 39 428)
    fi
    if [[ "$case" == *256 ]]; then requests=256; fi
    printf 'Starting %s; log=%s/%s.log\n' "$case" "$qual_root" "$case"
    "$PY" "$harness" \
        --flashinfer-checkout "$FLASHINFER_CHECKOUT" \
        --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
        --mcore-root "$MCORE_CHECKOUT" \
        --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
        --output "$qual_root/$case" \
        --precision "$precision" --request-count "$requests" \
        --model-token-budget 2048 --num-tokens-to-generate 64 \
        --model-graphs --shortcut-overlap --controlled-gc \
        --rounds 1 --timeout-seconds 1800 \
        "${extra[@]}" > "$qual_root/$case.log" 2>&1
done
printf 'Full-model qualification completed: %s\n' "$qual_root"
