#!/usr/bin/env bash
# Matched block-input diagnostic. No E2E throughput conclusions from these timings.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}" "${FLASHINFER_TRTLLM_NT4_EXPORT:?}"
root=${NT4_COSCHED_ROOT:-$NT4_LOG_DIR/nt4-coscheduling-headroom-$(date +%Y%m%d-%H%M%S)}
artifact=${NT4_ARTIFACT:-$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901}
source128=${NT4_MATCHED_SOURCE:-$NT4_LOG_DIR/nt4-trtllm-matched128-20261023-112340}
mkdir "$root"
echo "Results: $root"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
export NT4_ALLOW_SOURCE_JIT=1 NT4_BRANCH_SAMPLES=16 NT4_BRANCH_CLOCK=device NT4_BRANCH_TAIL=1
export NT4_EXPERT_PHASE_TIMING=1 NT4_DETACHED_TAIL=0
unset NT4_EXPERT_INTERNAL_TIMING NT4_TRT_DEBUG_INTERMEDIATES CUDA_LAUNCH_BLOCKING MEGA_USE_NCU
unset NT4_MODEL_GRAPH_SPLIT_AFTER NT4_MODEL_SYNC_BOUNDARIES NT4_CUTLASS_MAX_BLOCKS
unset NT4_MXFP8_AB_STAGE_REDUCTION NT4_MXFP8_WORKER_CLUSTERS
"$PY" - <<'PY' > "$root/preflight.log" 2>&1
import os
from pathlib import Path
import torch, flashinfer, megatron.core
assert torch.cuda.device_count() == 4
for i in range(4):
    assert 'GB300' in torch.cuda.get_device_name(i), torch.cuda.get_device_name(i)
    print(i, torch.cuda.get_device_name(i))
for module, env in [(flashinfer, 'FLASHINFER_CHECKOUT'), (megatron.core, 'MCORE_CHECKOUT')]:
    actual = Path(module.__file__).resolve()
    assert actual.is_relative_to(Path(os.environ[env]).resolve()), (actual, env)
    print(env, actual)
p = Path(os.environ['MCORE_CHECKOUT']) / 'megatron/core/models/hybrid/shortcut_block.py'
assert '_nt4_branch_probe' in p.read_text()
assert Path(os.environ['FLASHINFER_TRTLLM_NT4_EXPORT']).is_dir()
print('Environment preflight PASS')
PY
NT4_EXPERT_PHASE_TIMING=0 "$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_branch_calibrate.py" --output "$root/calibration.json" > "$root/calibration.log" 2>&1
# Build once without loading four model replicas or holding distributed ranks.
# The source build can exceed the model-run watchdog on a fresh cache.
echo "Preparing CUTLASS SM103 module; log=$root/build-cutlass.log"
FLASHINFER_JIT_VERBOSE=1 timeout --kill-after=30s 7200s "$PY" -u - <<'PYBUILD' > "$root/build-cutlass.log" 2>&1
from flashinfer.jit import env
from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
print("JIT workspace:", env.FLASHINFER_WORKSPACE_DIR, flush=True)
get_cutlass_fused_moe_module("103")
print("CUTLASS SM103 ready", flush=True)
PYBUILD
for count in ${NT4_COSCHED_COUNTS:-128 512}; do
    case_root="$root/requests$count"
    mkdir "$case_root"
    common=(--source-artifact "$artifact" --mcore-root "$MCORE_CHECKOUT"
        --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl
        --precision mxfp8 --request-count "$count" --num-tokens-to-generate 64
        --model-token-budget 2048 --rounds 2 --timeout-seconds 1800
        --model-graphs --shortcut-overlap --controlled-gc)
    if [[ "$count" == 128 && -f "$source128/record/summary.json" && -d "$source128/inputs" ]]; then
        ln -s "$source128/inputs" "$case_root/inputs"
        printf '%s\n' "$source128" > "$case_root/input-source.txt"
    else
        echo "Recording requests=$count"
        # Native CUTLASS records inputs without hybrid/TRTLLM expert markers.
        NT4_EXPERT_PHASE_TIMING=0 NT4_DETACHED_TAIL=0 "$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
            "${common[@]}" --output "$case_root/record" \
            --branch-contention overlap --branch-backend baseline \
            --branch-input-mode record --branch-inputs "$case_root/inputs" \
            > "$case_root/record.log" 2>&1
    fi
    for arm in hybrid-serial trtllm-serial trtllm-overlap hybrid-overlap; do
        extra=(); export NT4_DETACHED_TAIL=1
        if [[ "$arm" == trtllm-* ]]; then
            export NT4_DETACHED_TAIL=0
            extra+=(--trtllm-clamped-branch --trtllm-exploratory-timing)
        fi
        echo "Starting requests=$count $arm detached=$NT4_DETACHED_TAIL"
        "$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
            "${common[@]}" --output "$case_root/$arm" \
            --mxfp8-nvls-hybrid --branch-native-output --branch-contention "${arm##*-}" \
            --branch-backend direct --branch-input-mode replay --branch-inputs "$case_root/inputs" \
            "${extra[@]}" > "$case_root/$arm.log" 2>&1
        printf '%s\t%s\tPASS\n' "$count" "$arm" >> "$root/status.tsv"
    done
    "$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_coscheduling_headroom.py" "$case_root" > "$case_root/headroom.log" 2>&1
    echo "Finished requests=$count: $case_root/headroom.json"
done
echo "Complete: $root"
