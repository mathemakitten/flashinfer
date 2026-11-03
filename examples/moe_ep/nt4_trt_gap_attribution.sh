#!/usr/bin/env bash
# Matched native-continuation diagnostic. Instrumented timings are not E2E.
set -euo pipefail
schedule="${1:?Usage: script serial|overlap coarse|internal}"
detail="${2:?Usage: script serial|overlap coarse|internal}"
[[ "$schedule" == serial || "$schedule" == overlap ]] || exit 2
[[ "$detail" == coarse || "$detail" == internal ]] || exit 2
base=/lustre/fsw/portfolios/nemotron/users/helenn
export PY="${PY:-$(command -v python3)}"
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export MCORE_CHECKOUT="$base/megatron-lm-nt4-coscheduling"
export NT4_LOG_DIR="$base/logs" FLASHINFER_WORKSPACE_BASE="$base"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT"
export FLASHINFER_TRTLLM_NT4_EXPORT="$base/logs/nt4-trtllm-clamped-sweep-20261020-134327/export"
export NT4_ALLOW_SOURCE_JIT=1 MAX_JOBS=4 NT4_DETACHED_TAIL=1
export NT4_LOOKAHEAD_AFTER_MMA=0 NT4_PACKED_FC1_ACTIVATION=0
export NT4_BRANCH_SAMPLES=16 NT4_BRANCH_CLOCK=device NT4_BRANCH_TAIL=1
export NT4_EXPERT_PHASE_TIMING=1 NT4_EXPERT_INTERNAL_TIMING=0
if [[ "$detail" == internal ]]; then export NT4_EXPERT_INTERNAL_TIMING=1; fi
unset NT4_HARDWARE_TANH_LAYERS NT4_TANH_LOGIT_SOURCE NT4_FC1_FAILURE_DIR NT4_EXPERT_FAILURE_DIR
unset NT4_KERNEL_DUMP_DIR NT4_TRT_DEBUG_INTERMEDIATES CUDA_LAUNCH_BLOCKING MEGA_USE_NCU
unset NT4_MODEL_GRAPH_SPLIT_AFTER NT4_MODEL_SYNC_BOUNDARIES NT4_CUTLASS_MAX_BLOCKS
unset NT4_MXFP8_AB_STAGE_REDUCTION NT4_MXFP8_WORKER_CLUSTERS
source_root="$NT4_LOG_DIR/nt4-trtllm-matched128-20261023-112340"
test -f "$source_root/record/summary.json"
test -d "$FLASHINFER_TRTLLM_NT4_EXPORT"
"$PY" -c 'import nvshmem.core, torch, flashinfer; print("Dependency imports PASS", flush=True)'
root="$NT4_LOG_DIR/nt4-trt-gap-${schedule}-${detail}-$(date +%Y%m%d-%H%M%S)"
mkdir "$root"
printf 'Results: %s\n' "$root"
for arm in hardware-start trtllm hardware-end; do
  export NT4_HARDWARE_TANH=1 NT4_TANH_FC1_ADVISORY=1
  extra=()
  if [[ "$arm" == trtllm ]]; then
    export NT4_HARDWARE_TANH=0 NT4_TANH_FC1_ADVISORY=0
    extra+=(--trtllm-clamped-branch --trtllm-exploratory-timing)
  fi
  printf 'Starting %s; log=%s/%s.log\n' "$arm" "$root" "$arm"
  "$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
    --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
    --mcore-root "$MCORE_CHECKOUT" \
    --prompt-file "$base/research/routing/openmath2_unique2048.jsonl" \
    --output "$root/$arm" --precision mxfp8 --request-count 128 \
    --num-tokens-to-generate 64 --model-token-budget 2048 \
    --model-graphs --shortcut-overlap --controlled-gc --mxfp8-nvls-hybrid \
    --branch-native-output --branch-contention "$schedule" --branch-backend direct \
    --branch-input-mode replay --branch-inputs "$source_root/inputs" \
    --rounds 2 --timeout-seconds 1800 "${extra[@]}" > "$root/$arm.log" 2>&1
  printf '%s\tCOMPLETED_CHECK_NUMERICAL_EXCEPTIONS\n' "$arm" >> "$root/status.tsv"
done
printf 'Complete: %s\n' "$root"
