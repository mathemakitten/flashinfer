#!/usr/bin/env bash
# FC1 turnover attribution; preserves numerical gates, stops at first failure.
set -euo pipefail
base=/lustre/fsw/portfolios/nemotron/users/helenn
export PY="${PY:-$(command -v python3)}"
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export MCORE_CHECKOUT="$base/megatron-lm-nt4-coscheduling"
export NT4_LOG_DIR="$base/logs"
export FLASHINFER_WORKSPACE_BASE="$base"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT"
export NT4_ALLOW_SOURCE_JIT=1 MAX_JOBS=4 NT4_DETACHED_TAIL=1
export NT4_BRANCH_SAMPLES=16 NT4_BRANCH_CLOCK=device NT4_BRANCH_TAIL=1
export NT4_EXPERT_PHASE_TIMING=1
export NT4_EXPERT_INTERNAL_TIMING=1
unset NT4_TRT_DEBUG_INTERMEDIATES CUDA_LAUNCH_BLOCKING MEGA_USE_NCU
unset NT4_MODEL_GRAPH_SPLIT_AFTER NT4_MODEL_SYNC_BOUNDARIES NT4_CUTLASS_MAX_BLOCKS
unset NT4_MXFP8_AB_STAGE_REDUCTION NT4_MXFP8_WORKER_CLUSTERS
"$PY" -c 'import nvshmem.core' >/dev/null 2>&1 || "$PY" -m pip install --no-deps nvshmem4py-cu13==0.4.0
"$PY" -c 'import nvshmem.core; import torch, flashinfer; print("Dependency preflight PASS", flush=True)'
root="$NT4_LOG_DIR/nt4-fc1-turnover128-$(date +%Y%m%d-%H%M%S)"
source_root="$NT4_LOG_DIR/nt4-trtllm-matched128-20261023-112340"
test -f "$source_root/record/summary.json"
mkdir "$root"
ln -s "$source_root/inputs" "$root/inputs"
printf 'Results: %s\n' "$root"
for variant in turnover; do
  export NT4_LOOKAHEAD_AFTER_MMA=0
  [[ "$variant" != after-mma ]] || export NT4_LOOKAHEAD_AFTER_MMA=1
  mkdir "$root/$variant"
  printf '%s\n' "$NT4_LOOKAHEAD_AFTER_MMA" > "$root/$variant/lookahead-after-mma.txt"
  for schedule in serial overlap; do
    arm="hybrid-$schedule"
    printf 'Starting %s/%s\n' "$variant" "$arm"
    "$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
      --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
      --mcore-root "$MCORE_CHECKOUT" \
      --prompt-file "$base/research/routing/openmath2_unique2048.jsonl" \
      --output "$root/$variant/$arm" --precision mxfp8 --request-count 128 \
      --num-tokens-to-generate 64 --model-token-budget 2048 \
      --model-graphs --shortcut-overlap --controlled-gc --mxfp8-nvls-hybrid \
      --branch-native-output --branch-contention "$schedule" --branch-backend direct \
      --branch-input-mode replay --branch-inputs "$root/inputs" \
      --rounds 2 --timeout-seconds 1800 > "$root/$variant/$arm.log" 2>&1
    printf '%s\t%s\tPASS\n' "$variant" "$arm" >> "$root/status.tsv"
  done
done
printf 'Complete: %s\n' "$root"
