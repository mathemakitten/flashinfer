#!/usr/bin/env bash
# Matched-input attribution only; not an E2E throughput benchmark.
set -euo pipefail
replica="${1:?usage: nt4_lookup_worker_overlap.sh 0|1}"
case "$replica" in 0|1) ;; *) exit 2 ;; esac
case "$(hostname -s)" in nvl*) ;; *) echo 'Requires CMH compute node' >&2; exit 2 ;; esac
python3 -c 'import torch; assert torch.cuda.device_count() == 4, "Expected four visible GPUs"' 
base=/lustre/fsw/portfolios/nemotron/users/helenn
export PY="${PY:-$(command -v python3)}"
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export MCORE_CHECKOUT="$base/megatron-lm-nt4-coscheduling"
export NT4_LOG_DIR="$base/logs" FLASHINFER_WORKSPACE_BASE="$base"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT"
export FLASHINFER_TRTLLM_NT4_EXPORT="$base/logs/nt4-trtllm-clamped-sweep-20261020-134327/export"
export NT4_ALLOW_SOURCE_JIT=1 MAX_JOBS=4 NT4_DETACHED_TAIL=1 NT4_TE_STREAM_WORKSPACES=1
export NT4_LOOKAHEAD_AFTER_MMA=0 NT4_PACKED_FC1_ACTIVATION=0 NT4_MXFP8_GROUP_HINT=1024
unset NT4_EXPERT_PHASE_TIMING NT4_EXPERT_INTERNAL_TIMING NT4_EXPERT_SPARSE_TIMING NT4_KERNEL_DUMP_DIR
unset NT4_FC1_FAILURE_DIR NT4_HARDWARE_TANH NT4_HARDWARE_TANH_LAYERS NT4_TANH_FC1_ADVISORY
unset NT4_BRANCH_SAMPLES NT4_BRANCH_CLOCK NT4_BRANCH_TAIL NT4_BRANCH_NATIVE_OUTPUT
unset NT4_TRT_DEBUG_INTERMEDIATES CUDA_LAUNCH_BLOCKING MEGA_USE_NCU NT4_EXPERT_STAGE_CAPTURE
unset NT4_MODEL_GRAPH_SPLIT_AFTER NT4_MODEL_SYNC_BOUNDARIES NT4_CUTLASS_MAX_BLOCKS
unset NT4_MXFP8_AB_STAGE_REDUCTION NT4_MXFP8_WORKER_CLUSTERS NT4_PIPELINE_REDUCE_SIDE
unset NT4_FC1_DIRECT_STORE NT4_FC2_PROBE_AFTER_ISSUE NT4_SCHED_LOOKUP
"$PY" -c 'import nvshmem.core' >/dev/null 2>&1 || "$PY" -m pip install --no-deps nvshmem4py-cu13==0.4.0
unset NT4_MATCHED_FORWARD_MODE NT4_MATCHED_FORWARD_ROOT NT4_MATCHED_FORWARD_SAMPLES
unset NT4_TANH_LOGIT_SOURCE NT4_EXPERT_FAILURE_DIR NT4_TANH_EXPERT_ADVISORY NT4_TANH_OUTPUT_ADVISORY
export NT4_SCHED_LOOKUP=1

source_root="$NT4_LOG_DIR/nt4-trtllm-matched128-20261023-112340"
[[ -f "$source_root/record/summary.json" && -d "$source_root/inputs" ]]
export NT4_BRANCH_SAMPLES=16 NT4_BRANCH_CLOCK=device NT4_BRANCH_TAIL=1
# Branch clocks measure expert, mixer, join/tail. No internal warp markers.
unset NT4_EXPERT_PHASE_TIMING NT4_EXPERT_INTERNAL_TIMING NT4_EXPERT_SPARSE_TIMING
export NT4_HARDWARE_TANH=0 NT4_SCHED_LOOKUP=1
root="$NT4_LOG_DIR/nt4-lookup-worker-overlap-r${replica}-$(date +%Y%m%d-%H%M%S)"
mkdir "$root"
ln -s "$source_root/inputs" "$root/inputs"
printf 'Results: %s\n' "$root"
printf 'lookup=1 detached_tail=1 hardware_tanh=0 group_hint=1024 requests=128 samples=16 rounds=2\n' > "$root/config.txt"
arms=(152-overlap-start 144-overlap 152-overlap-end 152-serial 144-serial)
if [[ "$replica" == 1 ]]; then
  arms=(144-serial 152-serial 152-overlap-start 144-overlap 152-overlap-end)
fi
for arm in "${arms[@]}"; do
  cap=${arm%%-*}
  schedule=${arm#*-}
  schedule=${schedule%%-*}
  printf 'Starting %s; log=%s/%s.log\n' "$arm" "$root" "$arm"
  if "$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
    --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
    --mcore-root "$MCORE_CHECKOUT" \
    --prompt-file "$base/research/routing/openmath2_unique2048.jsonl" \
    --output "$root/$arm" --precision mxfp8 --request-count 128 \
    --num-tokens-to-generate 64 --model-token-budget 2048 \
    --model-graphs --shortcut-overlap --controlled-gc --mxfp8-nvls-hybrid \
    --branch-native-output --branch-contention "$schedule" --branch-backend direct \
    --branch-input-mode replay --branch-inputs "$root/inputs" \
    --mxfp8-worker-clusters "$cap" --rounds 2 --timeout-seconds 1800 \
    > "$root/$arm.log" 2>&1; then
    printf '%s\tPASS\n' "$arm" >> "$root/status.tsv"
  else
    rc=$?
    printf '%s\tFAIL\t%s\n' "$arm" "$rc" >> "$root/status.tsv"
    echo "Stopping after $arm failed ($rc)" >&2
    exit "$rc"
  fi
done
printf 'Completed: %s\n' "$root"
