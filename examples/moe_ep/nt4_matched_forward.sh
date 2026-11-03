#!/usr/bin/env bash
# All shortcut continuations restored; diagnostic latency, never tokens/s.
set -euo pipefail
case "$(hostname -s)" in nvl*) ;; *) exit 2 ;; esac
base=/lustre/fsw/portfolios/nemotron/users/helenn
export PY="${PY:-$(command -v python3)}"
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export MCORE_CHECKOUT="$base/megatron-lm-nt4-coscheduling"
export NT4_LOG_DIR="$base/logs" FLASHINFER_WORKSPACE_BASE="$base"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT"
export FLASHINFER_TRTLLM_NT4_EXPORT="$base/logs/nt4-trtllm-clamped-sweep-20261020-134327/export"
export NT4_ALLOW_SOURCE_JIT=1 MAX_JOBS=4 NT4_DETACHED_TAIL=1
export NT4_LOOKAHEAD_AFTER_MMA=0 NT4_PACKED_FC1_ACTIVATION=0
unset NT4_BRANCH_SAMPLES NT4_BRANCH_CLOCK NT4_BRANCH_TAIL
unset NT4_EXPERT_PHASE_TIMING
export NT4_SCHED_LOOKUP=1 NT4_MXFP8_GROUP_HINT=1024 NT4_TE_STREAM_WORKSPACES=1
unset NT4_EXPERT_INTERNAL_TIMING NT4_EXPERT_SPARSE_TIMING NT4_EXPERT_STAGE_CAPTURE
unset NT4_PIPELINE_REDUCE_SIDE NT4_FC2_PROBE_AFTER_ISSUE NT4_FC1_DIRECT_STORE
unset NT4_HARDWARE_TANH_LAYERS NT4_TANH_LOGIT_SOURCE NT4_FC1_FAILURE_DIR NT4_EXPERT_FAILURE_DIR
unset NT4_KERNEL_DUMP_DIR NT4_TRT_DEBUG_INTERMEDIATES CUDA_LAUNCH_BLOCKING MEGA_USE_NCU
unset NT4_MODEL_GRAPH_SPLIT_AFTER NT4_MODEL_SYNC_BOUNDARIES NT4_CUTLASS_MAX_BLOCKS
unset NT4_MXFP8_AB_STAGE_REDUCTION NT4_MXFP8_WORKER_CLUSTERS

"$PY" -c 'import nvshmem.core' >/dev/null 2>&1 || "$PY" -m pip install --no-deps nvshmem4py-cu13==0.4.0
root="$NT4_LOG_DIR/nt4-matched-forward128-$(date +%Y%m%d-%H%M%S)"
mkdir "$root"
export NT4_MATCHED_FORWARD_ROOT="$root/inputs" NT4_MATCHED_FORWARD_SAMPLES=8
printf 'Results: %s\n' "$root"
for arm in record exp-start hardware exp-end; do
  export NT4_MATCHED_FORWARD_MODE=replay NT4_HARDWARE_TANH=0
  if [[ "$arm" == record ]]; then export NT4_MATCHED_FORWARD_MODE=record; fi
  if [[ "$arm" == hardware ]]; then export NT4_HARDWARE_TANH=1; fi
  echo "Starting $arm; $root/$arm.log"
  "$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
    --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
    --mcore-root "$MCORE_CHECKOUT" \
    --prompt-file "$base/research/routing/openmath2_unique2048.jsonl" \
    --output "$root/$arm" --precision mxfp8 --request-count 128 \
    --num-tokens-to-generate 32 --model-token-budget 2048 \
    --model-graphs --controlled-gc --mxfp8-nvls-hybrid --shortcut-overlap \
    --branch-contention overlap --branch-backend direct \
    --rounds 1 --timeout-seconds 1800 > "$root/$arm.log" 2>&1
  printf '%s\tCOMPLETE\n' "$arm" >> "$root/status.tsv"
done
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_matched_forward_report.py" "$root"
