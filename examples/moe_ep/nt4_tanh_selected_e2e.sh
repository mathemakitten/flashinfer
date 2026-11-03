#!/usr/bin/env bash
# Exploratory full-model comparison; candidate outputs drive decode.
set -euo pipefail
base=/lustre/fsw/portfolios/nemotron/users/helenn
export PY="${PY:-$(command -v python3)}"
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export MCORE_CHECKOUT="$base/megatron-lm-nt4-coscheduling"
export NT4_LOG_DIR="$base/logs"
export FLASHINFER_WORKSPACE_BASE="$base"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT"
export NT4_ALLOW_SOURCE_JIT=1 MAX_JOBS=4 NT4_DETACHED_TAIL=1
export NT4_LOOKAHEAD_AFTER_MMA=0 NT4_PACKED_FC1_ACTIVATION=0
unset NT4_EXPERT_PHASE_TIMING NT4_EXPERT_INTERNAL_TIMING NT4_KERNEL_DUMP_DIR
unset NT4_FC1_FAILURE_DIR NT4_HARDWARE_TANH NT4_TANH_FC1_ADVISORY
unset NT4_BRANCH_SAMPLES NT4_BRANCH_CLOCK NT4_BRANCH_TAIL
unset NT4_TRT_DEBUG_INTERMEDIATES CUDA_LAUNCH_BLOCKING MEGA_USE_NCU
unset NT4_MODEL_GRAPH_SPLIT_AFTER NT4_MODEL_SYNC_BOUNDARIES NT4_CUTLASS_MAX_BLOCKS
unset NT4_MXFP8_AB_STAGE_REDUCTION NT4_MXFP8_WORKER_CLUSTERS
"$PY" -c 'import nvshmem.core' >/dev/null 2>&1 || "$PY" -m pip install --no-deps nvshmem4py-cu13==0.4.0
"$PY" -c 'import nvshmem.core; print("NVSHMEM dependency preflight PASS", flush=True)'
root="$NT4_LOG_DIR/nt4-tanh-selected-e2e128-$(date +%Y%m%d-%H%M%S)"
printf 'Results: %s\n' "$root"
"$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
  --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
  --mcore-root "$MCORE_CHECKOUT" \
  --prompt-file "$base/research/routing/openmath2_unique2048.jsonl" \
  --output "$root" --precision mxfp8 --request-count 128 \
  --num-tokens-to-generate 512 --model-token-budget 2048 \
  --model-graphs --shortcut-overlap --controlled-gc --mxfp8-nvls-hybrid \
  --cutlass-decode-profile-ids 39 428 --hardware-tanh-confirm --hardware-tanh-layers 32,68 \
  --rounds 3 --timeout-seconds 1800
