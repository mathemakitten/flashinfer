#!/usr/bin/env bash
# Export + correctness preflight, then real NT4 decode comparison on EP4 GB300.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}" "${TRTLLM_GEN_ROOT:?}"
export NT4_TRT_COMPARE_ROOT="${NT4_TRT_COMPARE_ROOT:-$NT4_LOG_DIR/nt4-trtllm-clamped-sweep-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$NT4_TRT_COMPARE_ROOT" ]]; then echo "Use a fresh output: $NT4_TRT_COMPARE_ROOT" >&2; exit 1; fi
mkdir -p "$NT4_TRT_COMPARE_ROOT"
export NT4_TRT_EXPORT_DIR="$NT4_TRT_COMPARE_ROOT/export"
export NT4_TRT_EXPORT_CONFIG="$TRTLLM_GEN_ROOT/cluster/mxfp8_nt4_sweep.json"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
printf 'Results: %s\n' "$NT4_TRT_COMPARE_ROOT"
bash "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_trtllm_clamped.sh"
export FLASHINFER_TRTLLM_NT4_EXPORT="$NT4_TRT_EXPORT_DIR"
# Clear diagnostic overrides inherited from earlier branch experiments.
unset NT4_BRANCH_INPUTS NT4_BRANCH_INPUT_MODE NT4_BRANCH_NATIVE_OUTPUT NT4_TRTLLM_RELU2_SURROGATE
unset NT4_CUTLASS_MAX_BLOCKS NT4_MXFP8_WORKER_CLUSTERS NT4_MXFP8_AB_STAGE_REDUCTION
counts=("$@")
if [[ ${#counts[@]} == 0 ]]; then counts=(32 128 512); fi
for requests in "${counts[@]}"; do
  case "$requests" in 32|128|512) ;; *) echo "Expected request count 32, 128 or 512" >&2; exit 2;; esac
  echo "Starting requests=$requests; log=$NT4_TRT_COMPARE_ROOT/requests$requests.log"
  "$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
    --source-artifact "${NT4_ARTIFACT:-$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901}" \
    --mcore-root "$MCORE_CHECKOUT" \
    --prompt-file "${NT4_PROMPT_FILE:-/lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl}" \
    --output "$NT4_TRT_COMPARE_ROOT/requests$requests" \
    --precision mxfp8 --request-count "$requests" --model-token-budget 2048 \
    --num-tokens-to-generate 512 --rounds 2 --timeout-seconds 7200 \
    --model-graphs --shortcut-overlap --controlled-gc \
    --mxfp8-nvls-hybrid --trtllm-clamped-compare \
    --cutlass-decode-profile-ids 39 428 \
    > "$NT4_TRT_COMPARE_ROOT/requests$requests.log" 2>&1
done
printf 'Complete: %s\n' "$NT4_TRT_COMPARE_ROOT"
