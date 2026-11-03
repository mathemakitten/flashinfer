#!/usr/bin/env bash
# Replay recorded native block inputs; compare branch latency, not E2E throughput.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
: "${NT4_MATCHED_ROOT:?}" "${FLASHINFER_TRTLLM_NT4_EXPORT:?}"
[[ -f "$NT4_MATCHED_ROOT/record/summary.json" ]] || { echo 'Finish input recording first'; exit 1; }
export NT4_BRANCH_SAMPLES=16 NT4_BRANCH_CLOCK=device NT4_BRANCH_TAIL=1
unset NT4_CUTLASS_MAX_BLOCKS
for arm in hybrid-serial trtllm-serial trtllm-overlap hybrid-overlap; do
  extra=()
  if [[ "$arm" == trtllm-* ]]; then
    extra+=(--trtllm-clamped-branch --trtllm-exploratory-timing)
  fi
  echo "Starting $arm; log=$NT4_MATCHED_ROOT/$arm.log"
  "$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
    --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
    --mcore-root "$MCORE_CHECKOUT" \
    --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
    --output "$NT4_MATCHED_ROOT/$arm" --precision mxfp8 --request-count 128 \
    --num-tokens-to-generate 64 --model-token-budget 2048 \
    --model-graphs --shortcut-overlap --controlled-gc --mxfp8-nvls-hybrid \
    --branch-native-output --branch-contention "${arm##*-}" --branch-backend direct \
    --branch-input-mode replay --branch-inputs "$NT4_MATCHED_ROOT/inputs" \
    --rounds 2 --timeout-seconds 1800 "${extra[@]}" > "$NT4_MATCHED_ROOT/$arm.log" 2>&1
done
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_branch_backend_summary.py" \
  "$NT4_MATCHED_ROOT" --precision mxfp8 --trtllm-matched
