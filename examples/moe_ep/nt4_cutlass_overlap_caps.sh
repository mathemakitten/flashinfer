#!/usr/bin/env bash
# Run on a separate node from the vLLM variance audit. No cache clearing needed.
set -euo pipefail
: "${PY:?Set PY to the experiment Python interpreter}"
: "${FLASHINFER_CHECKOUT:?Set FLASHINFER_CHECKOUT}"
: "${MCORE_CHECKOUT:?Set MCORE_CHECKOUT}"
: "${NT4_LOG_DIR:?Set NT4_LOG_DIR}"
result_root="${NT4_CUTLASS_CAP_ROOT:-$NT4_LOG_DIR/cutlass-overlap-caps-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$result_root"
printf 'Results: %s\n' "$result_root"

# This compiles the experimental module once, then reuses it at every cap.
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_cutlass_cap_smoke.py" \
    --output "$result_root/smoke" > "$result_root/smoke.log" 2>&1

# vLLM stays at the source model's expert-max-blocks setting in every pair.
# CUTLASS prepare/activation/finalize kernels and NVLS are unchanged.
for cap in 0 128 64 304; do
    printf 'CUTLASS max blocks=%s; log=%s/cap%s.log\n' "$cap" "$result_root" "$cap"
    "$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
        --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
        --mcore-root "$MCORE_CHECKOUT" \
        --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
        --output "$result_root/cap$cap" \
        --precision bf16 --request-count 128 --model-token-budget 2048 \
        --model-graphs --shortcut-overlap --cutlass-overlap-pair \
        --cutlass-max-blocks "$cap" --rounds 3 --timeout-seconds 7200 \
        > "$result_root/cap$cap.log" 2>&1
done
printf 'Complete: %s\n' "$result_root"
