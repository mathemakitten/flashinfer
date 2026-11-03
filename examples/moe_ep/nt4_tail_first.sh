#!/usr/bin/env bash
# Full-worker BF16 shortcut launch-order experiment; no phase instrumentation.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
root=${NT4_TAIL_FIRST_ROOT:-"$NT4_LOG_DIR/nt4-tail-first-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$root"
printf '{"group_hint":1024,"max_active_clusters":76}\n' > "$root/knobs.json"
echo "Results: $root/e2e"
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
  --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
  --mcore-root "$MCORE_CHECKOUT" \
  --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
  --output "$root/e2e" --precision bf16 --request-count 128 \
  --num-tokens-to-generate 512 --model-token-budget 2048 \
  --model-graphs --shortcut-overlap --controlled-gc \
  --bf16-knobs-json "$root/knobs.json" --tail-first-pair \
  --rounds 2 --timeout-seconds 7200
