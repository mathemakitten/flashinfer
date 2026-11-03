#!/usr/bin/env bash
# Modest worker reductions, independently checked before E2E comparison.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
: "${NT4_BF16_TUNE_ROOT:?}"
root=${NT4_HEADROOM_ROOT:-"$NT4_LOG_DIR/nt4-overlap-headroom-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$root"
example="$FLASHINFER_CHECKOUT/examples/moe_ep"
echo "Results: $root"
for clusters in 74 72 68; do
  echo "Checking clusters=$clusters; log=$root/check$clusters.log"
  timeout --signal=TERM --kill-after=30s 1800s \
    "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "$example/nt4_bf16_tune.py" --worker \
    --artifact "$NT4_BF16_TUNE_ROOT/capture128/bf16-capture" \
    --candidate "workers$clusters" --result "$root/check$clusters.json" \
    --warmup 10 --iterations 30 --rounds 3 > "$root/check$clusters.log" 2>&1
done
"$PY" "$example/nt4_e2e.py" \
  --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
  --mcore-root "$MCORE_CHECKOUT" \
  --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
  --output "$root/e2e" --precision bf16 --request-count 128 \
  --num-tokens-to-generate 512 --model-token-budget 2048 \
  --model-graphs --shortcut-overlap --controlled-gc \
  --overlap-headroom-sweep --rounds 2 --timeout-seconds 7200
"$PY" "$example/nt4_headroom_summary.py" "$root/e2e/summary.json"
