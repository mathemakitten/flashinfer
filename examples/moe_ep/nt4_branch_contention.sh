#!/usr/bin/env bash
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
root=${NT4_BRANCH_ROOT:-"$NT4_LOG_DIR/nt4-branch-contention-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$root"
echo "Results: $root"
export NT4_BRANCH_SAMPLES=16
export NT4_BRANCH_CLOCK=device
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_branch_calibrate.py" \
  --output "$root/calibration.json" > "$root/calibration.log" 2>&1
echo "Timestamp calibration passed; log=$root/calibration.log"
for clusters in 76 56; do
  printf '{"group_hint":1024,"max_active_clusters":%s}\n' "$clusters" > "$root/knobs$clusters.json"
  for schedule in serial overlap; do
    name="clusters$clusters-$schedule"
    echo "Starting $name; log=$root/$name.log"
    "$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
      --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
      --mcore-root "$MCORE_CHECKOUT" \
      --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
      --output "$root/$name" --precision bf16 --request-count 128 \
      --num-tokens-to-generate 64 --model-token-budget 2048 \
      --model-graphs --shortcut-overlap --controlled-gc \
      --bf16-knobs-json "$root/knobs$clusters.json" \
      --branch-contention "$schedule" --rounds 2 --timeout-seconds 1800 \
      > "$root/$name.log" 2>&1
  done
done
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_branch_summary.py" "$root"
