#!/usr/bin/env bash
# GB300 BF16 worker sweep. Each argument is a cluster count (two CTAs/cluster).
set -euo pipefail
: "${PY:?source the node setup first}"
: "${FLASHINFER_CHECKOUT:?}"
: "${MCORE_CHECKOUT:?}"
: "${NT4_LOG_DIR:?}"
: "${NT4_BF16_TUNE_ROOT:?}"
export NT4_WORKER_SWEEP_ROOT=${NT4_WORKER_SWEEP_ROOT:-"$NT4_LOG_DIR/nt4-bf16-workers-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$NT4_WORKER_SWEEP_ROOT"
counts=("$@")
if (( ${#counts[@]} == 0 )); then counts=(76 56 38 76); fi
example="$FLASHINFER_CHECKOUT/examples/moe_ep"
index=0
for clusters in "${counts[@]}"; do
  case "$clusters" in 76|56|38) ;; *) echo 'Supported cluster counts: 76 56 38' >&2; exit 2;; esac
  run="$NT4_WORKER_SWEEP_ROOT/$index-clusters$clusters"
  mkdir "$run"
  printf '{"group_hint":1024,"max_active_clusters":%s}\n' "$clusters" > "$run/knobs.json"
  echo "Correctness and isolated timing: $run/check.log"
  timeout --signal=TERM --kill-after=30s 1800s \
    "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "$example/nt4_bf16_tune.py" --worker \
    --artifact "$NT4_BF16_TUNE_ROOT/capture128/bf16-capture" \
    --candidate "workers$clusters" --result "$run/check.json" \
    --warmup 20 --iterations 100 --rounds 5 > "$run/check.log" 2>&1
  echo "Overlap comparison: $run/e2e.log"
  "$PY" "$example/nt4_e2e.py" \
    --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
    --mcore-root "$MCORE_CHECKOUT" \
    --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
    --output "$run/e2e" --precision bf16 --request-count 128 \
    --num-tokens-to-generate 512 --model-token-budget 2048 \
    --model-graphs --shortcut-overlap --controlled-gc \
    --bf16-knobs-json "$run/knobs.json" \
    --alternating-overlap-pairs 2 --rounds 2 --timeout-seconds 7200 \
    > "$run/e2e.log" 2>&1
  index=$((index + 1))
done
echo "Complete: $NT4_WORKER_SWEEP_ROOT"
