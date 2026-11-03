#!/usr/bin/env bash
set -euo pipefail
workers=${1:-152}
[[ "$workers" == 128 || "$workers" == 152 ]] || exit 2
base=/lustre/fsw/portfolios/nemotron/users/helenn
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$base/megatron-lm-nt4-coscheduling"
export FLASHINFER_WORKSPACE_BASE="$base" MAX_JOBS=4 NT4_ALLOW_SOURCE_JIT=1
unset NT4_EXPERT_STAGE_CAPTURE
root="$base/logs/nt4-fc12-mixed-w$workers-$(date +%Y%m%d-%H%M%S)"
mkdir "$root"
echo "Results: $root"
for rank in ${NT4_MIXED_RANKS:-0 1 2 3}; do
 echo "Starting workers=$workers rank=$rank"
 if timeout --kill-after=10s 600 python3 -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_expert_stage_replay.py" \
    --artifact "$base/logs/nt4-expert-stage32-20261029-144918/rank$rank" \
    --output "$root/rank$rank" --fc12-mixed-geometry --workers "$workers" > "$root/rank$rank.log" 2>&1; then
   echo "rank$rank PASS"
 else
   rc=$?; echo "Failed exit=$rc; see $root/rank$rank.log"; exit "$rc"
 fi
done
echo "Complete: $root"
