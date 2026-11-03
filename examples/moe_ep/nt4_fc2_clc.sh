#!/usr/bin/env bash
# Hardware CLC prototype. All-ready FC2 only; no E2E performance claim.
set -euo pipefail
base=/lustre/fsw/portfolios/nemotron/users/helenn
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$base/megatron-lm-nt4-coscheduling"
export FLASHINFER_WORKSPACE_BASE="$base" MAX_JOBS=4 NT4_ALLOW_SOURCE_JIT=1
unset NT4_EXPERT_STAGE_CAPTURE NT4_EXPERT_SPARSE_TIMING NT4_EXPERT_INTERNAL_TIMING
unset NT4_PIPELINE_REDUCE_SIDE NT4_FC2_PROBE_AFTER_ISSUE NT4_LOOKAHEAD_AFTER_MMA
root="$base/logs/nt4-fc2-clc-$(date +%Y%m%d-%H%M%S)"
mkdir "$root"
printf 'Results: %s\n' "$root"
for rank in 0 1 2 3; do
 for workers in 128 152; do
  printf 'Starting rank%s workers%s\n' "$rank" "$workers"
  timeout --kill-after=30s 1800 python3 -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_expert_stage_replay.py" \
    --artifact "$base/logs/nt4-expert-stage32-20261029-144918/rank$rank" \
    --output "$root/rank$rank-w$workers" --workers "$workers" \
    --fc2-experiment clc > "$root/rank$rank-w$workers.log" 2>&1
  python3 - "$root/rank$rank-w$workers/summary.json" <<'CHECK'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='PASS_FC2_CLC_DIAGNOSTIC', x['status']
assert len(x['fc2_variants'])==7
print(json.dumps(dict(stage='matrix_pass',summary=sys.argv[1],timings=x['timings'])),flush=True)
CHECK
done
done
printf 'Complete: %s\n' "$root"
