#!/usr/bin/env bash
set -euo pipefail
base=/lustre/fsw/portfolios/nemotron/users/helenn
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$base/megatron-lm-nt4-coscheduling"
export FLASHINFER_WORKSPACE_BASE="$base" MAX_JOBS=4 NT4_ALLOW_SOURCE_JIT=1
unset NT4_EXPERT_STAGE_CAPTURE
root="$base/logs/nt4-fc2-trt-scheduler-$(date +%Y%m%d-%H%M%S)"
mkdir "$root"
echo "Results: $root"
for rank in 0 1 2 3; do
 for arm in clc_start static clc_end; do
  config=7; scheduler=clc
  if [[ "$arm" == static ]]; then config=6; scheduler=static; fi
  echo "Starting rank$rank $arm"
  timeout 900 python3 -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_expert_stage_replay.py" \
    --artifact "$base/logs/nt4-expert-stage32-20261029-144918/rank$rank" \
    --output "$root/rank$rank-$arm" --trt-only --trt-fc2-config "$config" \
    --expected-scheduler "$scheduler" > "$root/rank$rank-$arm.log" 2>&1
 done
 python3 - "$root" "$rank" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);rank=sys.argv[2]
rows=[json.loads((r/f'rank{rank}-{a}/summary.json').read_text()) for a in ('clc_start','static','clc_end')]
for key in ('common_activation_sha256','common_scale_sha256','fc2_output_sha256'):
 assert len({d[key] for d in rows})==1, (rank,key,'scheduler comparison invalid')
print(json.dumps(dict(rank=rank,status='MATCHED_INPUT_OUTPUT_PASS',fc2_us=[d['timings']['trt_fc2']['median_us'] for d in rows])),flush=True)
PY
done
echo "Complete: $root"
