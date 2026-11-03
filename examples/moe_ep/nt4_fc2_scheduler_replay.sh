#!/usr/bin/env bash
set -euo pipefail
base=/lustre/fsw/portfolios/nemotron/users/helenn
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$base/megatron-lm-nt4-coscheduling"
export FLASHINFER_WORKSPACE_BASE="$base" MAX_JOBS=4 NT4_ALLOW_SOURCE_JIT=1
unset NT4_EXPERT_STAGE_CAPTURE NT4_EXPERT_SPARSE_TIMING NT4_PIPELINE_REDUCE_SIDE
export_root="$base/logs/nt4-fc2-scheduler-twins-20261030-101503"
root="$base/logs/nt4-fc2-scheduler-replay-$(date +%Y%m%d-%H%M%S)"
mkdir "$root"
printf 'Results: %s\n' "$root"
for rank in 0 1 2 3; do
 for arm in clc_start static_persistent clc_middle static clc_end; do
  config=3; scheduler=clc
  if [[ "$arm" == static_persistent ]]; then config=2; scheduler=static-persistent; fi
  if [[ "$arm" == static ]]; then config=1; scheduler=static; fi
  printf 'Starting rank%s %s\n' "$rank" "$arm"
  timeout --kill-after=30s 1800 python3 -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_expert_stage_replay.py" \
    --artifact "$base/logs/nt4-expert-stage32-20261029-144918/rank$rank" \
    --output "$root/rank$rank-$arm" --trt-only --trt-fc1-config 0 --trt-fc2-config "$config" \
    --trt-export "$export_root" --expected-scheduler "$scheduler" > "$root/rank$rank-$arm.log" 2>&1
 done
 python3 - "$root" "$rank" <<'CHECK'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);rank=sys.argv[2]
arms=('clc_start','static_persistent','clc_middle','static','clc_end')
rows=[json.loads((r/f'rank{rank}-{a}/summary.json').read_text()) for a in arms]
for key in ('common_activation_sha256','common_scale_sha256','fc2_output_sha256'):
 assert len({d[key] for d in rows})==1, (rank,key,'scheduler comparison invalid')
result=dict(rank=rank,status='MATCHED_INPUT_OUTPUT_PASS',arms={a:dict(timing=d['timings']['trt_fc2'],launch=d['fc2_launch']) for a,d in zip(arms,rows)})
(r/f'rank{rank}-comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
CHECK
done
printf 'Complete: %s\n' "$root"
