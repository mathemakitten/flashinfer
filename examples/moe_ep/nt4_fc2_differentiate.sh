#!/usr/bin/env bash
set -euo pipefail
experiment=${1:?atomic or geometry}
[[ "$experiment" == atomic || "$experiment" == geometry ]] || exit 2
base=/lustre/fsw/portfolios/nemotron/users/helenn
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$base/megatron-lm-nt4-coscheduling"
export FLASHINFER_WORKSPACE_BASE="$base" MAX_JOBS=4 NT4_ALLOW_SOURCE_JIT=1
unset NT4_EXPERT_STAGE_CAPTURE
python3 -c 'import nvshmem.core, cuda.core, torch; print("Dependencies PASS", torch.cuda.get_device_name(0), flush=True)'
root="$base/logs/nt4-fc2-$experiment-$(date +%Y%m%d-%H%M%S)"
mkdir "$root"
echo "Results: $root"
for rank in 0 1 2 3; do
  echo "Starting $experiment rank$rank"
  if timeout 1800 python3 -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_expert_stage_replay.py" \
    --artifact "$base/logs/nt4-expert-stage32-20261029-144918/rank$rank" \
    --output "$root/rank$rank" --fc2-experiment "$experiment" > "$root/rank$rank.log" 2>&1; then
    echo "rank$rank PASS"
  else
    rc=$?
    echo "rank$rank failed exit $rc; see $root/rank$rank.log"
    exit "$rc"
  fi
done
echo "Complete: $root"
