#!/usr/bin/env bash
set -euo pipefail
requested="${1:?usage: nt4_fused_worker_headroom.sh 96|128|144}"
case "$requested" in 96|128|144) ;; *) exit 2;; esac
ranks=(2 0 1 3)
case "$(hostname -s)" in nvl*) ;; *) exit 2;; esac
base=/lustre/fsw/portfolios/nemotron/users/helenn
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$base/megatron-lm-nt4-coscheduling"
export FLASHINFER_WORKSPACE_BASE="$base" MAX_JOBS=4 NT4_ALLOW_SOURCE_JIT=1
unset NT4_EXPERT_STAGE_CAPTURE NT4_EXPERT_SPARSE_TIMING NT4_EXPERT_INTERNAL_TIMING NT4_EXPERT_PHASE_TIMING
unset NT4_PIPELINE_REDUCE_SIDE NT4_FC2_PROBE_AFTER_ISSUE NT4_LOOKAHEAD_AFTER_MMA
unset NT4_HARDWARE_TANH NT4_HARDWARE_TANH_LAYERS NT4_TANH_FC1_ADVISORY NT4_SCHED_LOOKUP
python3 -c 'import nvshmem.core' >/dev/null 2>&1 || python3 -m pip install --no-deps nvshmem4py-cu13==0.4.0
root="$base/logs/nt4-fused-headroom-w${requested}-$(date +%Y%m%d-%H%M%S)"
mkdir "$root"
printf 'Results: %s\n' "$root"
for rank in "${ranks[@]}"; do
 for workers in 152 "$requested"; do
 echo "Starting saved rank $rank workers $workers"
 timeout --kill-after=30s 1200 python3 -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_lookup_stage_attribution.py" \
  --artifact "$base/logs/nt4-expert-stage32-20261029-144918/rank$rank" \
  --output "$root/rank${rank}-w${workers}" --workers "$workers" --iterations 200 \
  --lookup-enabled --fc12-single-group > "$root/rank${rank}-w${workers}.log" 2>&1
 echo "rank$rank workers$workers complete"
 done
done
printf 'Complete: %s\n' "$root"
