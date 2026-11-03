#!/usr/bin/env bash
# Matched-input MXFP8 scheduling diagnostic. Not an E2E throughput benchmark.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
resume=false
case "${1:-}" in
  "") ;;
  --resume) resume=true; : "${NT4_MXFP8_PIPELINE_ROOT:?set the previous output root}" ;;
  *) echo 'Usage: nt4_mxfp8_pipeline_sweep.sh [--resume]' >&2; exit 2 ;;
esac
root=${NT4_MXFP8_PIPELINE_ROOT:-"$NT4_LOG_DIR/nt4-mxfp8-pipeline-$(date +%Y%m%d-%H%M%S)"}
if ! $resume && [[ -e "$root" ]]; then echo "Output already exists: $root" >&2; exit 1; fi
mkdir -p "$root"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
export NT4_BRANCH_SAMPLES=16 NT4_BRANCH_CLOCK=device NT4_BRANCH_TAIL=1
unset NT4_CUTLASS_MAX_BLOCKS NT4_MXFP8_WORKER_CLUSTERS NT4_CUTLASS_DECODE_PROFILE_IDS NT4_MXFP8_AB_STAGE_REDUCTION
"$PY" - <<'PY'
import torch
assert torch.cuda.device_count() >= 4, 'Need four GB300 GPUs'
for i in range(4):
    p = torch.cuda.get_device_properties(i)
    assert (p.major, p.minor, p.multi_processor_count) == (10, 3, 152), p
print('GB300 geometry verified: 152 SMs, one CTA per NT4 MXFP8 cluster')
PY
printf 'mxfp8\n' > "$root/precision.txt"
echo "Results: $root"
calibration="calibration-$(date +%Y%m%d-%H%M%S)"
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_branch_calibrate.py" \
  --output "$root/$calibration.json" > "$root/$calibration.log" 2>&1
for arm in baseline-overlap baseline-serial directstage0-serial directstage0-overlap directstage1-serial directstage1-overlap directstage2-serial directstage2-overlap; do
  if $resume && [[ -f "$root/$arm/summary.json" ]]; then
    echo "Keeping completed $arm"
    continue
  fi
  if [[ -e "$root/$arm" ]]; then
    if [[ "$arm" == baseline-overlap && -e "$root/inputs" ]]; then
      echo 'Recording incomplete; use a fresh root to avoid mixing inputs.' >&2; exit 1
    fi
    suffix="interrupted-$(date +%Y%m%d-%H%M%S)"
    mv "$root/$arm" "$root/$arm.$suffix"
    if [[ -f "$root/$arm.log" ]]; then mv "$root/$arm.log" "$root/$arm.log.$suffix"; fi
  fi
  schedule=${arm##*-}
  input_mode=replay
  if [[ "$arm" == baseline-* ]]; then
    backend=baseline
    extra=(--cutlass-decode-profile-ids 39 428)
    if [[ "$arm" == baseline-overlap ]]; then input_mode=record; fi
  else
    backend=direct
    reduction=${arm#directstage}; reduction=${reduction%-*}
    extra=(--mxfp8-worker-clusters 152 --mxfp8-ab-stage-reduction "$reduction")
  fi
  echo "Starting $arm; log=$root/$arm.log"
  "$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
    --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
    --mcore-root "$MCORE_CHECKOUT" \
    --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
    --output "$root/$arm" --precision mxfp8 --request-count 128 \
    --num-tokens-to-generate 64 --model-token-budget 2048 \
    --model-graphs --shortcut-overlap --controlled-gc \
    --branch-contention "$schedule" --branch-backend "$backend" \
    --branch-input-mode "$input_mode" --branch-inputs "$root/inputs" \
    "${extra[@]}" --rounds 2 --timeout-seconds 1800 > "$root/$arm.log" 2>&1
done
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_branch_backend_summary.py" \
  "$root" --precision mxfp8 --pipeline-sweep
