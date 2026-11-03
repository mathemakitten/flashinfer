#!/usr/bin/env bash
# Diagnostic only: force the native block-input sequence before each timed fork.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
precision=${NT4_BRANCH_PRECISION:-bf16}
case "$precision" in bf16) native=vllm ;; mxfp8) native=baseline ;; *) echo "Invalid precision: $precision" >&2; exit 2 ;; esac
resume=false
case "${1:-}" in
  "") ;;
  --resume) resume=true; : "${NT4_BRANCH_BACKEND_ROOT:?set the existing experiment root for --resume}" ;;
  *) echo 'Usage: nt4_branch_backend.sh [--resume]' >&2; exit 2 ;;
esac
root=${NT4_BRANCH_BACKEND_ROOT:-"$NT4_LOG_DIR/nt4-branch-backend-$precision-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$root"
if [[ -f "$root/precision.txt" && "$(cat "$root/precision.txt")" != "$precision" ]]; then
  echo 'Experiment precision differs; choose a new root.' >&2; exit 1
fi
if [[ "$precision" == mxfp8 && -d "$root/inputs" && ! -f "$root/precision.txt" ]]; then
  echo 'Existing inputs lack MXFP8 provenance; choose a new root.' >&2; exit 1
fi
printf '%s\n' "$precision" > "$root/precision.txt"
echo "Results: $root"
export NT4_BRANCH_SAMPLES=16 NT4_BRANCH_CLOCK=device NT4_BRANCH_TAIL=1
calibration=calibration
if $resume; then calibration="calibration-resume-$(date +%Y%m%d-%H%M%S)-$$"; fi
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_branch_calibrate.py" \
  --output "$root/$calibration.json" > "$root/$calibration.log" 2>&1
printf '{"group_hint":1024,"max_active_clusters":74}\n' > "$root/knobs74.json"
extra=()
if [[ "$precision" == bf16 ]]; then extra+=(--bf16-knobs-json "$root/knobs74.json"); fi
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
unset NT4_CUTLASS_MAX_BLOCKS
for arm in "$native-overlap" direct-overlap direct-serial "$native-serial"; do
  if $resume; then
    if [[ -f "$root/$arm/summary.json" ]]; then
      echo "Keeping completed $arm"
      continue
    fi
    if [[ -e "$root/$arm" ]]; then
      suffix="interrupted-$(date +%Y%m%d-%H%M%S)-$$"
      mv "$root/$arm" "$root/$arm.$suffix"
      if [[ -f "$root/$arm.log" ]]; then mv "$root/$arm.log" "$root/$arm.log.$suffix"; fi
    fi
    if [[ "$arm" == "$native-overlap" && -e "$root/inputs" ]]; then
      echo 'Baseline recording is incomplete; use a fresh experiment root.' >&2
      exit 1
    fi
  fi
  backend=${arm%-*}
  schedule=${arm##*-}
  input_mode=replay
  if [[ "$arm" == "$native-overlap" ]]; then input_mode=record; fi
  echo "Starting $arm; log=$root/$arm.log"
  "$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
    --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
    --mcore-root "$MCORE_CHECKOUT" \
    --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
    --output "$root/$arm" --precision "$precision" --request-count 128 \
    --num-tokens-to-generate 64 --model-token-budget 2048 \
    --model-graphs --shortcut-overlap --controlled-gc \
    "${extra[@]}" \
    --branch-contention "$schedule" --branch-backend "$backend" \
    --branch-input-mode "$input_mode" --branch-inputs "$root/inputs" \
    --rounds 2 --timeout-seconds 1800 > "$root/$arm.log" 2>&1
done
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_branch_backend_summary.py" "$root" --precision "$precision"
