#!/usr/bin/env bash
# Matched-input hybrid pipeline footprint diagnostic. Sequential, with no filesystem polling.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
source_root="${NT4_MATCHED_SOURCE:-$NT4_LOG_DIR/nt4-trtllm-matched128-20261023-112340}"
[[ -f "$source_root/record/summary.json" && -d "$source_root/inputs" ]] || {
  echo "Missing completed input recording: $source_root" >&2; exit 1;
}
root="${NT4_HYBRID_PIPELINE_ROOT:-$NT4_LOG_DIR/nt4-hybrid-pipeline-$(date +%Y%m%d-%H%M%S)}"
mkdir "$root"
ln -s "$source_root/inputs" "$root/inputs"
export NT4_BRANCH_SAMPLES=16 NT4_BRANCH_CLOCK=device NT4_BRANCH_TAIL=1
# Retain coarse phase clocks; omit invasive warp/FC timing instrumentation.
export NT4_EXPERT_PHASE_TIMING=1 NT4_ALLOW_SOURCE_JIT=1 NT4_DETACHED_TAIL=0
unset NT4_EXPERT_INTERNAL_TIMING NT4_TRT_DEBUG_INTERMEDIATES CUDA_LAUNCH_BLOCKING
unset NT4_CUTLASS_MAX_BLOCKS NT4_MXFP8_AB_STAGE_REDUCTION NT4_MXFP8_WORKER_CLUSTERS
printf 'Hybrid pipeline sweep: %s\n' "$root"
failed=0
for reduction in 0 1 2; do
  for schedule in serial overlap; do
    arm="hybridstage${reduction}-${schedule}"
    printf 'Starting %s; log=%s/%s.log\n' "$arm" "$root" "$arm"
    if "$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
      --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
      --mcore-root "$MCORE_CHECKOUT" \
      --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
      --output "$root/$arm" --precision mxfp8 --request-count 128 \
      --num-tokens-to-generate 64 --model-token-budget 2048 \
      --model-graphs --shortcut-overlap --controlled-gc --mxfp8-nvls-hybrid \
      --branch-native-output --branch-contention "$schedule" --branch-backend direct \
      --branch-input-mode replay --branch-inputs "$root/inputs" \
      --mxfp8-worker-clusters 152 --mxfp8-ab-stage-reduction "$reduction" --rounds 2 --timeout-seconds 1800 \
      > "$root/$arm.log" 2>&1; then
      printf '%s\tPASS\n' "$arm" >> "$root/status.tsv"
    else
      rc=$?
      printf '%s\tFAIL\t%s\n' "$arm" "$rc" >> "$root/status.tsv"
      failed=1
      # A failed CUDA process may leave this node unhealthy. Preserve the remaining queue.
      echo "Stopping after $arm failed ($rc); inspect its log before continuing." >&2
      break 2
    fi
  done
done
printf 'Finished; statuses: %s/status.tsv\n' "$root"
exit "$failed"
