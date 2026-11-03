#!/usr/bin/env bash
# Resource-retirement prototype: qualify before measuring matched branch overlap.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
root="${NT4_DETACHED_ROOT:-$NT4_LOG_DIR/nt4-detached-tail-$(date +%Y%m%d-%H%M%S)}"
source_root="${NT4_MATCHED_SOURCE:-$NT4_LOG_DIR/nt4-trtllm-matched128-20261023-112340}"
artifact="${NT4_ARTIFACT:-$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901}"
[[ -d "$source_root/inputs" ]] || { echo "Missing matched inputs: $source_root"; exit 1; }
mkdir "$root"
printf 'Detached tail results: %s\n' "$root"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
unset NT4_EXPERT_INTERNAL_TIMING NT4_TRT_DEBUG_INTERMEDIATES CUDA_LAUNCH_BLOCKING MEGA_USE_NCU
unset NT4_MODEL_GRAPH_SPLIT_AFTER NT4_MODEL_SYNC_BOUNDARIES
export NT4_ALLOW_SOURCE_JIT=1 NT4_DETACHED_TAIL=1
# Independent numerical gates on the EP4 direct path (cross-PE cleanup), then
# local-expert hybrid graph buckets / changed routing / back-to-back replay.
knobs='{"mma_tiler_mnk":[128,256,128],"cluster_shape_mnk":[1,1,1],"use_2cta_instrs":false,"group_hint":1024,"flag_batch":1}'
timeout --kill-after=30s 1800s "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_checkpoint_replay.py" \
  --artifact "$artifact" --graph --routing-weight-placement after_fc2 --knobs-json "$knobs" \
  > "$root/direct-qualification.log" 2>&1
timeout --kill-after=30s 1800s "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_hybrid_graph_repro.py" \
  --artifact "$artifact" --mcore-root "$MCORE_CHECKOUT" --layers 3 --replays 8 --shared-pool \
  > "$root/hybrid-qualification.log" 2>&1
export NT4_BRANCH_SAMPLES=16 NT4_BRANCH_CLOCK=device NT4_BRANCH_TAIL=1 NT4_EXPERT_PHASE_TIMING=1
for arm in original-start detached original-end; do
  export NT4_DETACHED_TAIL=0
  if [[ "$arm" == detached ]]; then export NT4_DETACHED_TAIL=1; fi
  for schedule in serial overlap; do
    echo "Starting $arm-$schedule"
    "$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
      --source-artifact "$artifact" --mcore-root "$MCORE_CHECKOUT" \
      --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
      --output "$root/$arm-$schedule" --precision mxfp8 --request-count 128 \
      --num-tokens-to-generate 64 --model-token-budget 2048 --rounds 2 --timeout-seconds 1800 \
      --model-graphs --shortcut-overlap --controlled-gc --mxfp8-nvls-hybrid \
      --branch-native-output --branch-contention "$schedule" --branch-backend direct \
      --branch-input-mode replay --branch-inputs "$source_root/inputs" \
      > "$root/$arm-$schedule.log" 2>&1
    printf '%s\t%s\tPASS\n' "$arm" "$schedule" >> "$root/status.tsv"
  done
done
echo "Complete: $root"
