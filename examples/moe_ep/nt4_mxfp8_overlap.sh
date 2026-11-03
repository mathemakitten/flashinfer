#!/usr/bin/env bash
# Run on a free GB300 node, after syncing both checkouts.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
mode=${1:-all}
case "$mode" in all|test|e2e) ;; *) echo 'Usage: nt4_mxfp8_overlap.sh [all|test|e2e]' >&2; exit 2;; esac
root=${NT4_MXFP8_OVERLAP_ROOT:-"$NT4_LOG_DIR/nt4-mxfp8-overlap-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$root"
echo "Results: $root"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
# Keep the native baseline uncapped; this old research env controls BF16 only.
unset NT4_CUTLASS_MAX_BLOCKS
if [[ "$mode" != e2e ]]; then
  echo "Checking native staged MXFP8 experts; log=$root/stages.log"
  (
    cd "$MCORE_CHECKOUT"
    "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest -q \
      tests/unit_tests/determinism/kernels/test_shortcut_flashinfer_mxfp8.py \
      tests/unit_tests/inference/test_flashinfer_mxfp8.py::test_flashinfer_shortcut_stages_preserve_live_tensors_and_output \
      tests/unit_tests/transformer/test_transformer_config.py::test_shortcut_inference_overlap_accepts_flashinfer_mxfp8 \
      tests/unit_tests/transformer/test_transformer_config.py::test_shortcut_inference_overlap_rejects_vllm_mxfp8 \
      tests/unit_tests/transformer/test_transformer_config.py::test_shortcut_inference_overlap_rejects_flashinfer_vllm_cap
  ) > "$root/stages.log" 2>&1
fi
if [[ "$mode" != test ]]; then
  "$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
    --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
    --mcore-root "$MCORE_CHECKOUT" \
    --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
    --output "$root/e2e" --precision mxfp8 --request-count 128 \
    --num-tokens-to-generate "${NT4_GENERATE_TOKENS:-64}" --model-token-budget 2048 \
    --model-graphs --shortcut-overlap --controlled-gc \
    --rounds "${NT4_ROUNDS:-2}" --timeout-seconds 7200
fi
