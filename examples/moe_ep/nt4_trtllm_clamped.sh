#!/usr/bin/env bash
# Local changes must be synced to both source trees before running on GB300.
set -euo pipefail
: "${TRTLLM_GEN_ROOT:?Set TRTLLM_GEN_ROOT}"
: "${FLASHINFER_CHECKOUT:?Set FLASHINFER_CHECKOUT}"
: "${MCORE_CHECKOUT:?Set MCORE_CHECKOUT}"
: "${NT4_LOG_DIR:?Set NT4_LOG_DIR}"
: "${PY:?Set PY to the GPU environment Python}"
export NT4_TRT_EXPORT_CONFIG="${NT4_TRT_EXPORT_CONFIG:-$TRTLLM_GEN_ROOT/cluster/mxfp8_nt4_integration.json}"
export NT4_TRT_EXPORT_DIR="${NT4_TRT_EXPORT_DIR:-$NT4_LOG_DIR/trtllm-nt4-integration-$(date +%Y%m%d-%H%M%S)}"
# Use a fresh export directory. No shipped cache entries are replaced.
printf 'Export: %s\n' "$NT4_TRT_EXPORT_DIR"
bash "$TRTLLM_GEN_ROOT/cluster/export_static_persistent_cubins.sh" export \
  > "${NT4_TRT_EXPORT_DIR}.log" 2>&1
export FLASHINFER_TRTLLM_NT4_EXPORT="$NT4_TRT_EXPORT_DIR"
export PYTHONPATH="$FLASHINFER_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_trtllm_surrogate.py" \
  --artifact "${NT4_ARTIFACT:-$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901}" \
  --mcore-root "$MCORE_CHECKOUT" --clamp-scale 16 \
  > "$NT4_TRT_EXPORT_DIR/flashinfer-preflight.log" 2>&1
printf 'NT4 FlashInfer preflight complete: %s/flashinfer-preflight.log\n' "$NT4_TRT_EXPORT_DIR"
