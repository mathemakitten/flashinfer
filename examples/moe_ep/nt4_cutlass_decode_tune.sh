#!/usr/bin/env bash
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
: "${NT4_CUTLASS_ARTIFACT:?Set a complete real NT4 MXFP8 checkpoint-layer artifact}"
root=${NT4_CUTLASS_TUNE_ROOT:-"$NT4_LOG_DIR/nt4-cutlass-decode-tune-$(date +%Y%m%d-%H%M%S)"}
export PYTHONPATH="$FLASHINFER_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
unset NT4_CUTLASS_MAX_BLOCKS
mkdir -p "$(dirname "$root")"
echo "Results: $root; log=$root.log"
timeout --signal=TERM --kill-after=30s 7200s \
  "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_cutlass_decode_tune.py" \
  --artifact "$NT4_CUTLASS_ARTIFACT" --output "$root" \
  --warmup 10 --iterations 50 --rounds 3 > "$root.log" 2>&1
