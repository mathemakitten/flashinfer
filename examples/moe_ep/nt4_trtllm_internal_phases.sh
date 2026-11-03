#!/usr/bin/env bash
# Diagnostic only: internal markers perturb scheduling. Preserve uninstrumented controls.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
: "${FLASHINFER_TRTLLM_NT4_EXPORT:?}"
source_root="${NT4_MATCHED_SOURCE:-$NT4_LOG_DIR/nt4-trtllm-matched128-20261023-112340}"
[[ -f "$source_root/record/summary.json" && -d "$source_root/inputs" ]] || {
  echo "Missing matched inputs/record at $source_root" >&2; exit 1;
}
export NT4_MATCHED_ROOT="${NT4_INTERNAL_PHASE_ROOT:-$NT4_LOG_DIR/nt4-trtllm-internal-phases128-$(date +%Y%m%d-%H%M%S)}"
mkdir "$NT4_MATCHED_ROOT"
ln -s "$source_root/inputs" "$NT4_MATCHED_ROOT/inputs"
ln -s "$source_root/record" "$NT4_MATCHED_ROOT/record"
export NT4_EXPERT_PHASE_TIMING=1 NT4_EXPERT_INTERNAL_TIMING=1
unset NT4_TRT_DEBUG_INTERMEDIATES
printf 'Internal phase diagnostic: %s\n' "$NT4_MATCHED_ROOT"
bash "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_trtllm_matched.sh"
