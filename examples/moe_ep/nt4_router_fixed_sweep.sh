#!/usr/bin/env bash
# Reuse the validated clamped TRTLLM export; no export rebuild.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
export FLASHINFER_TRTLLM_NT4_EXPORT="${FLASHINFER_TRTLLM_NT4_EXPORT:-$NT4_LOG_DIR/nt4-trtllm-clamped-sweep-20261020-134327/export}"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
export NT4_ROUTER_SWEEP_ROOT="${NT4_ROUTER_SWEEP_ROOT:-$NT4_LOG_DIR/nt4-router-fixed-sweep-$(date +%Y%m%d-%H%M%S)}"
[[ -d "$FLASHINFER_TRTLLM_NT4_EXPORT" ]] || { echo "Missing TRTLLM export: $FLASHINFER_TRTLLM_NT4_EXPORT" >&2; exit 1; }
[[ ! -e "$NT4_ROUTER_SWEEP_ROOT" ]] || { echo "Use a fresh output: $NT4_ROUTER_SWEEP_ROOT" >&2; exit 1; }
# Validate source without importing CUDA or compiling anything in this process.
"$PY" - <<'PY'
import ast, os
from pathlib import Path
p = Path(os.environ['MCORE_CHECKOUT']) / 'megatron/core/transformer/moe/router.py'
tree = ast.parse(p.read_text())
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_compiled_topk_routing')
assert any(isinstance(d, ast.Call) and ast.unparse(d.func) == 'torch.compile'
           and any(k.arg == 'dynamic' and isinstance(k.value, ast.Constant) and k.value.value is True for k in d.keywords)
           for d in fn.decorator_list), 'Sync the dynamic=True MCore router fix first'
PY
mkdir -p "$NT4_ROUTER_SWEEP_ROOT"
printf 'Results: %s\n' "$NT4_ROUTER_SWEEP_ROOT"
# Preserve the stream-workspace setting from the passing full-model validation.
export NT4_TE_STREAM_WORKSPACES=1
unset NT4_NATIVE_CUTLASS_SERIAL_AUDIT NT4_BRANCH_INPUTS NT4_BRANCH_INPUT_MODE NT4_BRANCH_NATIVE_OUTPUT
unset NT4_TRTLLM_RELU2_SURROGATE NT4_CUTLASS_MAX_BLOCKS NT4_MXFP8_WORKER_CLUSTERS NT4_MXFP8_AB_STAGE_REDUCTION
unset NT4_SURROGATE_RUNNER NT4_SURROGATE_COMPARISON_ROOT
(
  cd "$MCORE_CHECKOUT"
  "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest -q \
    tests/unit_tests/determinism/kernels/test_inference_router_shapes.py
) > "$NT4_ROUTER_SWEEP_ROOT/router-regression.log" 2>&1
counts=("$@")
if [[ ${#counts[@]} == 0 ]]; then counts=(32 128 256 512); fi
for requests in "${counts[@]}"; do
  case "$requests" in 32|128|256|512) ;; *) echo "Expected 32, 128, 256, or 512 requests" >&2; exit 2;; esac
  printf 'Starting requests=%s; log=%s/requests%s.log\n' "$requests" "$NT4_ROUTER_SWEEP_ROOT" "$requests"
  "$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
    --source-artifact "${NT4_ARTIFACT:-$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901}" \
    --mcore-root "$MCORE_CHECKOUT" \
    --prompt-file "${NT4_PROMPT_FILE:-/lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl}" \
    --output "$NT4_ROUTER_SWEEP_ROOT/requests$requests" \
    --precision mxfp8 --request-count "$requests" --model-token-budget 2048 \
    --num-tokens-to-generate 512 --rounds 2 --timeout-seconds 7200 \
    --model-graphs --shortcut-overlap --controlled-gc \
    --mxfp8-nvls-hybrid --trtllm-clamped-compare \
    --cutlass-decode-profile-ids 39 428 \
    > "$NT4_ROUTER_SWEEP_ROOT/requests$requests.log" 2>&1
done
printf 'Complete: %s\n' "$NT4_ROUTER_SWEEP_ROOT"
