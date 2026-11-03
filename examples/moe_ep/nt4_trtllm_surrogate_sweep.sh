#!/usr/bin/env bash
# Matched-input timing only. TRTLLM uses unclamped Relu2, not the NT4 activation.
set -euo pipefail
: "${PY:?}" "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}" "${NT4_TRT_SURROGATE_ROOT:?}"
root="${NT4_SURROGATE_COMPARISON_ROOT:-$NT4_TRT_SURROGATE_ROOT/comparison-native-output}"
runner=${NT4_SURROGATE_RUNNER:-"$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py"}
inputs=${NT4_SURROGATE_INPUTS:-"$NT4_LOG_DIR/nt4-mxfp8-hybrid-20261019-140037/inputs"}
[[ ! -e "$root" ]] || { echo "Output already exists: $root" >&2; exit 1; }
[[ -f "$inputs/rank0/round1-step0.pt" ]] || { echo "Missing existing matched inputs: $inputs" >&2; exit 1; }
# Refuse to start the expensive model runs unless all four preflight ranks passed.
"$PY" - "$NT4_TRT_SURROGATE_ROOT/preflight.log" <<'PY'
import json,sys
text=open(sys.argv[1]).read()
dec=json.JSONDecoder(); passed=set()
for line in text.splitlines():
    for i,c in enumerate(line):
        if c!='{': continue
        try: r,_=dec.raw_decode(line[i:])
        except ValueError: continue
        if r.get('stage')=='surrogate_preflight' and r.get('status')=='PASS': passed.add(r['rank'])
assert passed=={0,1,2,3}, f'Preflight incomplete: {passed}'
PY
mkdir -p "$root"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_MODULE_LOADING=EAGER TORCHINDUCTOR_COMPILE_THREADS=1
export NT4_BRANCH_SAMPLES=16 NT4_BRANCH_CLOCK=device NT4_BRANCH_TAIL=1
unset NT4_CUTLASS_MAX_BLOCKS NT4_MXFP8_WORKER_CLUSTERS NT4_CUTLASS_DECODE_PROFILE_IDS NT4_MXFP8_AB_STAGE_REDUCTION NT4_TRTLLM_RELU2_SURROGATE
printf 'Matched inputs: %s\nResults: %s\n' "$inputs" "$root"
for arm in baseline-overlap baseline-serial hybrid-serial trtllm-serial hybrid-overlap trtllm-overlap baseline-end; do
  schedule=${arm##*-}
  [[ "$arm" != baseline-end ]] || schedule=overlap
  backend=direct
  extra=(--mxfp8-nvls-hybrid)
  if [[ "$arm" == baseline-* ]]; then
    backend=baseline
    extra=(--cutlass-decode-profile-ids 39 428)
  elif [[ "$arm" == trtllm-* ]]; then
    extra+=(--trtllm-relu2-surrogate)
  fi
  printf 'Starting %s; log=%s/%s.log\n' "$arm" "$root" "$arm"
  "$PY" "$runner" \
    --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
    --mcore-root "$MCORE_CHECKOUT" \
    --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
    --output "$root/$arm" --precision mxfp8 --request-count 128 \
    --num-tokens-to-generate 64 --model-token-budget 2048 \
    --model-graphs --shortcut-overlap --controlled-gc \
    --branch-contention "$schedule" --branch-backend "$backend" \
    --branch-input-mode replay --branch-inputs "$inputs" --branch-native-output \
    "${extra[@]}" --rounds 2 --timeout-seconds 1800 > "$root/$arm.log" 2>&1
done
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_trtllm_surrogate_summary.py" "$root"
