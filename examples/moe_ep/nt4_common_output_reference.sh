#!/usr/bin/env bash
set -euo pipefail
: "${FLASHINFER_CHECKOUT:?}" "${MCORE_CHECKOUT:?}" "${NT4_LOG_DIR:?}"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
export FLASHINFER_TRTLLM_NT4_EXPORT="${FLASHINFER_TRTLLM_NT4_EXPORT:-$NT4_LOG_DIR/nt4-trtllm-clamped-sweep-20261020-134327/export}"
export NT4_ALLOW_SOURCE_JIT=1
export MAX_JOBS=4 OMP_NUM_THREADS=1
unset NT4_DETACHED_TAIL NT4_EXPERT_PHASE_TIMING NT4_EXPERT_INTERNAL_TIMING
root="$NT4_LOG_DIR/nt4-common-output-reference-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$root"
printf 'Results: %s\n' "$root"
for requests in 128 256 512; do
    if [[ "$requests" == 128 ]]; then
        artifact="$NT4_LOG_DIR/nt4-m128-capacity-20261013-140420/requests128-budget1024"
    else
        artifact="$NT4_LOG_DIR/nt4-m128-capacity-20261013-151032/requests${requests}-budget1024"
    fi
    printf 'Starting %s; log=%s/requests%s.log\n' "$requests" "$root" "$requests"
    timeout --signal=TERM --kill-after=30s 1800s "${PY:-python3}" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=4 \
        "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_common_output_reference.py" \
        --artifact "$artifact" --mcore-root "$MCORE_CHECKOUT" \
        --output "$root/requests${requests}.json" > "$root/requests${requests}.log" 2>&1
done
printf 'Complete: %s\n' "$root"
