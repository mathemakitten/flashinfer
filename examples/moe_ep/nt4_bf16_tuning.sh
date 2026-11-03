#!/usr/bin/env bash
# Independent phases can run across node leases; all artifacts stay on Lustre.
set -euo pipefail
: "${PY:?Set PY to the node Python executable}"
: "${FLASHINFER_CHECKOUT:?Set FLASHINFER_CHECKOUT}"
: "${MCORE_CHECKOUT:?Set MCORE_CHECKOUT}"
: "${NT4_LOG_DIR:?Set NT4_LOG_DIR}"
: "${NT4_BF16_TUNE_ROOT:?Set a NEW experiment path, and reuse it across phases}"
phase="${1:-capture}"
mkdir -p "$NT4_BF16_TUNE_ROOT"
examples="$FLASHINFER_CHECKOUT/examples/moe_ep"
prompts="${NT4_PROMPT_FILE:-/lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl}"
source_artifact="${NT4_SOURCE_ARTIFACT:-$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901}"

run_e2e() {
    local requests="$1"
    local output="$2"
    shift 2
    "$PY" "$examples/nt4_e2e.py" \
        --source-artifact "$source_artifact" --mcore-root "$MCORE_CHECKOUT" \
        --prompt-file "$prompts" --output "$output" \
        --precision bf16 --request-count "$requests" --model-token-budget 2048 \
        --routed-graphs --timeout-seconds 7200 "$@"
}

case "$phase" in
capture)
    for requests in 128 256; do
        if [[ -f "$NT4_BF16_TUNE_ROOT/capture$requests/bf16-capture/manifest.json" ]]; then
            echo "Already captured $requests; keeping artifact"
            continue
        fi
        echo "Capturing $requests; log=$NT4_BF16_TUNE_ROOT/capture$requests.log"
        run_e2e "$requests" "$NT4_BF16_TUNE_ROOT/capture$requests" \
            --capture-bf16-only --capture-layers 7,32,68 --capture-samples 2 --rounds 1 \
            > "$NT4_BF16_TUNE_ROOT/capture$requests.log" 2>&1
    done
    ;;
profile)
    command -v nsys >/dev/null
    for requests in 128 256; do
        nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
            --capture-range-end=stop --cuda-graph-trace=node \
            --output "$NT4_BF16_TUNE_ROOT/profile$requests" \
            "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
            "$examples/nt4_bf16_tune.py" --worker --profile --layers 7 \
            --artifact "$NT4_BF16_TUNE_ROOT/capture$requests/bf16-capture" \
            > "$NT4_BF16_TUNE_ROOT/profile$requests.log" 2>&1
    done
    ;;
tune|resume)
    resume_args=()
    if [[ "$phase" == resume ]]; then resume_args=(--resume); fi
    "$PY" "$examples/nt4_bf16_tune.py" \
        --artifact "$NT4_BF16_TUNE_ROOT/capture128/bf16-capture" \
        --artifact "$NT4_BF16_TUNE_ROOT/capture256/bf16-capture" \
        --output "$NT4_BF16_TUNE_ROOT/tuning" --timeout-seconds 1800 "${resume_args[@]}" \
        >> "$NT4_BF16_TUNE_ROOT/tuning.log" 2>&1
    ;;
confirm)
    knobs="$NT4_BF16_TUNE_ROOT/tuning/candidate-knobs.json"
    if [[ ! -f "$knobs" ]]; then
        echo "No candidate cleared the cross-sample improvement gate. Inspect recommendation.json."
        exit 1
    fi
    for requests in 128 256; do
        run_e2e "$requests" "$NT4_BF16_TUNE_ROOT/confirm$requests" \
            --rounds 5 --bf16-knobs-json "$knobs" \
            > "$NT4_BF16_TUNE_ROOT/confirm$requests.log" 2>&1
    done
    ;;
*) echo "Usage: bash $0 {capture|profile|tune|resume|confirm}" >&2; exit 2 ;;
esac
