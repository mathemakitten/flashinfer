#!/usr/bin/env bash
# Real cohorts with fixed M128 knobs and production token budget.
# Cohort-sized model budgets can prevent full admission; do not use them here.
set -euo pipefail
: "${PY:?Set PY}"
: "${FLASHINFER_CHECKOUT:?Set FLASHINFER_CHECKOUT}"
: "${MCORE_CHECKOUT:?Set MCORE_CHECKOUT}"
: "${NT4_LOG_DIR:?Set NT4_LOG_DIR}"
# Optional positional request counts allow continuing after completed cases.
if (( $# == 0 )); then
  set -- 128 256 512
fi
for requests in "$@"; do
  case "$requests" in
    128|256|512) ;;
    *) printf 'Unsupported request count: %s (use 128, 256, 512)\n' "$requests" >&2; exit 2 ;;
  esac
done
capacity_flags=()
case "${NT4_MATCH_CAPACITY:-0}" in
  0) ;;
  1) capacity_flags=(--match-production-capacity) ;;
  *) printf 'NT4_MATCH_CAPACITY must be 0 or 1\n' >&2; exit 2 ;;
esac
unset NT4_DEBUG_PREQUANT
prompt_file="${NT4_PROMPT_FILE:-/lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl}"
source_artifact="${NT4_SOURCE_ARTIFACT:-$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901}"
run_root="$NT4_LOG_DIR/nt4-m128-capacity-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_root"
printf 'Results: %s\n' "$run_root"
for requests in "$@"; do
    budget=1024
    artifact="$run_root/requests${requests}-budget${budget}"
    printf 'Starting requests=%s budget=%s; log=%s.log\n' "$requests" "$budget" "$artifact"
    "$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_three_way.py" \
      --source-artifact "$source_artifact" \
      --mcore-root "$MCORE_CHECKOUT" --artifact "$artifact" \
      --prompt-file "$prompt_file" \
      --request-count "$requests" --min-active-tokens "$requests" \
      --max-local-tokens "$requests" --model-token-budget "$budget" \
      "${capacity_flags[@]}" \
      --samples 3 --capture-interval 8 --ignore-eos \
      --knobs-json '{"mma_tiler_mnk":[128,256,128],"cluster_shape_mnk":[1,1,1],"use_2cta_instrs":false,"group_hint":1024,"flag_batch":1}' \
      >"${artifact}.log" 2>&1
    # Check actual allocations, not just requested budgets. Never quietly label
    # a rounded-up or overridden production capacity as a matched comparison.
    "$PY" - "$artifact/manifest.json" "$requests" "$budget" "${NT4_MATCH_CAPACITY:-0}" <<'PY'
import json
import sys
from pathlib import Path
path, requests, budget = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = json.loads(Path(path).read_text())
assert m.get('capture_complete'), f'incomplete capture: {path}'
assert len(m['ranks']) == 4, 'expected EP4'
for rank in m['ranks']:
    assert len(rank['results']) == 3, 'expected three samples per rank'
    for result in rank['results']:
        audit = result['baseline_audit']
        assert audit['nvls_capacity_per_rank'] == budget, audit
        expected_capacity = budget if sys.argv[4] == '1' else requests
        assert audit['candidate_buffer_rows_per_rank'] == expected_capacity, audit
        assert result['tokens_per_rank'] == [requests] * 4, result['tokens_per_rank']
        assert all(step['decode_only'] and step['active_tokens'] == requests
                   for step in result['engine_steps']), result['engine_steps']
        assert result['correctness'] == 'PASS' and result['benchmark'], result
print(json.dumps(dict(stage='capacity_audit_pass', requests=requests, budget=budget, manifest=path)))
PY
done
printf 'Complete: %s\n' "$run_root"
