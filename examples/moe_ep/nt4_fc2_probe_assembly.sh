#!/usr/bin/env bash
# Matched-input diagnostic; preserves numerical gates, stops at first failure.
set -euo pipefail
internal=${1:-0}
[[ "$internal" == 0 || "$internal" == 1 ]] || exit 2
base=/lustre/fsw/portfolios/nemotron/users/helenn
export PY="${PY:-$(command -v python3)}"
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export MCORE_CHECKOUT="$base/megatron-lm-nt4-coscheduling"
export NT4_LOG_DIR="$base/logs"
export FLASHINFER_WORKSPACE_BASE="$base"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT"
export NT4_ALLOW_SOURCE_JIT=1 MAX_JOBS=4 NT4_DETACHED_TAIL=1
export NT4_BRANCH_SAMPLES=4 NT4_BRANCH_CLOCK=device NT4_BRANCH_TAIL=1
export NT4_EXPERT_PHASE_TIMING=1
unset NT4_TRT_DEBUG_INTERMEDIATES CUDA_LAUNCH_BLOCKING MEGA_USE_NCU
export NT4_EXPERT_INTERNAL_TIMING="$internal" NT4_HARDWARE_TANH=1 NT4_TANH_FC1_ADVISORY=1
export NT4_MXFP8_GROUP_HINT=1024 NT4_FC1_DIRECT_STORE=0
unset NT4_MODEL_GRAPH_SPLIT_AFTER NT4_MODEL_SYNC_BOUNDARIES NT4_CUTLASS_MAX_BLOCKS
unset NT4_MXFP8_AB_STAGE_REDUCTION NT4_MXFP8_WORKER_CLUSTERS
"$PY" -c 'import nvshmem.core' >/dev/null 2>&1 || "$PY" -m pip install --no-deps nvshmem4py-cu13==0.4.0
"$PY" -c 'import nvshmem.core; import torch, flashinfer; print("Dependency preflight PASS", flush=True)'
root="$NT4_LOG_DIR/nt4-fc2-probe-assembly-i$internal-$(date +%Y%m%d-%H%M%S)"
source_root="$NT4_LOG_DIR/nt4-trtllm-matched128-20261023-112340"
test -f "$source_root/record/summary.json"
mkdir "$root"
ln -s "$source_root/inputs" "$root/inputs"
printf 'Results: %s\n' "$root"
scratch=$(mktemp -d /tmp/nt4-fc1-assembly.XXXXXX)
printf 'Local assembly: %s\n' "$scratch"
"$PY" - <<'PYCODE' > "$root/pipeline-source.txt"
import inspect
import cutlass.pipeline as p
print('PIPELINE MODULE', p.__file__)
print(inspect.getsource(p.PipelineTmaUmma))
from cutlass.pipeline import PipelineProducer
print(inspect.getsource(PipelineProducer))
PYCODE
for variant in ${NT4_ASSEMBLY_VARIANTS:-control after}; do
  export NT4_LOOKAHEAD_AFTER_MMA=0
  export NT4_PACKED_FC1_ACTIVATION=0
  export NT4_FC2_PROBE_AFTER_ISSUE=0
  [[ "$variant" != after ]] || export NT4_FC2_PROBE_AFTER_ISSUE=1
  export NT4_KERNEL_DUMP_DIR="$scratch/$variant"
  mkdir "$root/$variant"
  printf '%s\n' "$NT4_FC2_PROBE_AFTER_ISSUE" > "$root/$variant/probe-after.txt"
  for schedule in serial; do
    arm="hybrid-$schedule"
    printf 'Starting %s/%s\n' "$variant" "$arm"
    "$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_e2e.py" \
      --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
      --mcore-root "$MCORE_CHECKOUT" \
      --prompt-file "$base/research/routing/openmath2_unique2048.jsonl" \
      --output "$root/$variant/$arm" --precision mxfp8 --request-count 128 \
      --num-tokens-to-generate 64 --model-token-budget 2048 \
      --model-graphs --shortcut-overlap --controlled-gc --mxfp8-nvls-hybrid \
      --mxfp8-worker-clusters 128 --branch-native-output --branch-contention "$schedule" --branch-backend direct \
      --branch-input-mode replay --branch-inputs "$root/inputs" \
      --rounds 1 --timeout-seconds 1800 > "$root/$variant/$arm.log" 2>&1
    printf '%s\t%s\tPASS\n' "$variant" "$arm" >> "$root/status.tsv"
  done
done
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_fc1_assembly_report.py" "$scratch"
tar -czf "$root/assembly.tar.gz" -C "$scratch" .
printf 'Complete: %s\n' "$root"
