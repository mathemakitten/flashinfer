#!/usr/bin/env bash
# Isolated export only; never install over artifacts used by other queues.
set -euo pipefail
base=/lustre/fsw/portfolios/nemotron/users/helenn
repo="$base/trtllm-gen"
fi="$base/flashinfer-nt4-gb300-nvls"
export PY="$(command -v python3)"
export NT4_TRT_EXPORT_CONFIG="$fi/examples/moe_ep/nt4_fc2_scheduler_twins.json"
export NT4_TRT_EXPORT_DIR="$base/logs/nt4-fc2-scheduler-twins-$(date +%Y%m%d-%H%M%S)"
export NT4_TRT_BUILD_DIR="$repo/build-gb300-public133"
export NT4_TRT_BUILD_JOBS=4 TLLM_EXPORT_JOBS=1 CUDA_PATH=/usr/local/cuda
export MAX_JOBS=4
unset NT4_EXPERT_SPARSE_TIMING NT4_EXPERT_INTERNAL_TIMING NT4_PIPELINE_REDUCE_SIDE
"$PY" -m pip install --no-deps nvshmem4py-cu13==0.4.0
printf 'Export root: %s\n' "$NT4_TRT_EXPORT_DIR"
# Use the existing build; exporter records executable/config provenance.
# A failed harness check stops export. Timing claims require matched NT4 replay.
timeout --kill-after=30s 7200 bash "$repo/cluster/export_static_persistent_cubins.sh" export \
 > "${NT4_TRT_EXPORT_DIR}.log" 2>&1
printf 'Export/harness PASS: %s\n' "$NT4_TRT_EXPORT_DIR"
