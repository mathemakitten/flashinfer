#!/usr/bin/env bash
set -euo pipefail
case "$(hostname -s)" in nvl*) ;; *) echo 'Requires CMH compute node' >&2; exit 2 ;; esac
python3 -c 'import torch; assert torch.cuda.device_count() == 4, "Expected four visible GPUs"' 
base=/lustre/fsw/portfolios/nemotron/users/helenn
export PY="${PY:-$(command -v python3)}"
export FLASHINFER_CHECKOUT="$base/flashinfer-nt4-gb300-nvls"
export MCORE_CHECKOUT="$base/megatron-lm-nt4-coscheduling"
export NT4_LOG_DIR="$base/logs" FLASHINFER_WORKSPACE_BASE="$base"
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT"
export FLASHINFER_TRTLLM_NT4_EXPORT="$base/logs/nt4-trtllm-clamped-sweep-20261020-134327/export"
export NT4_ALLOW_SOURCE_JIT=1 MAX_JOBS=4 NT4_DETACHED_TAIL=1 NT4_TE_STREAM_WORKSPACES=1
export NT4_LOOKAHEAD_AFTER_MMA=0 NT4_PACKED_FC1_ACTIVATION=0 NT4_MXFP8_GROUP_HINT=1024
unset NT4_EXPERT_PHASE_TIMING NT4_EXPERT_INTERNAL_TIMING NT4_EXPERT_SPARSE_TIMING NT4_KERNEL_DUMP_DIR
unset NT4_FC1_FAILURE_DIR NT4_HARDWARE_TANH NT4_HARDWARE_TANH_LAYERS NT4_TANH_FC1_ADVISORY
unset NT4_BRANCH_SAMPLES NT4_BRANCH_CLOCK NT4_BRANCH_TAIL NT4_BRANCH_NATIVE_OUTPUT
unset NT4_TRT_DEBUG_INTERMEDIATES CUDA_LAUNCH_BLOCKING MEGA_USE_NCU NT4_EXPERT_STAGE_CAPTURE
unset NT4_MODEL_GRAPH_SPLIT_AFTER NT4_MODEL_SYNC_BOUNDARIES NT4_CUTLASS_MAX_BLOCKS
unset NT4_MXFP8_AB_STAGE_REDUCTION NT4_MXFP8_WORKER_CLUSTERS NT4_PIPELINE_REDUCE_SIDE
unset NT4_FC1_DIRECT_STORE NT4_FC2_PROBE_AFTER_ISSUE NT4_SCHED_LOOKUP
"$PY" -c 'import nvshmem.core' >/dev/null 2>&1 || "$PY" -m pip install --no-deps nvshmem4py-cu13==0.4.0
unset NT4_MATCHED_FORWARD_MODE NT4_MATCHED_FORWARD_ROOT NT4_MATCHED_FORWARD_SAMPLES
unset NT4_TANH_LOGIT_SOURCE NT4_EXPERT_FAILURE_DIR NT4_TANH_EXPERT_ADVISORY NT4_TANH_OUTPUT_ADVISORY
export NT4_SCHED_LOOKUP=1

root="$NT4_LOG_DIR/nt4-tanh-fp64-audit-$(date +%Y%m%d-%H%M%S)"
mkdir "$root"
echo "Results: $root"
source="$NT4_LOG_DIR/nt4-lookup-tanh-failure256-20261030-142740"
"$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_tanh_fp64_audit.py" \
 --snapshot "$source/fc1-failure/layer14/expert180-rank0.pt" \
 --snapshot "$source/fc1-failure/layer7/expert234-rank0.pt" \
 --snapshot "$source/fc1-failure/layer7/expert236-rank0.pt" \
 --output "$root/fp64" > "$root/fp64.log" 2>&1
"$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_tanh_saved_expert.py" \
 --snapshot "$source/fc1-failure/layer14/expert180-rank0.pt" \
 --token "$source/expert-failure/layer14-row155-rank0.pt" \
 --output "$root/weighted-fc2" > "$root/weighted-fc2.log" 2>&1
echo "Completed: $root"
