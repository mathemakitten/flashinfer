#!/usr/bin/env bash
# Untimed same-prefix attribution using completed generation, not a new speed run.
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

source="$NT4_LOG_DIR/nt4-tanh-answer-screen-20261102-074414"
root="$NT4_LOG_DIR/nt4-tanh-answer-resumed-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$root/batch3"
echo "Results: $root"
ln -s "$source/data" "$root/data"
for batch in 0 1 2; do ln -s "$source/batch$batch" "$root/batch$batch"; done
for item in baseline-start baseline-start-command.json baseline-start.log; do ln -s "$source/batch3/$item" "$root/batch3/$item"; done
printf 'Resumed batch spans nodes; use for accuracy only. Original: %s\n' "$source" > "$root/resume-note.txt"
"$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_accuracy_resume_batch.py" \
 --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
 --mcore-root "$MCORE_CHECKOUT" --prompt-file "$source/data/prompts3.jsonl" \
 --output "$root/batch3" --precision mxfp8 --request-count 128 \
 --num-tokens-to-generate 2048 --model-token-budget 2048 \
 --model-graphs --shortcut-overlap --controlled-gc --mxfp8-nvls-hybrid \
 --cutlass-decode-profile-ids 39 428 \
 --hardware-tanh-confirm --tanh-expert-exploratory-timing --tanh-output-exploratory-timing \
 --rounds 1 --timeout-seconds 2400 > "$root/batch3.log" 2>&1
score_env=$(mktemp -d /tmp/nt4-answer-score-resume.XXXXXX)
"$PY" -m venv --system-site-packages "$score_env"
"$score_env/bin/python" -m pip install 'math-verify==0.8.0' > "$root/scoring-install.log" 2>&1
"$score_env/bin/python" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_tanh_accuracy_score.py" --data "$root/data" --root "$root" > "$root/scoring.log" 2>&1
# Keep diagnostics in a writable directory; prior batches remain read-only symlinks.
mkdir "$root/prefix-source"
for item in baseline-start hybrid-control-start hybrid-hardware-tanh hybrid-control-end baseline-end hybrid-hardware-tanh-command.json; do ln -s "$source/batch0/$item" "$root/prefix-source/$item"; done
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_tanh_compare_outputs.py" --root "$root/prefix-source" --output "$root/prefix-source/tanh-output-comparison.json" --rounds 1 > "$root/token-comparison.log" 2>&1
if "$PY" -c 'import json,sys;sys.exit(not bool(json.load(open(sys.argv[1]))["comparisons"][0]["differences"]))' "$root/prefix-source/tanh-output-comparison.json"; then
 "$PY" -u "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_tanh_accuracy_logits.py" --source "$root/prefix-source" --output "$root/prefix-logits" --mcore-root "$MCORE_CHECKOUT" > "$root/prefix-logits.log" 2>&1
fi
echo "Completed: $root"
