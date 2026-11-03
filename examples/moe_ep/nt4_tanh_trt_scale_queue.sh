#!/usr/bin/env bash
# Two fresh-process candidate orders. Stop on any failure; no polling.
set -euo pipefail
requests="${1:?usage: nt4_tanh_trt_scale_queue.sh 512|768}"
case "$requests" in 512) orders=(tanh-first trt-first) ;; 768) orders=(trt-first tanh-first) ;; *) exit 2 ;; esac
base=/lustre/fsw/portfolios/nemotron/users/helenn
script="$base/flashinfer-nt4-gb300-nvls/examples/moe_ep/nt4_tanh_trt_scale.sh"
root="$base/logs/nt4-scale-queue${requests}-$(date +%Y%m%d-%H%M%S)"
mkdir "$root"
printf 'Queue: %s\n' "$root"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv > "$root/gpus.csv"
for order in "${orders[@]}"; do
  printf 'Starting %s requests=%s; log=%s/%s.log\n' "$order" "$requests" "$root" "$order"
  # Two 100-minute maximum comparisons fit the fresh four-hour allocations.
  if timeout --signal=TERM --kill-after=30s 6000s bash "$script" "$requests" "$order" > "$root/$order.log" 2>&1; then
    printf '%s\tPASS\n' "$order" >> "$root/status.tsv"
  else
    rc=$?
    printf '%s\tFAIL\t%s\n' "$order" "$rc" >> "$root/status.tsv"
    echo "Stopping; inspect $root/$order.log" >&2
    exit "$rc"
  fi
done
printf 'Completed: %s\n' "$root"
