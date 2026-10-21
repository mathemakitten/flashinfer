"""Opt-in baseline diagnostics; no Nsight, no synchronization inside forward."""

import hashlib
import gc
import threading
import json
import os
from pathlib import Path
import socket
import resource
import subprocess
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class FirstForwardAudit:
    """Host-only spans from generation entry through the first forward entry.

    Nested spans overlap; do not add their durations. CPU counters distinguish
    CPU work from off-CPU waits, but cannot identify a CUDA wait on their own.
    """

    def __init__(self):
        self.hooks = []
        self.active = False
        self.records = []
        self.gc_records = []
        self.gc_active = False
        self.gc_pending = {}
        self.gc_callback = self._gc_callback
        gc.callbacks.append(self.gc_callback)

    def _gc_callback(self, phase, info):
        if not self.gc_active:
            return
        key = (threading.get_ident(), info["generation"])
        if phase == "start":
            self.gc_pending[key] = self.snapshot()
        elif phase == "stop":
            start = self.gc_pending.pop(key, None)
            if start is not None:
                self.gc_records.append(
                    dict(
                        stage="python_gc",
                        thread_id=key[0],
                        generation=key[1],
                        collected=info.get("collected"),
                        uncollectable=info.get("uncollectable"),
                        start=start,
                        end=self.snapshot(),
                    )
                )

    @staticmethod
    def snapshot():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return dict(
            wall_ns=time.perf_counter_ns(),
            thread_cpu_ns=time.thread_time_ns(),
            process_cpu_ns=time.process_time_ns(),
            minor_faults=usage.ru_minflt,
            major_faults=usage.ru_majflt,
            voluntary_switches=usage.ru_nvcsw,
            involuntary_switches=usage.ru_nivcsw,
        )

    def install(self, obj, name, label):
        original = getattr(obj, name, None)
        if original is None:
            return
        had_own = name in vars(obj)
        own = vars(obj).get(name)

        def wrapped(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            before = self.snapshot()
            try:
                return original(*args, **kwargs)
            finally:
                self.records.append(
                    dict(stage=label, start=before, end=self.snapshot())
                )

        setattr(obj, name, wrapped)
        self.hooks.append((obj, name, had_own, own))

    def begin(self):
        self.gc_records = []
        self.gc_pending = {}
        self.gc_active = True
        self.records = []
        self.start = self.snapshot()
        self.active = True

    def finish(self):
        if self.active:
            self.records.append(
                dict(
                    stage="generation_to_first_forward",
                    start=self.start,
                    end=self.snapshot(),
                )
            )
        self.active = False

    def close(self):
        self.active = False
        self.gc_active = False
        if self.gc_callback in gc.callbacks:
            gc.callbacks.remove(self.gc_callback)
        for obj, name, had_own, own in reversed(self.hooks):
            if had_own:
                setattr(obj, name, own)
            else:
                delattr(obj, name)
        self.hooks = []


class VarianceAudit:
    def __init__(self, engine, destination, events=False):
        import torch
        import torch.distributed as dist
        from megatron.core.transformer.moe import experts
        import importlib

        vllm_fused_moe = importlib.import_module(
            "megatron.core.inference.moe.vllm_fused_moe"
        )

        self.torch = torch
        self.engine = engine
        self.destination = destination
        self.rank = dist.get_rank()
        self.events_enabled = events
        self.active = False
        self.records = []
        self.configs = {}
        self.telemetry = None
        self.telemetry_file = None
        self.experts = experts
        self.original_prepare = experts.prepare_vllm_moe
        self.original_forward = engine.controller._dynamic_step_forward_logits
        self.first_forward = FirstForwardAudit()
        for obj, prefix, names in (
            (engine, "engine", ("add_request", "_add_request")),
            (
                engine.controller,
                "controller",
                ("_dynamic_step_context_init", "tokenize_prompt"),
            ),
            (
                engine.context,
                "context",
                ("initialize_attention_state", "current_input_and_position_ids"),
            ),
        ):
            for name in names:
                self.first_forward.install(obj, name, f"{prefix}.{name}")
        # Patch methods, not the class object, preserving isinstance/dataclass behavior.
        from megatron.core.inference.inference_request import DynamicInferenceRequest

        self.first_forward.install(
            DynamicInferenceRequest, "__init__", "request.construct"
        )
        self.first_forward.install(
            DynamicInferenceRequest, "__post_init__", "request.post_init"
        )
        # Allocate and initialize event handles outside measured generation.
        self.pool = []
        if events:
            self.pool = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(2048)
            ]
            for pair in self.pool:
                for event in pair:
                    event.record()
            torch.cuda.synchronize()

        def prepare(*args, **kwargs):
            result = self.original_prepare(*args, **kwargs)
            record = dict(
                capturing=torch.cuda.is_current_stream_capturing(),
                num_tokens_hint=kwargs.get("num_tokens_hint"),
                grid_size_limit=kwargs.get("grid_size_limit"),
                max_tokens=result.max_tokens,
                topk=result.topk,
                config=result.config,
                grid_size_fc1=result.grid_size_fc1,
                grid_size_fc2=result.grid_size_fc2,
                fc1_shape=list(result.fc1_weight.shape),
                fc2_shape=list(result.fc2_weight.shape),
                local_expert_start=result.local_expert_start,
            )
            key = json.dumps(record, sort_keys=True)
            self.configs[key] = self.configs.get(key, 0) + 1
            return result

        def forward(*args, **kwargs):
            self.first_forward.finish()
            if not self.active:
                return self.original_forward(*args, **kwargs)
            slot = len(self.records)
            if slot >= len(self.pool):
                raise RuntimeError(
                    "variance audit event capacity exceeded (2048 forwards)"
                )
            decode = engine.context.is_decode_only()
            rows = engine.context.get_active_request_count()
            start, end = self.pool[slot]
            host_start = time.perf_counter_ns()
            start.record()
            result = self.original_forward(*args, **kwargs)
            end.record()
            self.records.append(
                dict(
                    decode=decode,
                    requests=int(rows),
                    host_start_ns=host_start,
                    host_end_ns=time.perf_counter_ns(),
                )
            )
            return result

        experts.prepare_vllm_moe = prepare
        engine.controller._dynamic_step_forward_logits = forward
        paths = [experts.__file__, vllm_fused_moe.__file__, __file__]
        metadata = dict(
            hostname=socket.gethostname(),
            rank=self.rank,
            pid=os.getpid(),
            cpu_affinity=(
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=str(torch.cuda.get_device_properties(torch.cuda.current_device())),
            source_sha256={str(p): sha256(p) for p in paths},
            environment={
                k: v
                for k, v in os.environ.items()
                if k.startswith(
                    (
                        "CUDA_",
                        "NCCL_",
                        "NVSHMEM_",
                        "TRITON_",
                        "OMP_",
                        "MKL_",
                        "NT4_",
                        "FLASHINFER_",
                    )
                )
            },
            events=events,
        )
        (destination / f"audit-metadata-rank{self.rank}.json").write_text(
            json.dumps(metadata, indent=2)
        )

    def mark_generation_start(self):
        # Called after the round barrier/synchronize, immediately before run_inference.
        if self.index > 0:
            self.first_forward.begin()

    def begin(self, index):
        self.index = index
        self.records = []
        self.active = self.events_enabled and index > 0
        if (
            self.rank != 0 or index < 0
            or os.environ.get("NT4_AUDIT_GPU_TELEMETRY", "1") == "0"
        ):
            return
        path = self.destination / f"audit-gpu-round{index}.csv"
        self.telemetry_file = path.open("w")
        fields = "timestamp,index,uuid,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,memory.used"
        try:
            self.telemetry = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=" + fields,
                    "--format=csv",
                    "--loop-ms=200",
                ],
                stdout=self.telemetry_file,
                stderr=subprocess.STDOUT,
            )
            snapshot = subprocess.run(
                ["nvidia-smi", "-q"], capture_output=True, text=True, timeout=15
            )
            (self.destination / f"audit-gpu-detail-round{index}.txt").write_text(
                snapshot.stdout + snapshot.stderr
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.telemetry_file.write(str(exc) + "\n")

    def end(self):
        # Caller has already synchronized; event reads cannot affect timed work.
        self.active = False
        if hasattr(self, "first_forward"):
            self.first_forward.gc_active = False
        self.stop_telemetry()
        rows = []
        for i, record in enumerate(self.records):
            a, b = self.pool[i]
            rows.append(dict(record, gpu_elapsed_ms=a.elapsed_time(b)))
        result = dict(
            rank=self.rank,
            round=self.index,
            note="Diagnostic only. GPU intervals include device waits and host submission gaps; not isolated kernel durations. No per-step synchronization.",
            first_forward_host_spans=getattr(
                getattr(self, "first_forward", None), "records", []
            ),
            gc_spans=getattr(getattr(self, "first_forward", None), "gc_records", []),
            forward_intervals=rows,
            launch_configs=[
                dict(json.loads(k), observations=v) for k, v in self.configs.items()
            ],
        )
        (self.destination / f"audit-round{self.index}-rank{self.rank}.json").write_text(
            json.dumps(result, indent=2)
        )

    def stop_telemetry(self):
        if self.telemetry is not None:
            self.telemetry.terminate()
            try:
                self.telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.telemetry.kill()
                self.telemetry.wait()
            self.telemetry = None
        if self.telemetry_file is not None:
            self.telemetry_file.close()
            self.telemetry_file = None

    def close(self):
        self.active = False
        if hasattr(self, "first_forward"):
            self.first_forward.close()
        self.stop_telemetry()
        self.experts.prepare_vllm_moe = self.original_prepare
        self.engine.controller._dynamic_step_forward_logits = self.original_forward
