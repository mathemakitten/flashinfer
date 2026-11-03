"""Bounded, dependency-free Python stack sampling for NT4 host diagnostics."""

import json
import os
from pathlib import Path
import socket
import sys
import threading
import time


class HostStackSampler:
    """Sample the calling thread after the first routed event, without CUDA APIs.

    This thread needs the GIL: samples are biased if native code holds it.
    Actual sample timestamps expose gaps; these are not native stacks or GPU
    timings. Only code locations are saved, never frame locals or tensor data.
    """

    def __init__(self, timeline, interval_ms=10, duration_seconds=30):
        if interval_ms <= 0 or duration_seconds <= 0:
            raise ValueError("sampling interval and duration must be positive")
        self.timeline = timeline
        self.interval = interval_ms / 1000
        self.duration = duration_seconds
        self.ident = threading.get_ident()
        self.tid = threading.get_native_id()
        self.stop_event = threading.Event()
        self.samples = []
        self.stacks = []
        self.stack_ids = {}
        self.error = None
        self.limit_reached = False
        self.thread = threading.Thread(
            target=self._run, name="nt4-host-stacks", daemon=True
        )

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            started = None
            max_samples = int(self.duration / self.interval) + 1
            sched_path = Path(f"/proc/self/task/{self.tid}/schedstat")
            while not self.stop_event.wait(self.interval):
                if not self.timeline:
                    continue
                now = time.monotonic_ns()
                if started is None:
                    started = now
                if (now - started) / 1e9 >= self.duration or len(
                    self.samples
                ) >= max_samples:
                    self.limit_reached = True
                    break
                frame = sys._current_frames().get(self.ident)
                stack = []
                while frame is not None:
                    stack.append(
                        (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
                    )
                    frame = frame.f_back
                key = tuple(stack)
                if key not in self.stack_ids:
                    self.stack_ids[key] = len(self.stacks)
                    self.stacks.append(stack)
                latest = self.timeline[-1]
                sample = dict(
                    monotonic_ns=now,
                    stack_id=self.stack_ids[key],
                    routed_calls=len(self.timeline),
                    last_layer=latest["layer"],
                    last_rows=latest["rows"],
                    last_replay_end_ns=latest["replay_end_ns"],
                )
                try:
                    values = [int(v) for v in sched_path.read_text().split()]
                    sample["main_schedstat"] = values[:3]
                except (OSError, ValueError):
                    pass
                self.samples.append(sample)
        except Exception as exc:
            self.error = repr(exc)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("host stack sampler did not stop")

    def save(self, path, **metadata):
        if self.thread.is_alive():
            raise RuntimeError("stop sampler before saving")
        payload = dict(
            schema=1,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            main_tid=self.tid,
            cpu_affinity=(
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            interval_ms=self.interval * 1000,
            duration_limit_seconds=self.duration,
            limit_reached=self.limit_reached,
            error=self.error,
            scope="Python calling-thread stacks, innermost frame first; GIL-biased; no native/GPU stacks; diagnostic throughput only",
            clock="time.monotonic_ns; compare timestamps on the same host only",
            schedstat_fields=["cpu_runtime_ns", "runqueue_wait_ns", "timeslices"],
            schedstat_note="Linux optional cumulative main-thread counters; runqueue accounting may be disabled",
            stacks=self.stacks,
            samples=self.samples,
            **metadata,
        )
        Path(path).write_text(json.dumps(payload))
