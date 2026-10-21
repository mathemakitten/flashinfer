"""Diagnostic GPU timing of the native shortcut phases.

Serialized control removes competition between the two branches of selected
pairs. Other layers retain their normal schedule. Phase sums exclude explicit
inter-phase waits; spans include scheduling gaps. No E2E performance claims.
"""

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

PHASES = (
    "route_dispatch",
    "projection",
    "prepare",
    "mixer_core",
    "experts_combine",
    "mixer_post_shared",
)
MOE = ("route_dispatch", "prepare", "experts_combine")
MIXER = ("projection", "mixer_core", "mixer_post_shared")


TAIL_PHASES = (
    "mixer_post",
    "shared_norm_residual",
    "shared_unflatten",
    "shared_experts",
    "shared_fc1",
    "shared_fc2",
)


class BranchProbe:
    def __init__(self, layer, schedule):
        self.layer = layer
        self.schedule = schedule
        self.clock = os.environ.get("NT4_BRANCH_CLOCK", "device")
        if self.clock not in ("device", "events"):
            raise ValueError("NT4_BRANCH_CLOCK must be device or events")
        self.streams = {}
        self.after_join = None
        self.native_outputs = {}
        self.tail_enabled = os.environ.get("NT4_BRANCH_TAIL", "0") == "1"
        if self.tail_enabled and self.clock != "device":
            raise ValueError("Tail diagnostic requires the calibrated device clock")
        self.tail_active = False
        self.tail_seen = {}
        self.slot_count = 14 + (2 * len(TAIL_PHASES) if self.tail_enabled else 0)
        if self.clock == "device":
            from nt4_branch_clock import stamp

            self.stamp = stamp
            self.timestamps = torch.empty(
                self.slot_count, dtype=torch.int64, device="cuda"
            )
            # Compile every constant slot before model graph capture.
            for slot in range(self.slot_count):
                stamp(self.timestamps, slot)

        # External event nodes must execute on graph replay, not only capture.
        def event():
            return torch.cuda.Event(
                enable_timing=self.clock == "events", external=self.clock == "events"
            )

        self.fork = event()
        self.done = event()
        self.events = {name: (event(), event()) for name in PHASES}
        self.previous = None
        self.seen = set()
        self.snapshots = {}
        self.input_mode = os.environ.get("NT4_BRANCH_INPUT_MODE", "")
        self.forced_inputs = {}
        if self.input_mode:
            # Both record/replay graphs contain the same copy/select operations.
            # Input replacement happens before the timed fork, never in a branch.
            self.force_enabled = torch.zeros((), dtype=torch.bool, device="cuda")
        self.input_tag = torch.empty((), dtype=torch.int64, device="cuda")

    def begin(self, hidden_states):
        # Graph replay refreshes this snapshot before the timed fork.
        key = (tuple(hidden_states.shape), hidden_states.dtype)
        if key not in self.snapshots:
            self.snapshots[key] = (len(self.snapshots), torch.empty_like(hidden_states))
        tag, snapshot = self.snapshots[key]
        self.current_tag = tag
        if self.input_mode:
            if key not in self.forced_inputs:
                self.forced_inputs[key] = torch.zeros_like(hidden_states)
            hidden_states.copy_(
                torch.where(self.force_enabled, self.forced_inputs[key], hidden_states)
            )
        snapshot.copy_(hidden_states.detach())
        self.input_tag.fill_(tag)
        self.previous = None
        self.seen = set()
        self.tail_seen = {}
        self.tail_active = False
        if self.clock == "events":
            self.fork.record()
        else:
            self.stamp(self.timestamps, 0)

    def start(self, name):
        if name == "mixer_post_shared":
            self.tail_active = True
        if self.schedule == "serial" and self.previous is not None:
            torch.cuda.current_stream().wait_event(self.previous)
        self.streams[name] = int(torch.cuda.current_stream().cuda_stream)
        if self.clock == "events":
            self.events[name][0].record()
        else:
            self.stamp(self.timestamps, 1 + 2 * PHASES.index(name))

    def end(self, name):
        if name == "mixer_post_shared":
            self.tail_active = False
        if self.clock == "device":
            self.stamp(self.timestamps, 2 + 2 * PHASES.index(name))
        if self.clock == "events" or self.schedule == "serial":
            self.events[name][1].record()
        self.previous = self.events[name][1]
        self.seen.add(name)

    def tail_start(self, name):
        self.tail_seen[name] = self.tail_seen.get(name, 0) + 1
        self.stamp(self.timestamps, 14 + 2 * TAIL_PHASES.index(name))

    def tail_end(self, name):
        self.stamp(self.timestamps, 15 + 2 * TAIL_PHASES.index(name))

    def join(self):
        if self.tail_enabled and self.tail_seen != dict.fromkeys(TAIL_PHASES, 1):
            raise RuntimeError(f"missing or repeated tail operation: {self.tail_seen}")
        if self.seen != set(PHASES):
            raise RuntimeError(f"incomplete shortcut probe phases: {self.seen}")
        if self.clock == "events":
            self.done.record()
        else:
            self.stamp(self.timestamps, 13)
        if self.after_join is not None:
            self.native_outputs[self.current_tag] = self.after_join()

    def input_snapshot(self):
        tag = self.input_tag.item()
        return next(t for i, t in self.snapshots.values() if i == tag)

    def force_input(self, value):
        key = (tuple(value.shape), value.dtype)
        if key not in self.forced_inputs:
            raise RuntimeError(f"recorded input has no captured graph bucket: {key}")
        self.forced_inputs[key].copy_(value)
        self.force_enabled.fill_(True)

    def result(self):
        snapshot = self.input_snapshot()
        if self.clock == "events":
            offsets = {
                k: [self.fork.elapsed_time(e) * 1000 for e in pair]
                for k, pair in self.events.items()
            }
            join_us = self.fork.elapsed_time(self.done) * 1000
        else:
            ticks = self.timestamps.cpu().tolist()
            offsets = {
                k: [(ticks[1 + 2 * i + j] - ticks[0]) / 1000 for j in (0, 1)]
                for i, k in enumerate(PHASES)
            }
            join_us = (ticks[13] - ticks[0]) / 1000
        phases = {k: b - a for k, (a, b) in offsets.items()}
        if any(a < 0 or b < a or b > join_us for a, b in offsets.values()):
            raise RuntimeError("invalid branch timestamp order")
        tail = {}
        if self.tail_enabled:
            detail = {
                k: [(ticks[14 + 2 * i + j] - ticks[0]) / 1000 for j in (0, 1)]
                for i, k in enumerate(TAIL_PHASES)
            }
            start, end = offsets["mixer_post_shared"]
            if any(a < start or b < a or b > end for a, b in detail.values()):
                raise RuntimeError("tail timestamps outside parent interval")
            tail = dict(
                tail_offsets_us=detail,
                tail_phase_us={k: b - a for k, (a, b) in detail.items()},
                shared_fc1_to_fc2_gap_us=detail["shared_fc2"][0]
                - detail["shared_fc1"][1],
                tail_note="Inclusive intervals: shared_experts contains FC1/FC2. Gaps include activation, scheduling and timestamp overhead, not just arithmetic.",
            )
        return dict(
            **tail,
            native_output_policy=(
                "cutlass_after_join_v1" if self.after_join else "measured"
            ),
            native_output_sha256=(
                hashlib.sha256(
                    self.native_outputs[self.input_tag.item()]
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                ).hexdigest()
                if self.after_join
                else None
            ),
            layer=self.layer,
            clock=self.clock,
            capture_streams=self.streams,
            capture_streams_note="Capture/eager stream handles; graph replay may remap streams.",
            schedule=self.schedule,
            input_mode=self.input_mode or "live",
            phase_us=phases,
            input_sha256=hashlib.sha256(
                snapshot.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest(),
            phase_offsets_us=offsets,
            moe_phase_sum_us=sum(phases[k] for k in MOE),
            mixer_phase_sum_us=sum(phases[k] for k in MIXER),
            join_us=join_us,
        )


class BranchAudit:
    def __init__(self, engine, destination, schedule):
        from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock

        self.engine = engine
        self.destination = Path(destination)
        self.round = -1
        self.collect = True
        self.samples = 0
        self.decode_step = 0
        self.input_mode = os.environ.get("NT4_BRANCH_INPUT_MODE", "")
        self.input_root = Path(os.environ.get("NT4_BRANCH_INPUTS") or ".")
        if self.input_mode == "record":
            (self.input_root / f"rank{dist.get_rank()}").mkdir(
                parents=True, exist_ok=False
            )
        self.limit = int(os.environ.get("NT4_BRANCH_SAMPLES", "16"))
        self.probes = []
        wanted = {7, 32, 68}
        for module in engine.controller._language_model.modules():
            if isinstance(module, ShortcutMoEBlock):
                layer = module.moe_layer.layer_number
                if layer in wanted:
                    probe = BranchProbe(layer, schedule)
                    module._nt4_branch_probe = probe
                    if os.environ.get("NT4_BRANCH_NATIVE_OUTPUT") == "1":
                        from nt4_branch_native_output import install_native_output

                        install_native_output(module, probe)
                    if probe.tail_enabled:
                        install_tail_hooks(module, probe)
                    self.probes.append((module, probe))
        if {p.layer for _, p in self.probes} != wanted:
            raise RuntimeError("branch audit could not locate layers 7,32,68")
        self.original = engine.controller._dynamic_step_forward_logits

        def forward(*args, **kwargs):
            decode = engine.context.is_decode_only()
            active = int(engine.context.get_active_request_count()) if decode else 0
            paired = self.round > 0 and decode and bool(self.input_mode)
            path = (
                self.input_root
                / f"rank{dist.get_rank()}"
                / f"round{self.round}-step{self.decode_step}.pt"
            )
            recorded = None
            if paired and self.input_mode == "replay":
                recorded = torch.load(path, map_location="cpu", weights_only=True)
                if recorded["active_requests"] != active:
                    raise RuntimeError(
                        f"request occupancy differs from recorded sequence: {path}"
                    )
                for _, probe in self.probes:
                    probe.force_input(recorded["inputs"][probe.layer])
            result = self.original(*args, **kwargs)
            if paired:
                torch.cuda.synchronize()
                inputs = {p.layer: p.input_snapshot().cpu() for _, p in self.probes}
                if self.input_mode == "record":
                    torch.save(dict(active_requests=active, inputs=inputs), path)
                else:
                    for layer, value in inputs.items():
                        if not torch.equal(value, recorded["inputs"][layer]):
                            raise RuntimeError(
                                f"forced block input mismatch: {path}, layer={layer}"
                            )
                self.decode_step += 1
            if (
                self.collect
                and self.round > 0
                and decode
                and active == 128
                and self.samples < self.limit
            ):
                # All ranks take this path for the fixed replicated prompt workload.
                torch.cuda.synchronize()
                rows = [
                    dict(
                        rank=dist.get_rank(),
                        round=self.round,
                        sample=self.samples,
                        decode_step=self.decode_step - 1 if paired else None,
                        active_requests=active,
                        **p.result(),
                    )
                    for _, p in self.probes
                ]
                with (self.destination / f"branch-rank{dist.get_rank()}.jsonl").open(
                    "a"
                ) as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                self.samples += 1
            return result

        engine.controller._dynamic_step_forward_logits = forward

    def begin_round(self, index, *, collect=True):
        self.round = index
        self.collect = collect
        self.samples = 0
        self.decode_step = 0
        for _, probe in self.probes:
            if probe.input_mode:
                probe.force_enabled.fill_(False)


def install_tail_hooks(module, probe):
    """Wrap existing methods only during the sampled tail; no stream/wait changes."""

    def wrap(owner, method, label):
        original = getattr(owner, method)

        def call(*args, **kwargs):
            active = probe.tail_active
            if active:
                probe.tail_start(label)
            result = original(*args, **kwargs)
            if active:
                probe.tail_end(label)
            return result

        setattr(owner, method, call)

    compute = module.compute_layer
    post = (
        "inference_post"
        if hasattr(compute, "inference_post")
        else "forward_inference_post_core"
    )
    wrap(compute, post, "mixer_post")
    wrap(module.moe_layer, "_pre_mlp_layernorm_and_residual", "shared_norm_residual")
    wrap(module.moe_layer, "_maybe_unflatten_for_moe", "shared_unflatten")
    wrap(module.moe_layer.mlp, "shared_experts_compute", "shared_experts")
    shared = module.moe_layer.mlp.shared_experts
    wrap(shared.linear_fc1, "forward", "shared_fc1")
    wrap(shared.linear_fc2, "forward", "shared_fc2")
