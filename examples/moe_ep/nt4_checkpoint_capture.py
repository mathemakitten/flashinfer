"""Opt-in wrapper around an existing MCore inference command.

torchrun --nproc-per-node=4 nt4_checkpoint_capture.py --artifact DIR --layer 1 \
    --mcore-root /path/to/megatron-lm -- SCRIPT.py ORIGINAL_MODEL_ARGUMENTS...

The wrapper captures the selected loaded layer at the latent routed boundary,
replays the live baseline, validates MegaMoE, and optionally benchmarks both.
No production MCore files are modified. The original inference run then resumes.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import traceback

import torch
import torch.distributed as dist

from nt4_checkpoint_artifact import (
    SCHEMA,
    canonical_scale,
    save_manifest,
    save_payload,
    validate_sample,
    validate_weights,
)
from nt4_checkpoint_replay import Replay, benchmark, emit, gather, profile_layers
from nt4_token_sweep import make_sweep_sample, token_census, sweep_points


def revision(root, destination, prefix):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.PIPE,
            timeout=10,
        )

    result = dict(root=str(root), head=None, status=None, git_available=False)
    try:
        result["head"] = git("rev-parse", "HEAD").strip()
        result["status"] = git("status", "--short")
        diff = git("diff", "HEAD", "--")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        result["git_error"] = str(error)
        print(
            json.dumps(
                dict(
                    stage="provenance",
                    source=prefix,
                    git_available=False,
                    reason=str(error),
                )
            ),
            flush=True,
        )
        return result
    result["git_available"] = True
    (destination / f"{prefix}-tracked.patch").write_text(diff)
    return result


def validate_layer(layer, *, expected_backend="flashinfer"):
    c = layer.config
    required = dict(
        moe_latent_size=768,
        moe_ffn_hidden_size=4096,
        num_moe_experts=512,
        moe_router_topk=10,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        pipeline_model_parallel_size=1,
        activation_func_tanh_clamp_scale=16,
        gated_linear_unit=False,
    )
    for name, expected in required.items():
        if getattr(c, name, None) != expected:
            raise ValueError(
                f"unsupported {name}: {getattr(c, name, None)}, expected {expected}"
            )
    if "squared_relu" not in getattr(c.activation_func, "__name__", ""):
        raise ValueError("selected layer must use squared_relu")
    if (
        expected_backend
        not in str(layer.experts.inference_grouped_gemm_backend).lower()
    ):
        raise ValueError(f"live baseline must use {expected_backend}")
    if layer.token_dispatcher.__class__.__name__ != "NVLSAllGatherVDispatcher":
        raise ValueError("live baseline must be NVLSAllGatherVDispatcher")
    d = layer.token_dispatcher
    if d.shared_experts is not None and not d._external_shared_expert_launch:
        raise ValueError("shared expert must execute outside the routed boundary")
    if d.ep_group.size() != 4 or dist.get_world_size() != 4:
        raise ValueError("initial harness requires one EP4 world, no PP/DP replicas")
    if d.ep_group.rank() != dist.get_rank():
        raise ValueError("EP rank ordering must match world rank ordering")


def export_weights(experts, rank):
    if not experts._uses_mxfp8_weights:
        raise ValueError("selected checkpoint layer is not MXFP8; select another layer")
    weights = {}
    for linear, qkey, skey in (("linear_fc1", "w1", "s1"), ("linear_fc2", "w2", "s2")):
        canonical = experts._stack_mxfp8_linear_weight(linear, "triton")
        if canonical.data.dtype != torch.float8_e4m3fn:
            raise ValueError("checkpoint weights must use E4M3")
        _, rows, cols = canonical.data.shape
        weights[qkey] = canonical.data.detach().contiguous().view(torch.uint8).cpu()
        weights[skey] = torch.stack(
            [canonical_scale(s, rows, cols) for s in canonical.scale]
        ).cpu()
    weights["expert_ids"] = torch.arange(rank * 128, (rank + 1) * 128)
    validate_weights(weights, rank)
    return weights


class Capture:
    def __init__(self, args):
        self.args = args
        from nt4_capture_state import FilterStats

        self.filter_stats = FilterStats()
        self.calls = 0
        self.last_capture_call = None
        self.step_metadata = None
        self.count = 0
        self.busy = False
        self.weights = None
        self.runtime_hold = None
        self.weight_hashes = None
        self.sample_hashes = []
        self.results = []
        self.pending = None
        self.bound_layer = None
        self.observed_layers = set()

    def install(self):
        from megatron.core.transformer.moe.moe_layer import MoELayer

        original = MoELayer.preprocess
        capture = self

        def preprocess(layer, *args, **kwargs):
            capture.observed_layers.add(layer.layer_number)
            if capture.args.layer == 0 and capture.bound_layer is None:
                eligible = layer.experts._expert_weights_use_mxfp8()
                choices = gather(bool(eligible), layer.token_dispatcher.ep_group)
                if len(set(choices)) != 1:
                    raise ValueError("selected layer precision differs across EP ranks")
                if eligible:
                    capture.args.layer = layer.layer_number
                    emit(stage="selected_layer", layer=layer.layer_number)
            if layer.layer_number == capture.args.layer and not capture.busy:
                capture.bind(layer)
            return original(layer, *args, **kwargs)

        MoELayer.preprocess = preprocess

    def bind(self, layer):
        if self.bound_layer is layer:
            return
        if self.bound_layer is not None:
            raise ValueError("layer number matched multiple model instances")
        validate_layer(layer)
        self.args.provenance["selected_layer"] = dict(
            layer_number=layer.layer_number,
            module_type=f"{type(layer).__module__}.{type(layer).__name__}",
            expert_type=f"{type(layer.experts).__module__}.{type(layer.experts).__name__}",
            activation=getattr(layer.config.activation_func, "__name__", "unknown"),
            clamp=layer.config.activation_func_tanh_clamp_scale,
            router_score=getattr(layer.config, "moe_router_score_function", None),
            router_scale=getattr(layer.config, "moe_router_topk_scaling_factor", None),
        )
        self.bound_layer = layer
        d = layer.token_dispatcher
        before, after = d.dispatch_preprocess, d.combine_postprocess

        def begin(x, routes, probs):
            if not self.busy and self.count < self.args.samples:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "disable model CUDA graphs for capture; use wrapper --graph for replay"
                    )
                self.calls += 1
                counts = gather(x.numel() // 768, d.ep_group)
                from nt4_capture_state import snapshot, filter_reasons

                states = gather(snapshot(), d.ep_group)
                reasons = filter_reasons(
                    counts,
                    states,
                    call=self.calls,
                    skip_calls=self.args.skip_calls,
                    max_local_tokens=self.args.max_local_tokens,
                    last_capture_call=self.last_capture_call,
                    capture_interval=self.args.capture_interval,
                    decode_only=self.args.decode_only,
                    min_active_tokens=self.args.min_active_tokens,
                )
                if "missing_engine_context" in reasons:
                    raise RuntimeError(
                        "decode-only capture requires engine context from nt4_checkpoint_generate.py"
                    )
                eligible = not reasons
                improved = self.filter_stats.observe(
                    self.calls, counts, states, reasons
                )
                if not eligible and (
                    self.calls <= 8 or improved or self.calls % 32 == 0
                ):
                    emit(
                        stage="capture_filtered",
                        reasons=reasons,
                        min_active_tokens=self.args.min_active_tokens,
                        call=self.calls,
                        tokens_per_rank=counts,
                        skip_calls=self.args.skip_calls,
                        max_local_tokens=self.args.max_local_tokens,
                        engine_steps=states,
                    )
                if eligible:
                    self.last_capture_call = self.calls
                    self.step_metadata = states
                    if routes.shape != (x.numel() // 768, 10):
                        raise ValueError(
                            "expected selected top-k IDs, not dense routing mask"
                        )
                    self.pending = (
                        x.detach().clone(),
                        routes.detach().clone(),
                        probs.detach().clone(),
                    )
                    emit(
                        stage="capture_begin",
                        call=self.calls,
                        sample=self.count,
                        tokens_per_rank=counts,
                        engine_steps=states,
                    )
            return before(x, routes, probs)

        def end(output):
            result = after(output)
            if self.pending is not None and not self.busy:
                # Clone before replay reuses the baseline's symmetric buffers.
                saved_output = result.detach().clone()
                x, routes, probs = self.pending
                self.pending = None
                self.busy = True
                try:
                    self.process(layer, x, routes, probs, saved_output, before, after)
                finally:
                    self.busy = False
                return saved_output
            return result

        d.dispatch_preprocess = begin
        d.combine_postprocess = end
        if self.args.trace:
            # Host entry/return markers only: do not introduce GPU synchronizations
            # that could change the dispatcher's overlap or collective ordering.
            def traced(name, method):
                def call(*args, **kwargs):
                    emit(stage="host_enter", operation=name, call=self.calls)
                    result = method(*args, **kwargs)
                    emit(stage="host_return", operation=name, call=self.calls)
                    return result

                return call

            for name in (
                "dispatch_preprocess",
                "token_dispatch",
                "dispatch_postprocess",
                "combine_preprocess",
                "token_combine",
                "combine_postprocess",
            ):
                setattr(d, name, traced(name, getattr(d, name)))
            layer.experts.register_forward_pre_hook(
                lambda module, inputs: emit(
                    stage="host_enter", operation="experts", call=self.calls
                )
            )
            layer.experts.register_forward_hook(
                lambda module, inputs, output: emit(
                    stage="host_return", operation="experts", call=self.calls
                )
            )

    def process(self, layer, x, routes, probs, output, before, after):
        from megatron.core.typed_torch import apply_module

        args, d = self.args, layer.token_dispatcher
        group, rank = d.ep_group, d.ep_group.rank()
        torch.cuda.synchronize()  # finish the model's shared-expert side stream too
        if self.weights is None:
            self.weights = export_weights(layer.experts, rank)
            if list(layer.local_expert_indices) != self.weights["expert_ids"].tolist():
                raise ValueError(
                    "loaded layer expert order differs from canonical EP partition"
                )
            self.weight_hashes = save_payload(
                args.artifact / f"weights-rank{rank}.pt", self.weights
            )
        source_shape = x.shape
        source_n = x.numel() // 768
        source_real_tokens = d.__class__._real_token_count_tensor
        # Static replay uses this captured step's metadata. Update explicitly
        # even if the selected layer normally relies on an earlier layer.
        d.update_metadata(x.numel() // 768)

        baseline_audit = dict(
            dispatcher=type(d).__name__,
            expert_module=type(layer.experts).__name__,
            weight_type=type(layer.experts._fc1_weight).__name__,
            nvls_capacity_per_rank=d.__class__._per_rank_worst_case_token_count,
            scope="isolated routed section; production methods, model overlap excluded",
        )

        def baseline():
            h, p = before(x, routes, probs)
            h, p = d.token_dispatch(h, p)
            h, counts, p = d.dispatch_postprocess(h, p)
            baseline_audit["expert_input_rows"] = h.shape[0]
            h, bias = apply_module(layer.experts)(
                h, counts, p, routing_map=d.routing_map
            )
            if bias is not None:
                raise ValueError("routed expert bias is unsupported")
            h = d.combine_preprocess(h)
            return after(d.token_combine(h)).reshape(-1, 768)

        from nt4_checkpoint_replay import compare

        compare(
            baseline().clone(),
            output.reshape(-1, 768),
            "adapter_vs_original_model_call",
        )
        # baseline() has applied the engine's padding mask to routes in place.
        source_census = gather(token_census(dict(ids=routes, probs=probs)), group)
        engine_counts = gather(
            None if source_real_tokens is None else int(source_real_tokens.item()),
            group,
        )
        emit(
            stage="token_census",
            ranks=source_census,
            engine_real_token_count_per_rank=engine_counts,
            note="engine count is pre-SP; EP rank counts may describe replicated tokens",
        )
        if args.replay_counts is not None or args.mask_odd_slots:
            if args.replay_counts is not None:
                n = args.replay_counts[rank]
                if source_n == 0 and n > 0:
                    raise ValueError(
                        "cannot resample nonempty input from an empty capture"
                    )
                index = torch.arange(n, device=x.device) % max(1, source_n)
                x = x.reshape(-1, 768)[index].clone()
                routes, probs = routes[index].clone(), probs[index].clone()
                if n > d.__class__._per_rank_worst_case_token_count:
                    raise ValueError("resampled count exceeds allocated NVLS capacity")
                # Synthetic counts no longer describe the original padded
                # global token layout. Preserve explicit route masks.
                d.__class__._real_token_count_tensor = None
            if args.mask_odd_slots:
                routes = routes.clone()
                probs = probs.clone()
                routes[:, 1::2] = -1
                probs[:, 1::2] = 0
            d.update_metadata(x.numel() // 768)
            output = baseline().clone()
        sample = dict(
            x=x.reshape(-1, 768).cpu(),
            ids=routes.to(torch.int64).cpu(),
            probs=torch.where(routes < 0, 0, probs).float().cpu(),
            baseline=output.reshape(-1, 768).cpu(),
        )
        validate_sample(sample)
        self.sample_hashes.append(
            save_payload(args.artifact / f"sample{self.count}-rank{rank}.pt", sample)
        )
        self.publish(group, complete=False)
        capacity = max(gather(x.numel() // 768, group))
        if args.sweep_counts:
            points = sweep_points(
                args.sweep_counts, args.sweep_buffer_rows, args.sweep_bucket_rows
            )
            required = max(rows for _, rows in points)
            available = d.__class__._per_rank_worst_case_token_count
            if required > available:
                raise ValueError(
                    f"sweep needs {required} NVLS rows per rank, but allocated capacity is {available}; increase model --inference-dynamic-batching-max-tokens"
                )
            if args.sweep_bucket_rows is None:
                capacity = max(capacity, args.sweep_buffer_rows)
        if args.match_production_capacity:
            production_capacity = d.__class__._per_rank_worst_case_token_count
            if production_capacity < capacity:
                raise ValueError("production capacity is smaller than captured rows")
            capacity = production_capacity
        replay = Replay(
            self.weights,
            capacity,
            group,
            routing_weight_placement=args.routing_weight_placement,
            knobs=args.knobs,
        )
        if self.runtime_hold is None:
            # Retain the first runtime reference across samples; layer workspaces
            # can be destroyed without finalizing/reinitializing NVSHMEM.
            self.runtime_hold = replay.runtime
            replay.runtime = None
        replay.check(sample, baseline, graph=args.graph)
        hybrid = None
        if args.three_way:
            from nt4_nvls_hybrid import NvlsHybrid

            hybrid = NvlsHybrid(replay, args.mcore_root)

        def check_hybrid(point):
            if hybrid is None:
                return
            if len(set(gather(replay.n, group))) != 1:
                raise ValueError(
                    "three-way comparison requires equal captured rows on all ranks"
                )
            candidate = hybrid.run().clone()
            hybrid.check_fused_helpers()
            compare(
                candidate[: replay.n],
                point["baseline"].cuda(),
                "nvls_hybrid_vs_live_cutlass",
            )
            compare(candidate, replay.run(), "nvls_hybrid_vs_direct")
            if bool((candidate[replay.n :] != 0).any()):
                raise AssertionError("hybrid padded output must be zero")
            torch.testing.assert_close(candidate, hybrid.run(), rtol=0, atol=0)
            if args.graph:
                dist.barrier(group=group)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = hybrid.run()
                for _ in range(3):
                    graph.replay()
                    torch.testing.assert_close(actual, candidate, rtol=0, atol=0)
            emit(stage="three_way_correctness", status="PASS", graph=args.graph)

        check_hybrid(sample)
        baseline_audit["candidate_buffer_rows_per_rank"] = replay.capacity
        baseline_audit["capacity_mode"] = (
            "match_production" if args.match_production_capacity else "capture_rows"
        )
        baseline_audit["candidate_padding_rows"] = replay.capacity - replay.n
        baseline_audit["hybrid_communication_rows_per_rank"] = replay.capacity
        emit(stage="baseline_audit", **baseline_audit)
        result = dict(
            baseline_audit=dict(baseline_audit),
            engine_steps=self.step_metadata,
            workload=(
                "real_decode_capture"
                if args.decode_only
                else "captured_step_unclassified"
            ),
            three_way=args.three_way,
            sample=self.count,
            source_call=self.calls,
            correctness="PASS",
            knobs=args.knobs,
            layer=args.layer,
            tokens_per_rank=gather(x.numel() // 768, group),
            graph=args.graph,
            token_census=source_census,
            engine_real_token_count_per_rank=engine_counts,
        )
        if args.benchmark:
            result["benchmark"] = benchmark(
                replay,
                baseline,
                graph=args.graph,
                warmup=args.warmup,
                iterations=args.iterations,
                rounds=args.rounds,
                hybrid=None if hybrid is None else hybrid.run,
            )
        if args.profile_dir:
            profile_layers(
                replay,
                args.profile_dir / f"sample{self.count}",
                baseline=baseline,
                hybrid=None if hybrid is None else hybrid.run,
                graph=args.graph,
                iterations=args.profile_iterations,
            )
        if args.sweep_counts:
            pools = gather({key: sample[key] for key in ("x", "ids", "probs")}, group)
            pool = {key: torch.cat([part[key] for part in pools]) for key in pools[0]}
            result["sweep"] = []
            # Resampled workloads have explicit route masks, independent of the
            # engine's original token count. Each point is checked before timing.
            d.__class__._real_token_count_tensor = None
            for active_rows, buffer_rows in points:
                if replay.capacity != buffer_rows:
                    # Rebuild only at bucket boundaries, outside correctness and timing.
                    if hybrid is not None:
                        hybrid.workspace.destroy()
                        hybrid = None
                    previous_replay = replay
                    replay = Replay(
                        self.weights,
                        buffer_rows,
                        group,
                        routing_weight_placement=args.routing_weight_placement,
                        knobs=args.knobs,
                    )
                    # Keep one runtime reference alive across bucket changes;
                    # NVSHMEM must not be finalized and reinitialized here.
                    previous_replay.close()
                    del previous_replay
                    if args.three_way:
                        hybrid = NvlsHybrid(replay, args.mcore_root)
                point = make_sweep_sample(pool, active_rows, buffer_rows, rank)
                x, routes, probs = (point[key].cuda() for key in ("x", "ids", "probs"))
                d.update_metadata(buffer_rows)
                point["baseline"] = baseline().detach().cpu().clone()
                filename = f"sweep{self.count}-active{active_rows}-rank{rank}.pt"
                hashes = save_payload(args.artifact / filename, point)
                replay.check(point, baseline, graph=args.graph)
                check_hybrid(point)
                timing = benchmark(
                    replay,
                    baseline,
                    graph=args.graph,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    rounds=args.rounds,
                    hybrid=None if hybrid is None else hybrid.run,
                )
                record = dict(
                    active_rows_per_rank=active_rows,
                    buffer_rows=buffer_rows,
                    workload="repeated_captured_active_rows_fixed_buffers",
                    sample_file=filename,
                    sample_hashes=hashes,
                    token_census=gather(token_census(point), group),
                    benchmark=timing,
                )
                result["sweep"].append(record)
                emit(stage="sweep_point", **record)
                # Persist each completed point so later failures don't discard timings.
                if rank == 0:
                    save_manifest(
                        args.artifact / f"sweep{self.count}.json", result["sweep"]
                    )
        if hybrid is not None:
            hybrid.workspace.destroy()
        replay.close()
        d.__class__._real_token_count_tensor = source_real_tokens
        d.update_metadata(source_n)
        d.hidden_shape = source_shape
        d._local_tokens = source_n
        self.results.append(result)
        self.count += 1
        self.publish(group, complete=self.count == args.samples)
        emit(
            stage="capture_complete", sample=self.count - 1, artifact=str(args.artifact)
        )

    def publish(self, group, *, complete):
        parts = gather(
            dict(
                weights=self.weight_hashes,
                samples=self.sample_hashes,
                results=self.results,
            ),
            group,
        )
        if dist.get_rank(group) == 0:
            manifest = dict(
                schema=SCHEMA,
                world_size=4,
                layer=self.args.layer,
                ranks=parts,
                provenance=self.args.provenance,
                capture_complete=complete,
            )
            save_manifest(self.args.artifact / "manifest.json", manifest)
        dist.barrier(group=group)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--mcore-root", type=Path, required=True)
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="MCore layer_number; 0 selects the first MXFP8 MoE layer",
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--skip-calls", type=int, default=1)
    parser.add_argument("--max-local-tokens", type=int, default=64)
    parser.add_argument("--match-production-capacity", action="store_true")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--min-active-tokens", type=int, default=1)
    parser.add_argument(
        "--capture-interval",
        type=int,
        default=1,
        help="minimum layer calls between captured samples",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--three-way",
        action="store_true",
        help="time live NVLS+CUTLASS, NVLS+MegaMoE, and direct MegaMoE together (GB300 EP4)",
    )
    parser.add_argument(
        "--knobs-json",
        default="{}",
        help="explicit MegaMoE knobs for live baseline comparison",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="export warmed baseline and MegaMoE CPU/CUDA traces",
    )
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument(
        "--sweep-counts",
        help="comma-separated active rows per EP rank, e.g. 1,2,4,8,16,32,64",
    )
    parser.add_argument("--sweep-buffer-rows", type=int, default=64)
    parser.add_argument(
        "--sweep-bucket-rows",
        help="comma-separated increasing padded row buckets; overrides sweep-buffer-rows",
    )
    parser.add_argument(
        "--routing-weight-placement",
        choices=("before_quantization", "after_fc2"),
        default="after_fc2",
    )
    parser.add_argument(
        "--replay-counts",
        help="four comma-separated local counts; resamples captured rows, e.g. 0,4,8,16",
    )
    parser.add_argument(
        "--mask-odd-slots", action="store_true", help="synthetic partial-routing stress"
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="capture replay/timing only; leave full-model graphs off",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="log selected-layer host stages and dump Python stacks every 120 seconds",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from nt4_gb300_tune import validate_knobs

    try:
        args.knobs = validate_knobs(json.loads(args.knobs_json))
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    if (
        min(
            args.samples,
            args.min_active_tokens,
            args.capture_interval,
            args.max_local_tokens,
            args.warmup,
            args.iterations,
            args.rounds,
            args.timeout_seconds,
            args.profile_iterations,
        )
        < 1
        or args.skip_calls < 0
        or args.layer < 0
    ):
        parser.error("counts must be positive; skip-calls must be nonnegative")
    if args.match_production_capacity and (not args.three_way or args.sweep_counts):
        parser.error(
            "matched capacity requires three-way capture without synthetic sweep"
        )
    if args.three_way:
        if torch.cuda.get_device_capability() != (10, 3):
            parser.error("three-way hybrid requires GB300/SM103")
        if args.routing_weight_placement != "after_fc2":
            parser.error("three-way hybrid requires after_fc2 routing weights")
        args.knobs.setdefault("group_hint", 512)
        args.benchmark = True
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply -- SCRIPT.py ORIGINAL_MODEL_ARGUMENTS...")
    if Path(command[0]).name == "vlm_dynamic_inference.py":
        parser.error(
            "vlm_dynamic_inference.py is a helper module, not an executable. "
            "Use the absolute path to examples/moe_ep/nt4_checkpoint_generate.py "
            "in the FlashInfer checkout after --, keeping the model arguments."
        )
    if args.replay_counts is not None:
        try:
            args.replay_counts = [int(n) for n in args.replay_counts.split(",")]
        except ValueError:
            parser.error("replay-counts must be four nonnegative integers")
        if (
            len(args.replay_counts) != 4
            or min(args.replay_counts) < 0
            or max(args.replay_counts) == 0
        ):
            parser.error("replay-counts requires four nonnegative counts, not all zero")
    args.artifact = args.artifact.resolve()
    if args.sweep_counts:
        try:
            args.sweep_counts = [int(n) for n in args.sweep_counts.split(",")]
        except ValueError:
            parser.error("sweep-counts must be comma-separated integers")
        try:
            if args.sweep_bucket_rows is not None:
                args.sweep_bucket_rows = [
                    int(n) for n in args.sweep_bucket_rows.split(",")
                ]
            sweep_points(
                args.sweep_counts, args.sweep_buffer_rows, args.sweep_bucket_rows
            )
        except ValueError as error:
            parser.error(str(error))
        if args.replay_counts is not None or args.mask_odd_slots:
            parser.error(
                "sweep cannot be combined with replay-counts or mask-odd-slots"
            )
        args.benchmark = True
    args.mcore_root = args.mcore_root.resolve()
    rank = int(os.environ["RANK"])
    if rank == 0:
        args.artifact.mkdir(parents=True, exist_ok=False)
    # torch.distributed belongs to MCore; do not initialize it before the loader.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    faulthandler.enable()
    if args.trace:
        faulthandler.dump_traceback_later(120, repeat=True)
    watchdog = threading.Timer(
        args.timeout_seconds,
        lambda: (
            print("FAIL: checkpoint capture watchdog expired", flush=True),
            faulthandler.dump_traceback(all_threads=True),
            os._exit(124),
        ),
    )
    watchdog.daemon = True
    watchdog.start()
    sys.path.insert(0, str(args.mcore_root))
    args.provenance = dict(
        command=command,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nccl=torch.cuda.nccl.version(),
        versions={},
        source_files={},
        routing_weight_placement=args.routing_weight_placement,
        knobs=args.knobs,
        three_way=args.three_way,
        decode_only=args.decode_only,
        min_active_tokens=args.min_active_tokens,
        capture_interval=args.capture_interval,
        sweep_counts=args.sweep_counts,
        sweep_buffer_rows=args.sweep_buffer_rows,
        sweep_bucket_rows=args.sweep_bucket_rows,
        replay_counts=args.replay_counts,
        mask_odd_slots=args.mask_odd_slots,
        workload=(
            "resampled_capture"
            if args.replay_counts is not None or args.mask_odd_slots
            else "captured_step_unclassified"
        ),
    )
    for name in ("flashinfer-python", "nvidia-cutlass-dsl", "nvshmem4py-cu13"):
        try:
            args.provenance["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            args.provenance["versions"][name] = "metadata absent"
    if rank == 0:
        fi_root = Path(__file__).resolve().parents[2]
        args.provenance["mcore"] = revision(args.mcore_root, args.artifact, "mcore")
        args.provenance["flashinfer"] = revision(fi_root, args.artifact, "flashinfer")
        for filename, arguments in (
            ("gpu-topology.txt", ["nvidia-smi", "topo", "-m"]),
            ("gpu-state.txt", ["nvidia-smi"]),
        ):
            try:
                result = subprocess.run(
                    arguments, capture_output=True, text=True, timeout=10
                )
                (args.artifact / filename).write_text(result.stdout + result.stderr)
            except (OSError, subprocess.TimeoutExpired) as error:
                (args.artifact / filename).write_text(f"unavailable: {error}\n")
        # These scripts may be untracked; archive exact source alongside patches.
        import hashlib

        for path in Path(__file__).parent.glob("nt4*.py"):
            data = path.read_bytes()
            (args.artifact / path.name).write_bytes(data)
            args.provenance["source_files"][path.name] = hashlib.sha256(
                data
            ).hexdigest()
        # Also archive the owned kernel files, which may be untracked and are
        # therefore absent from git diff. Avoid archiving unrelated user files.
        kernel_root = (
            fi_root / "flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/shim"
        )
        sources = list((kernel_root / "nt4").glob("*.py")) + [
            kernel_root / "mxfp8.py",
            kernel_root / "clear_invalid_combine.py",
        ]
        for source in sources:
            relative = source.relative_to(fi_root)
            target = args.artifact / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = source.read_bytes()
            target.write_bytes(data)
            args.provenance["source_files"][str(relative)] = hashlib.sha256(
                data
            ).hexdigest()
    capture = Capture(args)
    capture.install()
    script = Path(command[0])
    if not script.is_absolute():
        script = args.mcore_root / script
    sys.argv = [str(script), *command[1:]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    summary = capture.filter_stats.report()
    emit(stage="capture_filter_summary", **summary)
    if dist.get_rank() == 0:
        save_manifest(args.artifact / "capture_filter_summary.json", summary)
    if capture.count != args.samples:
        raise RuntimeError(
            f"only captured {capture.count}/{args.samples}; "
            f"observed MoE layers={sorted(capture.observed_layers, key=str)}, "
            f"selected layer={args.layer}, dispatcher calls={capture.calls}. "
            f"decode-only calls={summary['decode_calls']}; "
            f"rejected_by_reason={summary['rejected_by_reason']}. "
            "See capture_filter_summary.json for peak decode occupancy and padded rows. "
            "No occupancy threshold was relaxed."
        )
    if capture.runtime_hold is not None:
        from flashinfer.moe_ep import finalize_moe_ep_runtime

        finalize_moe_ep_runtime(capture.runtime_hold)
    watchdog.cancel()
    if args.trace:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
