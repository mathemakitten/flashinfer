"""Checkpoint-layer MegaMoE reference replay; optionally time a live baseline.

Offline: torchrun --nproc-per-node=4 this_file.py --artifact DIR
Live baseline capture/comparison: nt4_checkpoint_capture.py --help
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import threading
import sys
import traceback

import torch
import torch.distributed as dist

from nt4_checkpoint_artifact import (
    SCHEMA,
    load_payload,
    validate_sample,
    validate_weights,
)
from nt4_megamoe_bringup import validate_contributions, weighted_bf16_recombine


def emit(**record):
    print(json.dumps(dict(rank=dist.get_rank(), **record)), flush=True)


def gather(value, group=None):
    results = [None] * dist.get_world_size(group)
    dist.all_gather_object(results, value, group=group)
    return results


def compare(actual, expected, label):
    if not bool(torch.isfinite(actual).all()) or not bool(
        torch.isfinite(expected).all()
    ):
        raise AssertionError(f"{label}: nonfinite output")
    delta = actual.float() - expected.float()
    global_l2 = (delta.norm() / expected.float().norm().clamp_min(1e-8)).item()
    row_l2 = delta.norm(dim=-1) / expected.float().norm(dim=-1).clamp_min(1e-8)
    metrics = dict(
        global_rel_l2=global_l2,
        max_row_rel_l2=row_l2.max().item() if row_l2.numel() else 0,
        max_abs=delta.abs().max().item() if delta.numel() else 0,
        strict_mismatches=int(
            (~torch.isclose(actual, expected, rtol=0.08, atol=0.002)).sum().item()
        ),
    )
    emit(stage="comparison", label=label, **metrics)
    if global_l2 > 0.015 or metrics["max_row_rel_l2"] > 0.015:
        raise AssertionError(
            f"{label}: provisional global/per-token L2 gate failed: {metrics}"
        )
    empty_rows = (expected == 0).all(dim=-1)
    if bool((actual[empty_rows] != 0).any()):
        raise AssertionError(f"{label}: nonzero output for an exact-zero reference row")
    return metrics


class Replay:
    def __init__(
        self,
        weights,
        capacity,
        group=None,
        routing_weight_placement="after_fc2",
        knobs=None,
    ):
        self.routing_weight_placement = routing_weight_placement
        from flashinfer.moe_ep import (
            BootstrapConfig,
            FleetParams,
            MegaConfig,
            MoEEpLayer,
            MoEWeightPack,
            Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
            bootstrap_moe_ep_runtime,
        )
        from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

        self.group = group
        self.rank = dist.get_rank(group)
        if dist.get_world_size(group) != 4:
            raise ValueError("checkpoint replay requires EP4")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise ValueError(
                "checkpoint replay currently targets data-center Blackwell"
            )
        validate_weights(weights, self.rank)
        self.weights = {k: v.cuda() for k, v in weights.items()}
        self.capacity = max(1, capacity)
        config = Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=4096,
            top_k=10,
            activation="clamped_relu2",
            clamped_relu2_limit=16.0,
            routing_weight_placement=routing_weight_placement,
            enable_in_kernel_fc2_reduce=False,
            knobs={} if knobs is None else knobs,
        )
        bootstrap = BootstrapConfig(
            world_size=4,
            rank=self.rank,
            process_group=group,
            device=torch.cuda.current_device(),
            auto_bootstrap=False,
        )
        self.runtime = bootstrap_moe_ep_runtime(
            bootstrap, create_mega_kernel(config).runtime_requirements(bootstrap)
        )
        w = self.weights
        start = time.perf_counter()
        self.layer = MoEEpLayer(
            bootstrap=bootstrap,
            fleet_params=FleetParams(
                num_experts=512,
                max_tokens_per_rank=self.capacity,
                token_hidden_size=768,
            ),
            weights=MoEWeightPack(
                w13=w["w1"].view(torch.float8_e4m3fn),
                w2=w["w2"].view(torch.float8_e4m3fn),
                w13_scale=w["s1"],
                w2_scale=w["s2"],
            ),
            backend=MegaConfig(
                megakernel=config, quantize_input=True, preprocess_weights=True
            ),
        )
        self.x = torch.zeros(self.capacity, 768, dtype=torch.bfloat16, device="cuda")
        self.ids = torch.full((self.capacity, 10), -1, dtype=torch.int64, device="cuda")
        self.probs = torch.zeros(self.capacity, 10, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        emit(
            stage="setup",
            capacity=self.capacity,
            allocation_and_pack_seconds=time.perf_counter() - start,
        )

    def stage(self, sample):
        validate_sample(sample)
        self.n = sample["x"].shape[0]
        if self.n > self.capacity:
            raise ValueError("sample exceeds configured capacity")
        self.x.zero_()
        self.ids.fill_(-1)
        self.probs.zero_()
        self.x[: self.n].copy_(sample["x"])
        self.ids[: self.n].copy_(sample["ids"])
        self.probs[: self.n].copy_(sample["probs"])

    def run(self):
        from flashinfer.moe_ep import MoEEpTensors

        return self.layer.forward(
            MoEEpTensors(
                hidden_states=self.x, topk_ids=self.ids, topk_weights=self.probs
            )
        )

    def reference(self, *, diagnose_input=False, diagnose_routing=False):
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
            mxfp8_quantize_per_block_32,
        )

        def roundtrip(x):
            q, scales = mxfp8_quantize_per_block_32(x.contiguous(), torch.float8_e4m3fn)
            return q.float() * scales.float().repeat_interleave(32, -1)

        def dequant(q, s):
            return q.view(torch.float8_e4m3fn).float() * torch.pow(
                2.0, s.float() - 127
            ).repeat_interleave(32, -1)

        selected = sorted(
            set(
                e
                for values in gather(self.ids.cpu().unique().tolist(), self.group)
                for e in values
                if e >= 0
            )
        )
        terms = torch.zeros(self.capacity, 10, 768, device="cuda")
        products = torch.zeros_like(terms)
        late_terms = torch.zeros_like(terms) if diagnose_routing else None
        late_bf16_terms = torch.zeros_like(terms) if diagnose_routing else None
        x = roundtrip(self.x.float())
        if diagnose_input:
            from flashinfer import mxfp8_quantize

            # Match the baseline input quantizer, requesting linear scales only
            # to make the representation comparison independent of swizzling.
            cq, cs = mxfp8_quantize(self.x.contiguous(), False, 32)
            rq, rs = mxfp8_quantize_per_block_32(self.x.float(), torch.float8_e4m3fn)
            cs = cs.view(torch.uint8).reshape(self.capacity, -1)
            rs = rs.view(torch.uint8).reshape(self.capacity, -1)
            cx = dequant(cq, cs)
            emit(
                stage="input_quantization_comparison",
                fp8_byte_mismatches=int(
                    (cq.view(torch.uint8) != rq.view(torch.uint8)).sum().item()
                ),
                scale_byte_mismatches=int((cs != rs).sum().item()),
                dequantized_rel_l2=((cx - x).norm() / x.norm().clamp_min(1e-8)).item(),
                dequantized_max_abs=(cx - x).abs().max().item(),
            )
        # Broadcast only selected experts, in identical order on every rank.
        buffers = {
            k: torch.empty_like(v[0])
            for k, v in self.weights.items()
            if k != "expert_ids"
        }
        for e in selected:
            owner, local = divmod(e, 128)
            for key in ("w1", "s1", "w2", "s2"):
                if self.rank == owner:
                    buffers[key].copy_(self.weights[key][local])
                dist.broadcast(
                    buffers[key],
                    src=(
                        dist.get_global_rank(self.group, owner)
                        if self.group is not None
                        else owner
                    ),
                    group=self.group,
                )
            positions = (self.ids == e).nonzero()
            if not positions.numel():
                continue
            rows, slots = positions[:, 0], positions[:, 1]
            up, down = dequant(buffers["w1"], buffers["s1"]), dequant(
                buffers["w2"], buffers["s2"]
            )
            z = x[rows] @ up.T
            activated = (16 * torch.tanh(torch.relu(z) / 16)).square()
            if self.routing_weight_placement == "before_quantization":
                activated = activated * self.probs[rows, slots, None]
            a = roundtrip(activated)
            terms[rows, slots] = (a @ down.T).to(torch.bfloat16).float()
            products[rows, slots] = (a.double().abs() @ down.double().abs().T).float()
            if diagnose_routing:
                unweighted = roundtrip((16 * torch.tanh(torch.relu(z) / 16)).square())
                fc2 = unweighted @ down.T
                p = self.probs[rows, slots, None]
                late_terms[rows, slots] = fc2 * p
                late_bf16_terms[rows, slots] = fc2.to(torch.bfloat16).float() * p
        self.routing_diagnostics = {}
        if diagnose_routing:
            self.routing_diagnostics = {
                "routing_after_fc2_fp32": late_terms.sum(1).to(torch.bfloat16),
                "routing_after_fc2_bf16": late_bf16_terms.sum(1).to(torch.bfloat16),
            }
        if self.routing_weight_placement == "after_fc2":
            expected = weighted_bf16_recombine(terms, self.probs)
        else:
            expected = terms.sum(1).to(torch.bfloat16)
        return expected, terms, products

    def check(
        self,
        sample,
        baseline=None,
        graph=False,
        diagnose_input=False,
        diagnose_routing=False,
    ):
        self.stage(sample)
        captured = sample["baseline"].cuda()
        # First qualify the live adapter against the untouched model invocation.
        if baseline is not None:
            baseline_output = baseline().clone().reshape(-1, 768)
            compare(baseline_output, captured, "live_baseline_vs_capture")
            torch.testing.assert_close(
                baseline_output, baseline().reshape(-1, 768), rtol=0, atol=0
            )
            if graph:
                torch.cuda.synchronize()
                dist.barrier(group=self.group)
                baseline_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(baseline_graph):
                    baseline_graph_output = baseline()
                baseline_graph.replay()
                graph_copy = baseline_graph_output.clone().reshape(-1, 768)
                compare(graph_copy, captured, "graph_baseline_vs_capture")
                baseline_graph.replay()
                torch.testing.assert_close(
                    graph_copy, baseline_graph_output.reshape(-1, 768), rtol=0, atol=0
                )
        start = time.perf_counter()
        eager_output = self.run()
        torch.cuda.synchronize()
        emit(stage="first_launch", seconds=time.perf_counter() - start)
        eager_copy = eager_output.clone()
        run = self.run
        if graph:
            dist.barrier(group=self.group)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                output = self.run()

            def run():
                g.replay()
                return output

        actual = run().clone()
        torch.testing.assert_close(actual, eager_copy, rtol=0, atol=0)
        torch.testing.assert_close(actual, run(), rtol=0, atol=0)
        torch.cuda.synchronize()
        terms = self.layer._ensure_workspace().debug_combine_snapshot().float()
        expected, reference_terms, abs_products = self.reference(
            diagnose_input=diagnose_input, diagnose_routing=diagnose_routing
        )
        for label, candidate in self.routing_diagnostics.items():
            # Diagnostic hypotheses only: never substitute for the correctness gate.
            try:
                compare(captured, candidate[: self.n], "diagnostic_" + label)
            except AssertionError as error:
                emit(stage="routing_diagnostic_failure", label=label, error=str(error))
        metrics = validate_contributions(
            actual,
            expected,
            terms,
            reference_terms,
            self.ids,
            abs_products,
            4096,
            routing_weights=(
                self.probs if self.routing_weight_placement == "after_fc2" else None
            ),
        )
        emit(
            stage="contribution_validation",
            routing_weight_placement=self.routing_weight_placement,
            validation="bf16_fp32_contributions_v2",
            status="PASS",
            **metrics,
        )
        # Report both sides of the comparison before failing: a baseline/oracle
        # disagreement must not hide the direct MegaMoE/baseline measurement.
        comparison_errors = []
        for lhs, rhs, label in (
            (captured, expected[: self.n], "captured_baseline_vs_reference"),
            (actual[: self.n], captured, "megamoe_vs_baseline"),
        ):
            try:
                compare(lhs, rhs, label)
            except AssertionError as error:
                comparison_errors.append(str(error))
        if comparison_errors:
            raise AssertionError("; ".join(comparison_errors))
        if bool((actual[self.n :] != 0).any()):
            raise AssertionError("nonzero padded output")
        emit(
            stage="correctness",
            status="PASS",
            graph=graph,
            validation="bf16_fp32_contributions_v2",
            **metrics,
        )

    def close(self):
        from flashinfer.moe_ep import finalize_moe_ep_runtime

        self.layer.destroy()
        finalize_moe_ep_runtime(self.runtime)


def benchmark(replay, baseline, *, graph=False, warmup=20, iterations=100, rounds=5):
    """Call only after correctness. No snapshots, references or host copies in calls."""
    functions = {"megamoe": replay.run}
    if baseline is not None:
        functions = {"baseline": baseline, **functions}
    graphs = []
    for name, fn in list(functions.items()):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        if graph:
            dist.barrier(group=replay.group)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            graphs.append(g)
            functions[name] = g.replay
    samples = {name: [] for name in functions}
    walls = {name: [] for name in functions}
    torch.cuda.reset_peak_memory_stats()
    for round_id in range(rounds):
        for name in list(functions)[:: 1 if round_id % 2 == 0 else -1]:
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(iterations)
            ]
            dist.barrier(group=replay.group)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for begin, end in events:
                begin.record()
                functions[name]()
                end.record()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - start) * 1e6 / iterations
            local = [begin.elapsed_time(end) * 1000 for begin, end in events]
            ranks = gather(local, replay.group)
            samples[name].extend(max(values) for values in zip(*ranks))
            walls[name].append(max(gather(wall, replay.group)))
    report = {}
    for name, values in samples.items():
        values = torch.tensor(values, dtype=torch.float64)
        report[name] = dict(
            median_us=values.median().item(),
            p95_us=torch.quantile(values, 0.95).item(),
            batch_wall_us=walls[name],
        )
    if baseline is not None:
        report["baseline_over_megamoe"] = (
            report["baseline"]["median_us"] / report["megamoe"]["median_us"]
        )
    report["tokens_per_rank"] = gather(replay.n, replay.group)
    report["capacity"] = replay.capacity
    report["graph"] = graph
    report["peak_allocated_bytes"] = max(
        gather(torch.cuda.max_memory_allocated(), replay.group)
    )
    report["peak_reserved_bytes"] = max(
        gather(torch.cuda.max_memory_reserved(), replay.group)
    )
    report["memory_scope"] = "whole process including loaded model and both backends"
    emit(stage="benchmark", **report)
    return report


def profile_layers(replay, destination, *, baseline=None, graph=True, iterations=10):
    """Export per-rank Chrome traces after correctness and compilation warmup.

    Trace timings are diagnostic, not a replacement for unprofiled benchmarks.
    """
    from torch.profiler import ProfilerActivity, profile, record_function

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    functions = {"megamoe": replay.run}
    if baseline is not None:
        functions = {"baseline": baseline, **functions}
    for name, fn in functions.items():
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        dist.barrier(group=replay.group)
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            run = g.replay
            run()
            torch.cuda.synchronize()
        else:
            run = fn
        dist.barrier(group=replay.group)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for iteration in range(iterations):
                with record_function(f"nt4/{name}/iteration_{iteration}"):
                    run()
            torch.cuda.synchronize()
        filename = destination / f"{name}-rank{replay.rank}.json"
        prof.export_chrome_trace(str(filename))
        emit(
            stage="profile_saved",
            backend=name,
            graph=graph,
            iterations=iterations,
            path=str(filename),
        )
        dist.barrier(group=replay.group)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument(
        "--knobs-json",
        default="{}",
        help="explicit NT4 scheduling knobs; bypasses tuning cache",
    )
    parser.add_argument(
        "--benchmark-megamoe",
        action="store_true",
        help="time MegaMoE only after correctness; saved baseline is not executable",
    )
    parser.add_argument("--require-gb300", action="store_true")
    parser.add_argument(
        "--profile-dir", type=Path, help="export warmed per-rank CPU/CUDA Chrome traces"
    )
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument(
        "--routing-weight-placement",
        choices=("before_quantization", "after_fc2"),
        default="after_fc2",
    )
    parser.add_argument("--diagnose-input-quantization", action="store_true")
    parser.add_argument("--diagnose-routing-weight-placement", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="replay saved samples from a failed/partial capture",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    from nt4_gb300_tune import validate_knobs

    try:
        knobs = validate_knobs(json.loads(args.knobs_json))
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    if args.timeout_seconds < 1:
        parser.error("timeout-seconds must be positive")
    if args.profile_iterations < 1:
        parser.error("profile-iterations must be positive")
    watchdog = threading.Timer(
        args.timeout_seconds,
        lambda: (print("FAIL: replay watchdog expired", flush=True), os._exit(124)),
    )
    watchdog.daemon = True
    watchdog.start()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if args.require_gb300 and (
        torch.cuda.get_device_capability() != (10, 3)
        or "GB300" not in torch.cuda.get_device_name()
    ):
        raise RuntimeError("This tuning experiment requires GB300 (SM103)")
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    rank = dist.get_rank()
    manifest = json.loads((args.artifact / "manifest.json").read_text())
    if not manifest.get("capture_complete", False) and not args.allow_incomplete:
        raise ValueError(
            "capture incomplete; use --allow-incomplete to inspect saved samples"
        )
    if (
        manifest["schema"] != SCHEMA
        or manifest["world_size"] != 4
        or dist.get_world_size() != 4
    ):
        raise ValueError("artifact requires schema v1 and world size 4")
    part = manifest["ranks"][rank]
    weights = load_payload(args.artifact / f"weights-rank{rank}.pt", part["weights"])
    samples = [
        load_payload(args.artifact / f"sample{i}-rank{rank}.pt", hashes)
        for i, hashes in enumerate(part["samples"])
    ]
    counts = gather(len(samples))
    if min(counts) == 0 or min(counts) != max(counts):
        raise ValueError("incomplete artifact sample set")
    capacity = max(gather(max(s["x"].shape[0] for s in samples)))
    replay = Replay(
        weights,
        capacity,
        routing_weight_placement=args.routing_weight_placement,
        knobs=knobs,
    )
    emit(
        stage="tuning_config",
        knobs=knobs,
        gpu=torch.cuda.get_device_name(),
        cc=torch.cuda.get_device_capability(),
        sm_count=torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )
    for i, sample in enumerate(samples):
        emit(
            stage="sample",
            sample=i,
            comparison="saved baseline; no live baseline timing",
        )
        replay.check(
            sample,
            graph=args.graph,
            diagnose_input=args.diagnose_input_quantization,
            diagnose_routing=args.diagnose_routing_weight_placement,
        )
        if args.profile_dir:
            profile_layers(
                replay,
                args.profile_dir / f"sample{i}",
                graph=args.graph,
                iterations=args.profile_iterations,
            )
        if args.benchmark_megamoe:
            benchmark(replay, None, graph=args.graph)
    replay.close()
    dist.destroy_process_group()
    watchdog.cancel()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
