"""Real-route NT4 CUTLASS MXFP8 coordinate sweep; no model load or MCore import.

Times quantization + local CUTLASS experts + BF16 output, under a CUDA graph.
NVLS and shortcut overlap are excluded. Recommendations need E2E confirmation.
"""

import argparse
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist

from nt4_checkpoint_artifact import (
    SCHEMA,
    canonical_scale,
    load_payload,
    validate_sample,
    validate_weights,
)


def pack_scale(scale):
    """Logical E8M0 -> MCore/cuBLAS byte layout -> CUTLASS int32 ABI."""
    experts, rows, cols = scale.shape
    packed = (
        scale.reshape(experts, rows // 128, 4, 32, cols // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(experts, rows, cols)
    )
    for e in range(experts):
        if not torch.equal(canonical_scale(packed[e], rows, cols * 32), scale[e]):
            raise AssertionError("weight scale packing failed byte round trip")
    return packed.view(torch.int32)


def error_metrics(actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("nonfinite output")
    delta = actual.float() - expected.float()
    rel = float(delta.norm() / expected.float().norm().clamp_min(1e-8))
    row = float(
        (delta.norm(dim=-1) / expected.float().norm(dim=-1).clamp_min(1e-8)).max()
    )
    if rel > 0.015 or row > 0.015:
        raise AssertionError(f"provisional output gate: global={rel}, row={row}")
    if (actual[(expected == 0).all(-1)] != 0).any():
        raise AssertionError("nonzero output on empty reference row")
    return dict(global_rel_l2=rel, max_row_rel_l2=row, max_abs=float(delta.abs().max()))


def exact_comparison(actual, expected):
    equal = torch.equal(actual, expected)
    mismatch = actual != expected
    indices = mismatch.nonzero()[:8]
    delta = (actual.float() - expected.float()).abs()
    return dict(
        exact=equal,
        finite=bool(torch.isfinite(actual).all()),
        mismatched_elements=int(mismatch.sum()),
        max_abs=float(delta.max()),
        examples=[
            dict(
                index=index.tolist(),
                actual=float(actual[tuple(index)]),
                expected=float(expected[tuple(index)]),
            )
            for index in indices
        ],
    )


def select_pair(records, controls):
    """One common pair must improve every captured sample, using worst-rank time."""
    eligible = []
    for pair, times in records:
        ratios = [c / t for c, t in zip(controls, times)]
        eligible.append(
            dict(
                profile_ids=list(pair),
                worst_speedup=min(ratios),
                median_speedup=statistics.median(ratios),
            )
        )
    return max(eligible, key=lambda r: (r["worst_speedup"], r["median_speedup"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--rounds", type=int, default=3)
    a = p.parse_args()
    if min(a.warmup, a.iterations, a.rounds) < 1:
        p.error("warmup, iterations and rounds must be positive")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        run(a, rank)
    finally:
        dist.destroy_process_group()


def run(a, rank):
    from flashinfer import mxfp8_quantize
    from flashinfer.fused_moe import ActivationType, cutlass_fused_moe
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    from flashinfer.autotuner import AutoTuner
    from flashinfer.utils import device_support_pdl

    if dist.get_world_size() != 4:
        raise ValueError("requires four GPUs")
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    manifest = json.loads((a.artifact / "manifest.json").read_text())
    if (
        manifest["schema"] != SCHEMA
        or manifest["world_size"] != 4
        or not manifest.get("capture_complete")
    ):
        raise ValueError("requires complete NT4 MXFP8 EP4 artifact")
    weights = load_payload(
        a.artifact / f"weights-rank{rank}.pt", manifest["ranks"][rank]["weights"]
    )
    validate_weights(weights, rank)
    w1, w2 = [weights[k].cuda().view(torch.float8_e4m3fn) for k in ("w1", "w2")]
    scales = [
        pack_scale(weights["s1"]).cuda(),
        torch.ones(128, device="cuda"),
        pack_scale(weights["s2"]).cuda(),
        torch.ones(128, device="cuda"),
    ]
    limit = torch.tensor([16.0], dtype=torch.float32, device="cuda")
    samples = []
    count = len(manifest["ranks"][0]["samples"])
    if not count or any(len(r["samples"]) != count for r in manifest["ranks"]):
        raise ValueError("incomplete samples")
    census = []
    for i in range(count):
        parts = []
        for pe in range(4):
            part = load_payload(
                a.artifact / f"sample{i}-rank{pe}.pt",
                manifest["ranks"][pe]["samples"][i],
            )
            validate_sample(part)
            parts.append(part)
        s = {
            k: torch.cat([part[k] for part in parts]).cuda()
            for k in ("x", "ids", "probs", "baseline")
        }
        valid = (s["ids"] >= 0) & (s["probs"] > 0)
        hist = torch.bincount(s["ids"][valid], minlength=512)
        census.append(
            dict(
                sample=i,
                buffer_rows=int(s["x"].shape[0]),
                active_rows=int(valid.any(-1).sum()),
                expert_assignments=hist.cpu().tolist(),
            )
        )
        samples.append(s)

    selected = []

    def forward(s, pair):
        x, sf = mxfp8_quantize(s["x"].contiguous(), True, 32)
        out = cutlass_fused_moe(
            x,
            s["ids"].to(torch.int32),
            s["probs"],
            w1,
            w2,
            torch.bfloat16,
            quant_scales=scales,
            input_sf=sf,
            ep_size=4,
            ep_rank=rank,
            use_mxfp8_act_scaling=True,
            swizzled_input_sf=True,
            activation_type=ActivationType.ClampedRelu2,
            clamped_relu2_limit=limit,
            profile_ids=None if pair is None else list(pair),
        )
        return out[0] if isinstance(out, (list, tuple)) else out

    # Observe the actual auto-selected IDs without changing selection or cache.
    tuner = AutoTuner.get()
    original = tuner.choose_one

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        selected.append(dict(op=args[0], tactic=result[1]))
        return result

    tuner.choose_one = observe
    try:
        references = [forward(s, None).clone() for s in samples]
    finally:
        tuner.choose_one = original
    for s, ref in zip(samples, references):
        combined = ref.float().clone()
        dist.all_reduce(combined)
        error_metrics(combined.to(torch.bfloat16), s["baseline"])

    major, minor = torch.cuda.get_device_capability()
    runner = get_cutlass_fused_moe_module(str(major * 10 + minor)).MoERunner(
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=w1.dtype,
        output_dtype=torch.bfloat16,
        top_k=10,
        tp_size=1,
        tp_rank=0,
        ep_size=4,
        ep_rank=rank,
        cluster_size=1,
        cluster_rank=0,
        enable_alltoall=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=True,
        min_latency_mode=False,
        enable_pdl=device_support_pdl(w1.device),
        activation_type=ActivationType.ClampedRelu2,
        use_packed_weights=False,
        use_fused_finalize=True,
        use_wfp4afp8_humming=False,
    )
    x, _ = mxfp8_quantize(samples[0]["x"], True, 32)
    candidates = {}
    for stage in (1, 2):
        runner.gemm_idx_for_tuning = stage
        candidates[stage] = runner.get_valid_tactics([x, w1, None, w2, None], None)
    candidate_sets = [None] * 4
    dist.all_gather_object(candidate_sets, candidates)
    if any(c != candidates for c in candidate_sets):
        raise ValueError("ranks enumerate different tactics; use matching GPUs/builds")
    audit = dict(
        rank=rank,
        artifact=str(a.artifact),
        gpu=torch.cuda.get_device_name(),
        candidates=candidates,
        auto_selected=selected,
        census=census,
        tactic_geometry="IDs and validity metadata only; tile shapes are not exposed by this Python runner",
        scope="Quantization + local CUTLASS expert operation; graph replay; no NVLS/shortcut overlap. Provisional output gates.",
    )
    (a.output / f"audit-rank{rank}.json").write_text(json.dumps(audit, indent=2))

    def measure(pair, label):
        times = []
        for i, (s, ref) in enumerate(zip(samples, references)):
            if rank == 0:
                print(
                    json.dumps(
                        dict(
                            stage="tactic_start",
                            label=label,
                            profile_ids=pair,
                            sample=i,
                        )
                    ),
                    flush=True,
                )
            # Preserve independent snapshots across warmup and repeated launches.
            out = forward(s, pair).clone()
            validation_error = None
            metrics = {}
            try:
                metrics = error_metrics(out, ref)
            except AssertionError as error:
                validation_error = str(error)
            eager_checks = [
                exact_comparison(forward(s, pair).clone(), out) for _ in range(2)
            ]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(a.warmup):
                    forward(s, pair)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_out = forward(s, pair)
            graph_checks = []
            for _ in range(3):
                graph.replay()
                torch.cuda.synchronize()
                graph_checks.append(exact_comparison(graph_out, out))
            passed = validation_error is None and all(
                c["exact"] for c in eager_checks + graph_checks
            )
            verdicts = [None] * dist.get_world_size()
            dist.all_gather_object(verdicts, passed)
            if not all(verdicts):
                diagnostic = dict(
                    stage="tactic_rejected",
                    rank=rank,
                    label=label,
                    profile_ids=pair,
                    sample=i,
                    error=validation_error,
                    eager_repeat=eager_checks,
                    graph_vs_eager=graph_checks,
                    passing_ranks=verdicts,
                    eligible=False,
                )
                with (a.output / f"rejections-rank{rank}.jsonl").open("a") as f:
                    f.write(json.dumps(diagnostic) + "\n")
                print(json.dumps(diagnostic), flush=True)
                del graph, graph_out
                if label in ("auto-start", "auto-end", "finalist-confirm"):
                    raise AssertionError(
                        f"{label} failed validation; see rejections-rank*.jsonl"
                    )
                return None  # All ranks reject together; never time/rank this pair.
            elapsed = []
            for _ in range(a.rounds):
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(a.iterations):
                    graph.replay()
                end.record()
                end.synchronize()
                value = torch.tensor(
                    start.elapsed_time(end) * 1000 / a.iterations, device="cuda"
                )
                dist.all_reduce(value, op=dist.ReduceOp.MAX)
                elapsed.append(value.item())
            us = statistics.median(elapsed)
            times.append(us)
            row = dict(
                stage="cutlass_tactic",
                label=label,
                profile_ids=pair,
                sample=i,
                worst_rank_median_us=us,
                rounds_us=elapsed,
                metrics=metrics,
            )
            with (a.output / f"trials-rank{rank}.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            if rank == 0:
                print(json.dumps(row), flush=True)
            del graph, graph_out
        return times

    controls = measure(None, "auto-start")
    records = []
    # Coordinate descent, not an exhaustive Cartesian search. Score whole MoE.
    current = (-1, -1)
    for stage in (1, 2, 1):
        sweep = []
        for tactic in candidates[stage]:
            pair = list(current)
            pair[stage - 1] = tactic
            pair = tuple(pair)
            times = measure(pair, f"fc{stage}")
            if times is not None:
                sweep.append((pair, times))
                records.append((pair, times))
        if not sweep:
            raise RuntimeError(
                f"no FC{stage} candidate passed; inspect rejection diagnostics"
            )
        current = tuple(select_pair(sweep, controls)["profile_ids"])
    end = measure(None, "auto-end")
    faster = [min(x, y) for x, y in zip(controls, end)]
    best = select_pair(records, faster)
    drift = max(abs(x / y - 1) for x, y in zip(controls, end))
    confirmed = measure(tuple(best["profile_ids"]), "finalist-confirm")
    confirmed_speedup = min(c / t for c, t in zip(faster, confirmed))
    best.update(
        control_drift=drift,
        confirmed_worst_speedup=confirmed_speedup,
        eligible=drift <= 0.03
        and min(best["worst_speedup"], confirmed_speedup) >= 1.02,
        validation="provisional output L2 + exact graph/eager",
        e2e_required=True,
        scope=audit["scope"],
    )
    if rank == 0:
        (a.output / "recommendation.json").write_text(json.dumps(best, indent=2))
        print(json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
