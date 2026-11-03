"""Capture a baseline-agreement failure; replay CUTLASS without MCore/checkpoint.

Run replay with torchrun --nproc-per-node=4. No performance claims or gate changes.
"""

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from nt4_checkpoint_artifact import SCHEMA, load_payload, save_manifest, save_payload


def capture_failure(replay, sample, actual, reference, errors):
    root = os.environ.get("NT4_BASELINE_FAILURE_ROOT")
    if not root:
        return
    group = replay.group
    rank = dist.get_rank(group)
    stamp = [time.time_ns() if rank == 0 else None]
    src = dist.get_global_rank(group, 0) if group is not None else 0
    dist.broadcast_object_list(stamp, src=src, group=group)
    directory = (
        Path(root) / f"layer{getattr(replay, 'layer_number', 'unknown')}-{stamp[0]}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    part = None
    error = None
    try:
        weights = save_payload(directory / f"weights-rank{rank}.pt", replay.weights)
        payload = {k: sample[k] for k in ("x", "ids", "probs", "baseline")}
        payload.update(megamoe=actual[: replay.n], reference=reference[: replay.n])
        hashes = save_payload(directory / f"sample0-rank{rank}.pt", payload)
        part = dict(weights=weights, samples=[hashes], results=[])
    except Exception as exc:
        error = str(exc)
    reports = [None] * dist.get_world_size(group)
    dist.all_gather_object(reports, dict(part=part, error=error), group=group)
    if rank == 0:
        save_manifest(
            directory / "manifest.json",
            dict(
                schema=SCHEMA,
                world_size=len(reports),
                layer=getattr(replay, "layer_number", None),
                capacity=replay.capacity,
                ranks=[r["part"] for r in reports],
                capture_complete=all(r["error"] is None for r in reports),
                capture_errors=[r["error"] for r in reports],
                validation_errors=errors,
                provenance=dict(
                    scope="live baseline-agreement failure; not a passing validation",
                    cutlass_decode_profile_ids=os.environ.get(
                        "NT4_CUTLASS_DECODE_PROFILE_IDS"
                    ),
                    knobs=replay.knobs,
                    torch=str(torch.__version__),
                ),
            ),
        )
        print(
            json.dumps(
                dict(
                    stage="baseline_failure_saved",
                    artifact=str(directory),
                    complete=all(r["error"] is None for r in reports),
                )
            ),
            flush=True,
        )


def comparison(a, b):
    a, b = a.double(), b.double()
    row = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-8)
    worst = int(row.argmax())
    return dict(
        global_rel_l2=float((a - b).norm() / b.norm().clamp_min(1e-8)),
        max_row_rel_l2=float(row[worst]),
        worst_row=worst,
        reference_norm=float(b[worst].norm()),
        max_abs=float((a - b).abs().max()),
        exact=bool(torch.equal(a, b)),
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    )


def isolate_slot(probs, slot):
    """Keep routing IDs/geometry unchanged; retain only one weight column."""
    result = torch.zeros_like(probs)
    result[:, slot] = probs[:, slot]
    return result


def boundary_candidates(reference, lower, upper, quantize):
    """One-block endpoint alternatives, each independently boundary-validated.

    This is a bounded search, not an inversion of the kernel's intermediate.
    It cannot exclude multi-block alternatives when no single block explains it.
    """
    from nt4_expert_diagnostics import boundary_blocks

    seen = set()
    for endpoint, value in [("lower", lower), ("upper", upper)]:
        q, scales = quantize(value)
        actual = q.double() * scales.repeat_interleave(32, -1)
        allowed = boundary_blocks(actual, reference, lower, upper, scales, quantize)
        changed = (actual.reshape(-1, 32) != reference.reshape(-1, 32)).any(-1)
        for block in (allowed.reshape(-1) & changed).nonzero().flatten().tolist():
            values = actual.reshape(-1, 32)[block]
            key = (block, tuple(values.tolist()))
            if key in seen:
                continue
            seen.add(key)
            candidate = reference.clone()
            candidate.reshape(-1, 32)[block] = values
            yield endpoint, block, candidate


def snapshot_scale_indices(row, width):
    """CUTLASS Sm1xxBlockScaledConfig<32> scale atom: 128 rows x 4 scales."""
    k = torch.arange(width // 32)
    return (
        ((row // 128) * ((width // 32 + 3) // 4) + k // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + k % 4
    )


def inspect_snapshot(prefix, s, weights, forward):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        mxfp8_quantize_per_block_32,
    )
    from nt4_expert_diagnostics import (
        boundary_blocks,
        decode_e8m0_exact,
        fc1_activation_interval,
    )

    token, slot, expert = 155, 5, 180
    if int(s["ids"][token, slot]) != expert:
        raise ValueError("snapshot target no longer matches saved routes")
    probs = isolate_slot(s["probs"], slot)
    before = forward(probs)
    os.environ["NT4_CUTLASS_FC1_SNAPSHOT_PREFIX"] = str(prefix)
    try:
        actual = forward(probs)
    finally:
        os.environ.pop("NT4_CUTLASS_FC1_SNAPSHOT_PREFIX", None)
    meta = json.loads(Path(str(prefix) + "-meta.json").read_text())
    if (meta["sf_m_alignment"], meta["sf_k_alignment"]) != (128, 128):
        raise ValueError("unsupported CUTLASS scale atom; do not guess layout")
    q = torch.frombuffer(
        bytearray(Path(str(prefix) + "-q.bin").read_bytes()), dtype=torch.uint8
    )
    sf = torch.frombuffer(
        bytearray(Path(str(prefix) + "-sf.bin").read_bytes()), dtype=torch.uint8
    )
    if q.numel() != meta["width"] or sf.numel() != meta["sf_bytes"]:
        raise ValueError("incomplete snapshot")
    row = meta["permuted_row"] - meta["expert_begin"]
    indices = snapshot_scale_indices(row, meta["width"])
    scales = decode_e8m0_exact(sf[indices]).cuda()[None, :]
    intermediate = q.cuda().view(torch.float8_e4m3fn).double()[
        None, :
    ] * scales.repeat_interleave(32, -1)
    local = weights["expert_ids"].tolist().index(expert)

    def dequant(q, scale):
        return q.cuda().view(torch.float8_e4m3fn).double() * decode_e8m0_exact(
            scale.cuda().view(torch.uint8)
        ).repeat_interleave(32, -1)

    def quantize(v):
        q, sf = mxfp8_quantize_per_block_32(v.float().contiguous(), torch.float8_e4m3fn)
        return q, sf.double()

    xq, xs = quantize(s["x"][token : token + 1])
    x = xq.double() * xs.repeat_interleave(32, -1)
    up = dequant(weights["w1"][local], weights["s1"][local])
    down = dequant(weights["w2"][local], weights["s2"][local])
    z = x @ up.T
    activation = (16 * torch.tanh(z.clamp_min(0) / 16)).square()
    rq, rs = quantize(activation)
    reference = rq.double() * rs.repeat_interleave(32, -1)
    lo, hi = fc1_activation_interval(x, up, z)
    boundary_details = []
    accepted = boundary_blocks(
        intermediate, reference, lo, hi, scales, quantize, diagnostics=boundary_details
    )
    exact = (intermediate.reshape(-1, 32) == reference.reshape(-1, 32)).all(-1)
    predicted = (
        (intermediate @ down.T).to(torch.bfloat16).float()
        * s["probs"][token, slot].float()
    ).to(torch.bfloat16)
    changed = (intermediate != reference).nonzero()[:, -1]
    return dict(
        metadata=meta,
        snapshot_vs_undumped=comparison(actual, before),
        fc1_quantized_vs_reference=comparison(intermediate, reference),
        unexplained_blocks=int((~(accepted | exact)).sum()),
        boundary_details=boundary_details,
        changed_channels=changed.tolist(),
        actual_quantized=intermediate[0, changed].tolist(),
        reference_quantized=reference[0, changed].tolist(),
        activation_lower=lo[0, changed].tolist(),
        activation_upper=hi[0, changed].tolist(),
        fc2_and_weighting_using_actual_fc1=comparison(
            actual[token : token + 1], predicted
        ),
        raw_fc1_accumulator_available=False,
        scope="actual post-activation MXFP8 FC1 snapshot; FC2 plus weighting compared jointly",
    )


def probe_slots(s, weights, forward, original):
    """Diagnostic interventions, not raw fused-kernel intermediate snapshots."""
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        mxfp8_quantize_per_block_32,
    )
    from nt4_expert_diagnostics import decode_e8m0_exact, fc1_activation_interval

    token = comparison(s["baseline"], s["reference"])["worst_row"]
    xq, xs = mxfp8_quantize_per_block_32(
        s["x"][token : token + 1].float().contiguous(), torch.float8_e4m3fn
    )
    x = xq.double() * xs.double().repeat_interleave(32, -1)
    expert_ids = weights["expert_ids"].tolist()
    isolated = []
    records = []
    reference_sum = torch.zeros_like(x)
    for slot in range(s["ids"].shape[1]):
        # Do not drop routes, shrink top-k, or alter expert batching.
        out = forward(isolate_slot(s["probs"], slot))
        isolated.append(out[token : token + 1].float())
        expert = int(s["ids"][token, slot])
        if expert not in expert_ids:
            continue
        local = expert_ids.index(expert)

        def dequant(q, scale):
            return q.cuda().view(torch.float8_e4m3fn).double() * decode_e8m0_exact(
                scale.cuda().view(torch.uint8)
            ).repeat_interleave(32, -1)

        up = dequant(weights["w1"][local], weights["s1"][local])
        down = dequant(weights["w2"][local], weights["s2"][local])
        z = x @ up.T
        activation = (16 * torch.tanh(z.clamp_min(0) / 16)).square()
        aq, scales = mxfp8_quantize_per_block_32(
            activation.float().contiguous(), torch.float8_e4m3fn
        )
        activated = aq.double() * scales.double().repeat_interleave(32, -1)
        term = (activated @ down.T).to(torch.bfloat16)
        weighted = term.float() * s["probs"][token, slot].float()
        reference_sum += weighted.double()
        observed = out[token : token + 1]
        comparison_record = comparison(observed, weighted.to(torch.bfloat16))
        search = None
        if comparison_record["global_rel_l2"] > 0.001:
            lower, upper = fc1_activation_interval(x, up, z)

            def quantize(v):
                q, sf = mxfp8_quantize_per_block_32(
                    v.float().contiguous(), torch.float8_e4m3fn
                )
                return q, sf.double()

            best = None
            count = 0
            for endpoint, block, candidate in boundary_candidates(
                activated, lower, upper, quantize
            ):
                candidate_term = (candidate @ down.T).to(torch.bfloat16)
                predicted = (
                    candidate_term.float() * s["probs"][token, slot].float()
                ).to(torch.bfloat16)
                result = comparison(observed, predicted)
                count += 1
                if (
                    best is None
                    or result["global_rel_l2"] < best["comparison"]["global_rel_l2"]
                ):
                    channels = (candidate != activated).nonzero()[:, -1]
                    best = dict(
                        endpoint=endpoint,
                        block=block,
                        channels=channels.tolist(),
                        reference_quantized=activated[0, channels].tolist(),
                        alternative_quantized=candidate[0, channels].tolist(),
                        activation_lower=lower[0, channels].tolist(),
                        activation_upper=upper[0, channels].tolist(),
                        comparison=result,
                    )
            search = dict(
                scope="single-block endpoint witnesses; provisional independent bound",
                candidates=count,
                best=best,
                actual_fc1_intermediate_observed=False,
            )
        records.append(
            dict(
                slot=slot,
                expert=expert,
                weight=float(s["probs"][token, slot]),
                isolated_vs_fp64_staged_reference=comparison_record,
                boundary_search=search,
            )
        )
    reconstructed = torch.stack(isolated).sum(0).to(torch.bfloat16)
    return dict(
        token=token,
        scope="weight-slot interventions; FC1/FC2 intermediates not observed",
        reconstruction_vs_original=comparison(
            reconstructed, original[token : token + 1]
        ),
        original_vs_local_fp64_staged_reference=comparison(
            original[token : token + 1], reference_sum.to(torch.bfloat16)
        ),
        experts=records,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--probe-slots", action="store_true")
    p.add_argument("--snapshot-fc1", action="store_true")
    a = p.parse_args()
    if a.snapshot_fc1:
        os.environ["NT4_CUTLASS_FC1_SNAPSHOT_BUILD"] = "1"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    try:
        run(a)
    finally:
        dist.destroy_process_group()


def run(a):
    from flashinfer import mxfp8_quantize
    from flashinfer.fused_moe import ActivationType, cutlass_fused_moe
    from nt4_cutlass_decode_tune import pack_scale

    rank = dist.get_rank()
    manifest = json.loads((a.artifact / "manifest.json").read_text())
    if (
        dist.get_world_size() != 4
        or manifest["schema"] != SCHEMA
        or not manifest["capture_complete"]
    ):
        raise ValueError("requires complete EP4 capture")
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    weights = load_payload(
        a.artifact / f"weights-rank{rank}.pt", manifest["ranks"][rank]["weights"]
    )
    parts = [
        load_payload(
            a.artifact / f"sample0-rank{i}.pt", manifest["ranks"][i]["samples"][0]
        )
        for i in range(4)
    ]
    s = {k: torch.cat([v[k] for v in parts]).cuda() for k in parts[0]}
    w1, w2 = [weights[k].cuda().view(torch.float8_e4m3fn) for k in ["w1", "w2"]]
    scales = [
        pack_scale(weights["s1"]).cuda(),
        torch.ones(128, device="cuda"),
        pack_scale(weights["s2"]).cuda(),
        torch.ones(128, device="cuda"),
    ]
    limit = torch.tensor([16.0], device="cuda", dtype=torch.float32)
    x, sf = mxfp8_quantize(s["x"].contiguous(), True, 32)
    report = dict(
        layer=manifest["layer"],
        scope="CUTLASS local expert calls plus FP32 all-reduce; NVLS not reproduced",
        captured_megamoe_vs_reference=comparison(s["megamoe"], s["reference"]),
        captured_cutlass_vs_reference=comparison(s["baseline"], s["reference"]),
        variants={},
    )
    for label, pair in [("auto", None), ("39_428", [39, 428])]:

        def forward(probs=None):
            out = cutlass_fused_moe(
                x,
                s["ids"].to(torch.int32),
                s["probs"] if probs is None else probs,
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
                profile_ids=pair,
            )
            return (out[0] if isinstance(out, (list, tuple)) else out).clone()

        first = forward()
        second = forward()
        combined = first.float()
        dist.all_reduce(combined)
        combined = combined.to(torch.bfloat16)
        record = dict(
            local_repeat=comparison(first, second),
            vs_captured=comparison(combined, s["baseline"]),
            vs_reference=comparison(combined, s["reference"]),
            vs_megamoe=comparison(combined, s["megamoe"]),
        )
        report["variants"][label] = record
        if a.probe_slots:
            record["slot_probe"] = probe_slots(s, weights, forward, first)
        if a.snapshot_fc1 and rank == 1 and label == "39_428":
            record["fc1_snapshot"] = inspect_snapshot(
                a.output / "expert180", s, weights, forward
            )
        # Preserve local terms for further FC1/FC2 diagnosis; no checkpoint needed.
        torch.save(
            dict(local=first.cpu(), combined=combined.cpu()),
            a.output / f"{label}-rank{rank}.pt",
        )
    (a.output / f"diagnostic-rank{rank}.json").write_text(json.dumps(report, indent=2))
    if rank == 0:
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
