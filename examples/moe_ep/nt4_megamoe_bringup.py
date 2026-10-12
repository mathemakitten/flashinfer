"""GB300 bring-up for the explicit NT4 MegaMoE development variant.

Run with torchrun; see nt4_megamoe_bringup.md. No model checkpoint is needed.
The baseline mode is SwiGLU infrastructure validation, not NT4 inference.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import threading
import traceback


def weighted_bf16_recombine(terms, weights):
    """Reference the top-k reducer's slot-ordered FP32 multiply/FMA sequence.

    BF16 terms times FP32 weights are evaluated in FP64 before rounding each
    accumulator update to FP32, avoiding a separate FP32 product rounding.
    """
    import torch

    acc = torch.zeros_like(terms[:, 0], dtype=torch.float32)
    for slot in range(terms.shape[1]):
        acc = (
            terms[:, slot].double() * weights[:, slot, None].double() + acc.double()
        ).float()
    return acc.to(torch.bfloat16)


def validate_contributions(
    actual,
    expected,
    kernel_terms,
    reference_terms,
    ids,
    fc2_abs_products=None,
    reduction_size=0,
    routing_weights=None,
):
    """Provisional BF16 + FP32 dot-product roundoff gate.

    Two FP32 accumulation orders can differ by 2*gamma_K*sum(abs(a*w)),
    gamma_K=K*u/(1-K*u), u=2**-24. This conservative forward-error bound
    assumes identical finite operands and normal-range FP32 arithmetic; it
    does not bound upstream activation/quantization differences. A one-value
    BF16 allowance handles small differences there. Per-expert vector and
    global L2 gates independently limit aggregate error.
    """
    import torch

    for name, value in (
        ("actual", actual),
        ("expected", expected),
        ("kernel_terms", kernel_terms),
        ("reference_terms", reference_terms),
    ):
        if not bool(torch.isfinite(value).all()):
            raise AssertionError(f"{name} contains nonfinite values")

    def bf16_step(value):
        v = value.to(torch.bfloat16)
        up = torch.nextafter(v, torch.full_like(v, float("inf"))).float()
        down = torch.nextafter(v, torch.full_like(v, -float("inf"))).float()
        return torch.maximum(up - v.float(), v.float() - down)

    ref = reference_terms.float()
    terms = kernel_terms.float()
    delta = (terms - ref).abs()
    neighbor = torch.nextafter(ref.to(torch.bfloat16), terms.to(torch.bfloat16))
    budget = (neighbor.float() - ref).abs()
    # Do not grant a rounding allowance to exact zero contributions.
    budget = torch.where(ref == 0, torch.zeros_like(budget), budget)
    if fc2_abs_products is not None:
        if not 0 < reduction_size * 2**-24 < 1:
            raise ValueError("invalid FP32 reduction size")
        if not bool(torch.isfinite(fc2_abs_products).all()) or bool(
            (fc2_abs_products < 0).any()
        ):
            raise AssertionError("invalid absolute-product sums")
        ku = reduction_size * 2**-24
        budget = budget + (2 * ku / (1 - ku)) * fc2_abs_products
    inactive = (ids < 0).unsqueeze(-1).expand_as(terms)
    budget = torch.where(inactive, torch.zeros_like(budget), budget)
    bad_terms = delta > budget
    if bool(bad_terms.any()):
        index = tuple(bad_terms.nonzero()[0].tolist())
        raise AssertionError(
            f"expert term exceeds rounding budget at {index}: "
            f"actual={terms[index].item()}, reference={ref[index].item()}, "
            f"budget={budget[index].item()}"
        )
    if bool((terms.masked_select(inactive) != 0).any()):
        raise AssertionError("masked expert contribution is nonzero")
    zero_outputs = (
        ref.abs().sum(dim=1) == 0
        if fc2_abs_products is None
        else fc2_abs_products.sum(dim=1) == 0
    )
    if bool((actual[zero_outputs] != 0).any()):
        raise AssertionError("zero-contribution output is nonzero")

    if routing_weights is not None:
        if (
            routing_weights.shape != ids.shape
            or not bool(torch.isfinite(routing_weights).all())
            or bool((routing_weights < 0).any())
        ):
            raise ValueError("invalid late routing weights")
        weights = torch.where(ids < 0, 0, routing_weights).float().unsqueeze(-1)
    else:
        weights = torch.ones_like(terms[..., :1])
    recombined = (
        weighted_bf16_recombine(terms, weights.squeeze(-1))
        if routing_weights is not None
        else terms.sum(dim=1).to(torch.bfloat16)
    )
    torch.testing.assert_close(actual, recombined, rtol=0, atol=0)
    output_budget = (budget * weights).sum(dim=1) + 0.5 * (
        bf16_step(actual) + bf16_step(expected)
    )
    if bool(((actual.float() - expected.float()).abs() > output_budget).any()):
        raise AssertionError("output exceeds cancellation-aware rounding budget")
    expert_rel_l2 = (terms - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-8)
    if bool((expert_rel_l2 > 0.015).any()):
        raise AssertionError("per-expert vector relative L2 exceeds 0.015")
    rel_l2 = (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-8)
    ).item()
    if rel_l2 > 0.015:
        raise AssertionError(f"relative L2 exceeds 0.015: {rel_l2}")
    return dict(
        expert_max_abs=delta.max().item(),
        expert_max_budget_fraction=(delta / budget.clamp_min(1e-30)).max().item(),
        expert_max_rel_l2=expert_rel_l2.max().item(),
        rel_l2=rel_l2,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preflight", "baseline", "nt4"), default="preflight"
    )
    parser.add_argument("--full-shape", action="store_true")
    parser.add_argument("--checks", type=int, default=8)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--prequantized", action="store_true")
    parser.add_argument("--weights", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--skip-empty-rank",
        action="store_true",
        help="Diagnostic control only; does not qualify empty-rank support",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.checks < 2 or args.timeout_seconds < 1:
        parser.error("checks must be >=2 and timeout-seconds must be positive")
    if not 0 <= args.seed < 2**31:
        parser.error("seed must be in [0, 2**31)")
    if not math.isfinite(args.input_scale) or args.input_scale <= 0:
        parser.error("input-scale must be positive and finite")

    def timeout():
        print("FAIL: bring-up watchdog expired", flush=True)
        os._exit(124)

    watchdog = threading.Timer(args.timeout_seconds, timeout)
    watchdog.daemon = True
    watchdog.start()
    import torch
    import flashinfer

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required; a skipped test is not a successful preflight"
        )
    torch.cuda.set_device(local_rank)
    # The reference uses FP32 GEMMs, not TF32 approximations of them.
    torch.backends.cuda.matmul.allow_tf32 = False
    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3)):
        raise RuntimeError(f"This bring-up targets data-center Blackwell, got {cc}")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
        bootstrap_moe_ep_runtime,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        mxfp8_quantize_per_block_32,
    )

    versions = {}
    for name in (
        "torch",
        "flashinfer-python",
        "nvidia-cutlass-dsl",
        "nvshmem4py-cu13",
        "cuda-python",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "distribution metadata absent"
    # Resolve the actual dependencies, not just wheel names/version strings.
    import cutlass.cute  # noqa: F401
    import nvshmem.core  # noqa: F401
    import cuda.bindings.driver  # noqa: F401

    print(
        json.dumps(
            dict(
                rank=rank,
                mode=args.mode,
                gpu=torch.cuda.get_device_name(),
                cc=cc,
                cuda=torch.version.cuda,
                nccl=torch.cuda.nccl.version(),
                flashinfer_source=flashinfer.__file__,
                versions=versions,
            )
        ),
        flush=True,
    )
    if args.mode == "preflight":
        watchdog.cancel()
        return
    if world != 4:
        raise RuntimeError(
            "Use torchrun --standalone --nproc-per-node=4 for this EP4 test"
        )

    hidden, intermediate, experts = (
        (768, 4096, 512) if args.full_shape else (128, 256, 16)
    )
    topk, capacity = 10, 4
    is_nt4 = args.mode == "nt4"
    config = Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=intermediate,
        top_k=topk,
        activation="clamped_relu2" if is_nt4 else "swiglu",
        clamped_relu2_limit=16.0 if is_nt4 else None,
        enable_in_kernel_fc2_reduce=False,
        knobs={},
    )
    bootstrap = BootstrapConfig(world_size=world, rank=rank)
    kernel = create_mega_kernel(config)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, kernel.runtime_requirements(bootstrap)
    )
    layer = None
    try:
        local_experts = experts // world
        first_expert = rank * local_experts
        w1 = torch.zeros(
            local_experts, intermediate, hidden, dtype=torch.bfloat16, device="cuda"
        )
        w2 = torch.zeros(
            local_experts, hidden, intermediate, dtype=torch.bfloat16, device="cuda"
        )
        j = torch.arange(intermediate, device="cuda")
        h = torch.arange(hidden, device="cuda")

        def dense_expert(e):
            # Seed by GLOBAL expert ID, independent of ownership and call order.
            # Reference regeneration avoids gathering E full expert matrices.
            generator = torch.Generator(device="cuda")
            generator.manual_seed(args.seed + 2 * e)
            up = (
                torch.randn(intermediate, hidden, device="cuda", generator=generator)
                / math.sqrt(hidden)
            ).to(torch.bfloat16)
            generator.manual_seed(args.seed + 2 * e + 1)
            down = (
                torch.randn(hidden, intermediate, device="cuda", generator=generator)
                / math.sqrt(intermediate)
            ).to(torch.bfloat16)
            return up, down

        for le in range(local_experts):
            e = first_expert + le
            if args.weights == "dense":
                up, down = dense_expert(e)
                w1[le].copy_(up)
                w2[le].copy_(down)
                del up, down
            else:
                w1[le, j, (j + e) % hidden] = 2.0 ** (e % 3 - 1)
                w2[le, h, (h + e * 7) % intermediate] = 1.0
                w2[le, h, (h + e * 7 + intermediate // 2) % intermediate] = 0.5
        # Baseline deliberately has gate+up legs. NT4 has exactly I FC1 rows.
        fc1 = w1 if is_nt4 else torch.cat((w1, w1), dim=1)
        if args.prequantized:
            q1, sf1 = mxfp8_quantize_per_block_32(
                fc1.float().reshape(-1, hidden), torch.float8_e4m3fn
            )
            q2, sf2 = mxfp8_quantize_per_block_32(
                w2.float().reshape(-1, intermediate), torch.float8_e4m3fn
            )
            pack = MoEWeightPack(
                w13=q1.reshape(fc1.shape),
                w2=q2.reshape(w2.shape),
                w13_scale=sf1.reshape(local_experts, fc1.shape[1], hidden // 32),
                w2_scale=sf2.reshape(local_experts, hidden, intermediate // 32),
            )
        else:
            pack = MoEWeightPack(w13=fc1, w2=w2)
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=world, rank=rank, auto_bootstrap=False
            ),
            fleet_params=FleetParams(
                num_experts=experts,
                max_tokens_per_rank=capacity,
                token_hidden_size=hidden,
            ),
            weights=pack,
            backend=MegaConfig(
                megakernel=config, quantize_input=True, preprocess_weights=True
            ),
        )
        del pack, fc1, w1, w2
        if args.prequantized:
            del q1, q2, sf1, sf2

        x = torch.empty(capacity, hidden, dtype=torch.bfloat16, device="cuda")
        ids = torch.empty(capacity, topk, dtype=torch.int64, device="cuda")
        probs = torch.empty(capacity, topk, dtype=torch.float32, device="cuda")
        inputs = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=probs)

        def fill(epoch):
            x.copy_(
                (
                    (
                        torch.arange(capacity * hidden, device="cuda").reshape(
                            capacity, hidden
                        )
                        + epoch * 7
                        + rank * 11
                    )
                    % 33
                    - 16
                ).float()
                / 4
                * args.input_scale
            )
            route = torch.arange(capacity * topk, device="cuda").reshape(capacity, topk)
            if args.weights == "dense":
                # Exercise every owner even at E=512, rather than selecting
                # only the first rank's experts in the full-shape test.
                ids.copy_(
                    (route % world) * local_experts
                    + (route // world + rank * 13 + epoch * 17) % local_experts
                )
            else:
                ids.copy_((route + rank * 13 + epoch * 3) % experts)
            probs.copy_(((route % 4).float() + 1) * (3.16 / 16))
            # A completely empty receiver/producer rank, with fixed capacity.
            if epoch % 3 == 1 and rank == 0 and not args.skip_empty_rank:
                ids.fill_(-1)
                probs.zero_()
            # Mix valid and masked slots after the completely empty epoch.
            # Other ranks still supply contributions to this rank's valid slots.
            elif epoch % 3 == 2 and rank == 0 and not args.skip_empty_rank:
                ids[:, 1::2] = -1
                probs[:, 1::2] = 0

        def roundtrip(value):
            q, sf = mxfp8_quantize_per_block_32(value.contiguous(), torch.float8_e4m3fn)
            return q.float() * sf.float().repeat_interleave(32, dim=-1)

        def oracle(*, fp64_gemm=False):
            # Sparse power-of-two weights are exactly representable in MXFP8.
            # This evaluates global expert IDs directly, independently of the
            # kernel's dispatch, packing, peer offsets and reduction machinery.
            staged_x = roundtrip(x.float())
            terms = torch.zeros(capacity, topk, hidden, device="cuda")
            abs_products = torch.zeros_like(terms)
            if args.weights == "dense":
                routes = ids.cpu().tolist()
                selected = sorted({e for row in routes for e in row if e >= 0})
                for e in selected:
                    positions = [
                        (row, slot)
                        for row in range(capacity)
                        for slot in range(topk)
                        if routes[row][slot] == e
                    ]
                    rows, slots = zip(*positions)
                    rows = torch.tensor(rows, device="cuda", dtype=torch.long)
                    slots = torch.tensor(slots, device="cuda", dtype=torch.long)
                    up, down = dense_expert(e)
                    # Quantize canonical weights along the GEMM reduction axis.
                    # Never use the kernel's packed/swizzled weights in the oracle.
                    up = roundtrip(up.float())
                    down = roundtrip(down.float())
                    # Failure-only cross-check isolates reference GEMM rounding;
                    # activation/quantization still follow the FP32 contract.
                    z = (
                        (staged_x[rows].double() @ up.double().T).float()
                        if fp64_gemm
                        else staged_x[rows] @ up.T
                    )
                    if is_nt4:
                        a = (16 * torch.tanh(torch.relu(z) / 16)).square()
                    else:
                        a = torch.nn.functional.silu(z) * z
                    a = roundtrip(a * probs[rows, slots, None])
                    fc2 = a.double() @ down.double().T if fp64_gemm else a @ down.T
                    terms[rows, slots] = fc2.to(torch.bfloat16).float()
                    # FP64 prevents underestimating the positive sum through
                    # FP32 reduction error. This is reference-only overhead.
                    abs_products[rows, slots] = (
                        a.double().abs() @ down.double().abs().T
                    ).float()
                    del up, down
                return terms.sum(dim=1).to(torch.bfloat16), terms, abs_products
            for row in range(capacity):
                for slot in range(topk):
                    e = int(ids[row, slot])
                    if e < 0:
                        continue
                    z = staged_x[row, (j + e) % hidden] * (2.0 ** (e % 3 - 1))
                    if is_nt4:
                        a = (16 * torch.tanh(torch.relu(z) / 16)).square()
                    else:
                        a = torch.nn.functional.silu(z) * z
                    a = roundtrip((a * probs[row, slot]).reshape(1, -1))[0]
                    term = (
                        a[(h + e * 7) % intermediate]
                        + 0.5 * a[(h + e * 7 + intermediate // 2) % intermediate]
                    )
                    terms[row, slot] = term.to(torch.bfloat16).float()
                    abs_products[row, slot] = (
                        a[(h + e * 7) % intermediate].abs()
                        + 0.5 * a[(h + e * 7 + intermediate // 2) % intermediate].abs()
                    )
            return terms.sum(dim=1).to(torch.bfloat16), terms, abs_products

        fill(0)
        layer.forward(inputs)  # compile/allocate outside capture
        torch.cuda.synchronize()
        graph = None
        if args.graph:
            # All ranks capture the same collective sequence after eager warmup.
            import torch.distributed as dist

            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = layer.forward(inputs)

        def run():
            if graph is not None:
                graph.replay()
                return graph_output
            return layer.forward(inputs)

        for epoch in range(args.checks):

            def trace(stage):
                if args.trace:
                    print(
                        json.dumps(dict(rank=rank, epoch=epoch, stage=stage)),
                        flush=True,
                    )

            trace("fill")
            fill(epoch)
            trace("oracle")
            expected, reference_terms, reference_abs_products = oracle()
            trace("forward")
            actual = run().clone()
            trace("replay")
            replay = run().clone()
            trace("synchronize")
            torch.cuda.synchronize()
            trace("validate")
            torch.testing.assert_close(actual, replay, rtol=0, atol=0)
            # Tanh/exp implementation and scale-boundary differences require
            # numerical qualification, not bit equality against the torch oracle.
            rel_l2 = (
                (actual.float() - expected.float()).norm()
                / expected.float().norm().clamp_min(1e-8)
            ).item()
            max_abs = (actual.float() - expected.float()).abs().max().item()
            print(
                json.dumps(
                    dict(
                        rank=rank,
                        epoch=epoch,
                        stage="error_metrics",
                        weights=args.weights,
                        seed=args.seed,
                        input_scale=args.input_scale,
                        rel_l2=rel_l2,
                        max_abs=max_abs,
                    )
                ),
                flush=True,
            )
            close = torch.isclose(actual, expected, rtol=0.08, atol=0.002)
            # Snapshot every check, not just output mismatches: opposing expert
            # errors must not hide behind an apparently correct final sum.
            kernel_terms = layer._ensure_workspace().debug_combine_snapshot().float()
            recombined = kernel_terms.sum(dim=1).to(torch.bfloat16)
            if not bool(close.all()):
                # No extra collective/kernel invocation: peers may already be
                # waiting in the next epoch. Diagnose locally, then fail fast.
                # The runner deliberately uses the private layer workspace only
                # for debugging this explicit backend. Raw kernel layout stays
                # encapsulated in the shim's snapshot helper.
                precise, _, _ = oracle(fp64_gemm=True)
                bad = (~close).nonzero().cpu().tolist()
                details = []
                for row, channel in bad[:8]:
                    contributions = reference_terms[row, :, channel]
                    abs_sum = contributions.abs().sum().item()
                    signed_sum = contributions.sum().item()
                    details.append(
                        dict(
                            row=row,
                            channel=channel,
                            actual=actual[row, channel].item(),
                            expected=expected[row, channel].item(),
                            fp64_gemm_reference=precise[row, channel].item(),
                            expert_ids=ids[row].cpu().tolist(),
                            routing_weights=probs[row].cpu().tolist(),
                            reference_expert_terms=contributions.cpu().tolist(),
                            kernel_expert_terms=kernel_terms[row, :, channel]
                            .cpu()
                            .tolist(),
                            expert_term_deltas=(
                                kernel_terms[row, :, channel] - contributions
                            )
                            .cpu()
                            .tolist(),
                            kernel_terms_recombined=recombined[row, channel].item(),
                            reference_abs_sum=abs_sum,
                            reference_signed_sum=signed_sum,
                            cancellation_ratio=abs_sum / max(abs(signed_sum), 1e-30),
                        )
                    )
                print(
                    json.dumps(
                        dict(
                            rank=rank,
                            epoch=epoch,
                            stage="reference_mismatch",
                            mode=args.mode,
                            shape=[hidden, intermediate, experts, topk],
                            weights=args.weights,
                            seed=args.seed,
                            input_scale=args.input_scale,
                            graph=args.graph,
                            prequantized=args.prequantized,
                            rel_l2=rel_l2,
                            mismatched_elements=len(bad),
                            kernel_recombine_max_abs=(
                                actual.float() - recombined.float()
                            )
                            .abs()
                            .max()
                            .item(),
                            reference_fp32_fp64_max_abs=(
                                expected.float() - precise.float()
                            )
                            .abs()
                            .max()
                            .item(),
                            details=details,
                        )
                    ),
                    flush=True,
                )
            try:
                contribution_metrics = validate_contributions(
                    actual,
                    expected,
                    kernel_terms,
                    reference_terms,
                    ids,
                    reference_abs_products,
                    intermediate,
                )
            except AssertionError as error:
                # Contribution failures can occur even when the original final
                # output comparison passes. Diagnose those before exiting too.
                neighbor = torch.nextafter(
                    reference_terms.to(torch.bfloat16),
                    kernel_terms.to(torch.bfloat16),
                ).float()
                budget = (neighbor - reference_terms).abs()
                budget[reference_terms == 0] = 0
                ku = intermediate * 2**-24
                budget += (2 * ku / (1 - ku)) * reference_abs_products
                budget.masked_fill_((ids < 0).unsqueeze(-1), 0)
                bad = ((kernel_terms - reference_terms).abs() > budget).nonzero()
                details = []
                for row, slot, channel in bad[:4].cpu().tolist():
                    e = int(ids[row, slot])
                    detail = dict(
                        row=row,
                        slot=slot,
                        channel=channel,
                        expert=e,
                        actual=kernel_terms[row, slot, channel].item(),
                        reference=reference_terms[row, slot, channel].item(),
                        absolute_error=abs(
                            (kernel_terms - reference_terms)[row, slot, channel].item()
                        ),
                        bf16_step=budget[row, slot, channel].item(),
                    )
                    if args.weights == "dense" and e >= 0:
                        up, down = dense_expert(e)
                        up, down = roundtrip(up.float()), roundtrip(down.float())
                        # Match the batched GEMM shape used by oracle(), since
                        # a standalone matrix-vector call can round differently.
                        positions = (ids == e).nonzero()
                        rows, slots = positions[:, 0], positions[:, 1]
                        index = int(((rows == row) & (slots == slot)).nonzero()[0, 0])
                        staged = roundtrip(x.float())[rows]
                        z = staged @ up.T
                        z64 = (staged.double() @ up.double().T).float()

                        def activated(value):
                            if is_nt4:
                                return (
                                    16 * torch.tanh(torch.relu(value) / 16)
                                ).square()
                            return torch.nn.functional.silu(value) * value

                        a = roundtrip(activated(z) * probs[rows, slots, None])
                        a64 = roundtrip(activated(z64) * probs[rows, slots, None])
                        products = a[index].double() * down[channel].double()
                        dot = products.sum().item()
                        abs_sum = products.abs().sum().item()
                        detail.update(
                            fc2_reference_abs_product_sum=abs_sum,
                            fc2_reference_fp64_dot=dot,
                            fc2_reference_cancellation_ratio=abs_sum
                            / max(abs(dot), 1e-30),
                            error_over_abs_product_sum=detail["absolute_error"]
                            / max(abs_sum, 1e-30),
                            reference_fp32_fc2=(a @ down.T)[index, channel].item(),
                            reference_fp64_fc2_bf16=torch.tensor(
                                dot, dtype=torch.bfloat16
                            ).item(),
                            reference_fp64_both_bf16=(a64.double() @ down.double().T)[
                                index, channel
                            ]
                            .to(torch.bfloat16)
                            .item(),
                            reference_fc1_fp32_fp64_max_abs=(z[index] - z64[index])
                            .abs()
                            .max()
                            .item(),
                            reference_intermediate_changed_elements=int(
                                (a[index] != a64[index]).sum().item()
                            ),
                        )
                    details.append(detail)
                print(
                    json.dumps(
                        dict(
                            rank=rank,
                            epoch=epoch,
                            stage="contribution_mismatch",
                            error=str(error),
                            seed=args.seed,
                            input_scale=args.input_scale,
                            shape=[hidden, intermediate, experts, topk],
                            rel_l2=rel_l2,
                            failed_contributions=bad.shape[0],
                            details=details,
                        )
                    ),
                    flush=True,
                )
                raise
            print(
                json.dumps(
                    dict(
                        rank=rank,
                        epoch=epoch,
                        mode=args.mode,
                        graph=args.graph,
                        prequantized=args.prequantized,
                        weights=args.weights,
                        seed=args.seed,
                        input_scale=args.input_scale,
                        shape=[hidden, intermediate, experts, topk],
                        rel_l2=rel_l2,
                        contribution_metrics=contribution_metrics,
                        strict_output_mismatches=int((~close).sum().item()),
                        validation="bf16_fp32_contributions_v2",
                        replay_exact=True,
                        empty_rank_test=not args.skip_empty_rank,
                        status="PASS",
                    )
                ),
                flush=True,
            )
        # This is a correctness runner, deliberately not a performance benchmark.
    except BaseException:
        # A rank-local validation error must be visible BEFORE collective
        # destruction. Peers may already be entering the next kernel; waiting
        # for them in destroy/finalize can hide the original exception forever.
        traceback.print_exc()
        import sys

        sys.stderr.flush()
        os._exit(1)  # torchrun terminates peers; do not run collective cleanup.
    finally:
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)
        watchdog.cancel()


if __name__ == "__main__":
    main()
