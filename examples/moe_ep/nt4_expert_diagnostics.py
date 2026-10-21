"""Untimed workspace snapshots, reference diagnostics, and stagewise checks."""

import torch


def decode_e8m0_exact(scale_bytes):
    """Decode finite E8M0 bytes as exact binary64 powers of two.

    E8M0 exponent b-127 becomes binary64 exponent b-127+1023. All finite
    E8M0 scales are normal binary64, including byte 0 (2^-127). Construct
    exponent bits instead of CUDA pow(), whose rounding can break exact
    scale equality / FP8 representability checks. Byte 255 denotes NaN.
    """
    if scale_bytes.dtype != torch.uint8:
        raise ValueError("E8M0 scales must be raw uint8 bytes")
    if bool((scale_bytes == 255).any()):
        raise AssertionError("nonfinite E8M0 scale")
    return ((scale_bytes.to(torch.int64) + 896) << 52).contiguous().view(torch.float64)


def fc1_activation_interval(x, up, z64):
    """Operand-derived envelope for NT4's unweighted (16*tanh(relu(z)/16))^2.

    Provisional model: FP32 dot accumulation with <=K rounding steps, using
    eps=2^-23 (covers directed as well as nearest rounding); exact MXFP8
    products in the normal FP32 range. gamma_K * sum(abs(x*w)) bounds FC1.
    FP64 reference error uses gamma_2K and inflates its positive product sum.
    No measured kernel discrepancy enters this bound.

    tanhf is documented within 2 ULP of correctly rounded; 3*eps relative
    covers that plus rounding to FP32 (with an absolute underflow allowance).
    This assumes the non-approximate tanh contract of the owned epilogue;
    it is not a proof of arbitrary compiler/Tensor Core lowering semantics.
    https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html
    """
    if x.ndim != 2 or up.ndim != 2 or x.shape[1] != up.shape[1]:
        raise ValueError("invalid FC1 operands")
    if z64.shape != (x.shape[0], up.shape[0]):
        raise ValueError("invalid FC1 reference shape")
    k = x.shape[1]
    eps = 2.0**-23
    if not 0 < k * eps < 0.5:
        raise ValueError("unsupported FC1 reduction length")
    gamma = k * eps / (1 - k * eps)
    gamma64 = (2 * k * 2.0**-53) / (1 - 2 * k * 2.0**-53)
    absolute_products = x.double().abs() @ up.double().abs().T
    absolute_products = absolute_products / (1 - gamma64)
    # Conservative allowance for flushed FP32 subnormal intermediates.
    tiny = 2.0**-126
    radius = (gamma + gamma64) * absolute_products + k * tiny
    lo_z, hi_z = z64.double() - radius, z64.double() + radius
    # Dividing/multiplying by 16 is exact for normal FP32 values. Monotonic
    # tanh/relu propagate the FC1 endpoints without a loose global derivative.
    lo_t = torch.tanh(lo_z.clamp_min(0) / 16)
    hi_t = torch.tanh(hi_z.clamp_min(0) / 16)
    lo_c = 16 * (lo_t * (1 - 3 * eps) - tiny).clamp_min(0)
    hi_c = 16 * (hi_t * (1 + 3 * eps) + tiny)
    lower = (lo_c.square() * (1 - eps) - tiny).clamp_min(0)
    upper = hi_c.square() * (1 + eps) + tiny
    if not all(bool(torch.isfinite(v).all()) for v in (lower, upper, z64)):
        raise AssertionError("nonfinite FC1 error interval")
    return lower, upper


def boundary_blocks(
    actual,
    reference,
    lower,
    upper,
    actual_scales,
    quantize,
    *,
    diagnostics=None,
    diagnostic_mask=None,
):
    """Accept only adjacent codes justified by an independent activation interval.

    quantize returns (FP8 values, decoded per-32 scales). Scale must be stable
    across both interval endpoints and equal the stored scale. Ambiguous scale
    boundaries fail closed. This function never expands the supplied interval.
    """
    if any(v.shape != actual.shape for v in (reference, lower, upper)):
        raise ValueError("boundary shape mismatch")
    if actual_scales.shape != (*actual.shape[:-1], actual.shape[-1] // 32):
        raise ValueError("boundary scale shape mismatch")
    if not all(
        bool(torch.isfinite(v).all())
        for v in (actual, reference, lower, upper, actual_scales)
    ):
        raise AssertionError("nonfinite boundary operands")
    if bool(((lower < 0) | (lower > upper)).any()) or bool((actual_scales <= 0).any()):
        raise AssertionError("invalid boundary interval or scale")
    # Round interval endpoints outward when handing FP32 to the quantizer.
    lo = lower.float()
    hi = upper.float()
    lo = torch.where(
        lo.double() > lower, torch.nextafter(lo, torch.full_like(lo, -float("inf"))), lo
    )
    hi = torch.where(
        hi.double() < upper, torch.nextafter(hi, torch.full_like(hi, float("inf"))), hi
    )
    _, sl = quantize(lo.contiguous())
    _, sh = quantize(hi.contiguous())
    stable = (sl == sh) & (sl == actual_scales)
    scale = actual_scales.double().repeat_interleave(32, -1)
    qa = (actual.double() / scale).float().to(torch.float8_e4m3fn)
    qr = (reference.double() / scale).float().to(torch.float8_e4m3fn)
    representable = (qa.double() * scale == actual) & (qr.double() * scale == reference)
    codes_a = qa.view(torch.uint8).to(torch.int16)
    codes_r = qr.view(torch.uint8).to(torch.int16)
    adjacent = (codes_a - codes_r).abs() == 1
    midpoint = (actual.double() + reference.double()) / 2
    reaches = (lower <= midpoint) & (midpoint <= upper)
    allowed = (
        representable
        & (actual >= 0)
        & (reference >= 0)
        & ((actual == reference) | (adjacent & reaches))
    )
    valid = allowed.reshape(-1, 32).all(-1) & stable.reshape(-1)
    if diagnostics is not None:
        selected = ~valid
        if diagnostic_mask is not None:
            if diagnostic_mask.shape != valid.shape:
                raise ValueError("diagnostic mask shape mismatch")
            selected &= diagnostic_mask
        blocks_per_row = actual.shape[-1] // 32
        for block in selected.nonzero().flatten()[:8].tolist():
            row, column_block = divmod(block, blocks_per_row)
            start = column_block * 32
            stop = start + 32
            details = []
            for offset in range(start, stop):
                if (
                    actual[row, offset] == reference[row, offset]
                    and allowed[row, offset]
                ):
                    continue
                details.append(
                    dict(
                        channel=offset,
                        actual=actual[row, offset].item(),
                        reference=reference[row, offset].item(),
                        actual_code=int(codes_a[row, offset]),
                        reference_code=int(codes_r[row, offset]),
                        representable=bool(representable[row, offset]),
                        adjacent=bool(adjacent[row, offset]),
                        midpoint=midpoint[row, offset].item(),
                        activation_lower=lower[row, offset].item(),
                        activation_upper=upper[row, offset].item(),
                        interval_reaches_midpoint=bool(reaches[row, offset]),
                        element_allowed=bool(allowed[row, offset]),
                    )
                )
            reasons = []
            if not bool((sl == sh)[row, column_block]):
                reasons.append("ambiguous_scale_interval")
            if not bool((sl == actual_scales)[row, column_block]):
                reasons.append("stored_scale_differs_from_lower_endpoint")
            if not bool((sh == actual_scales)[row, column_block]):
                reasons.append("stored_scale_differs_from_upper_endpoint")
            if not bool(allowed.reshape(-1, 32)[block].all()):
                reasons.append("element_outside_boundary_rule")
            diagnostics.append(
                dict(
                    expert_row=row,
                    channel_start=start,
                    reasons=reasons,
                    stored_scale=actual_scales[row, column_block].item(),
                    lower_endpoint_scale=sl[row, column_block].item(),
                    upper_endpoint_scale=sh[row, column_block].item(),
                    block_amax_lower=lower[row, start:stop].max().item(),
                    block_amax_upper=upper[row, start:stop].max().item(),
                    details=details,
                )
            )
    return valid


def validate_intermediate(actual, variants, *, boundary=None):
    """Exact block variants with an optional independently bounded fallback.

    Every 32-element block must equal one independent reference block in full.
    A valid block from each variant may be selected, but arbitrary mixtures of
    elements within a block need the boundary callback to justify the entire
    block. Never derive variants or error intervals from actual.
    """
    if not variants or actual.shape[-1] % 32:
        raise ValueError("expected reference variants and complete MXFP8 blocks")
    if any(v.shape != actual.shape for v in variants):
        raise ValueError("intermediate variant shape mismatch")
    if not all(bool(torch.isfinite(v).all()) for v in [actual, *variants]):
        raise AssertionError("nonfinite intermediate")
    blocks = actual.reshape(-1, 32)
    valid = torch.zeros(blocks.shape[0], dtype=torch.bool, device=actual.device)
    for v in variants:
        valid |= (blocks == v.reshape(-1, 32)).all(-1)
    boundary_count = 0
    if boundary is not None and not bool(valid.all()):
        # Lazy: the additional FP64 abs-product GEMM runs only for experts
        # whose exact independent block variants were insufficient.
        fallback = boundary()
        if fallback.shape != valid.shape or fallback.dtype != torch.bool:
            raise ValueError("invalid boundary block mask")
        boundary_count = int((~valid & fallback).sum())
        valid |= fallback
    if not bool(valid.all()):
        block = int((~valid).nonzero()[0])
        refs = [v.reshape(-1, 32)[block] for v in variants]
        changed = (blocks[block] != refs[0]).nonzero().flatten()[:4].tolist()
        detail = [
            dict(
                offset=j,
                actual=blocks[block, j].item(),
                references=[v[j].item() for v in refs],
            )
            for j in changed
        ]
        raise AssertionError(
            f"FC1 quantization block {block} matches no independent reference variant or justified boundary interval; "
            f"row={block // (actual.shape[-1] // 32)}, "
            f"channel_start={(block % (actual.shape[-1] // 32)) * 32}, details={detail}"
        )
    rel = (actual - variants[0]).norm(dim=-1) / variants[0].norm(dim=-1).clamp_min(1e-8)
    if bool((rel > 0.015).any()):
        raise AssertionError("FC1 intermediate relative L2 exceeds 0.015")
    return dict(
        boundary_blocks=boundary_count,
        changed_elements=int((actual != variants[0]).sum()),
        changed_blocks=int((blocks != variants[0].reshape(-1, 32)).any(-1).sum()),
        max_row_rel_l2=rel.max().item() if rel.numel() else 0.0,
    )


def snapshot_fc1_expert(workspace, counts, local_expert):
    """GPU tensors for all valid rows of one expert's retained output pool."""
    mega = workspace._frontend._mega
    kernel = mega.kernel

    def region(name):
        spec = kernel._local_region_by_name[name]
        return mega.local_workspace.narrow(0, kernel._local_offsets[name], spec.nbytes)

    def prefix(block):
        return sum((n + block - 1) // block * block for n in counts[:local_expert])

    start = prefix(kernel.token_padding_block)
    sf_start = prefix(kernel.sf_padding_block)
    count = counts[local_expert]
    shape = kernel._local_region_by_name["fc1_output"].shape
    width = shape[1]
    metadata = (
        region("token_src_metadata").view(torch.int64)[start : start + count].clone()
    )
    q = (
        region("fc1_output")
        .view(torch.uint8)
        .reshape(shape)[start : start + count]
        .clone()
    )
    rows = torch.arange(sf_start, sf_start + count, device=q.device)[:, None]
    k = torch.arange(width // 32, device=q.device)[None, :]
    offsets = (
        ((rows // 128) * (width // 128) + k // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + k % 4
    )
    scales = region("fc1_output_sf").view(torch.uint8)[offsets].contiguous()
    return metadata, q, scales


def snapshot_prequant_expert(workspace, counts, local_expert):
    mega = workspace._frontend._mega
    kernel = mega.kernel
    spec = kernel._local_region_by_name["debug_prequant"]
    start = sum(
        (n + kernel.token_padding_block - 1)
        // kernel.token_padding_block
        * kernel.token_padding_block
        for n in counts[:local_expert]
    )
    return (
        mega.local_workspace.narrow(
            0, kernel._local_offsets["debug_prequant"], spec.nbytes
        )
        .view(torch.float32)
        .reshape(spec.shape)[start : start + counts[local_expert]]
        .clone()
    )


def prequant_record(
    z, activated, z64, reference_activation, actual_q, reference_q, scale
):
    """Scalar report. Midpoint is meaningful only for adjacent same-scale codes."""
    values = torch.tensor(
        [z, activated, z64, reference_activation, actual_q, reference_q, scale],
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(values).all()) or scale <= 0:
        raise AssertionError("missing or nonfinite prequant instrumentation")
    codes = (
        torch.tensor([actual_q / scale, reference_q / scale])
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .tolist()
    )
    adjacent = actual_q >= 0 and reference_q >= 0 and abs(codes[0] - codes[1]) == 1
    midpoint = (actual_q + reference_q) / 2
    requantized = (
        torch.tensor(activated / scale).to(torch.float8_e4m3fn).float() * scale
    ).item()
    activation_at_kernel_z = (
        (16 * torch.tanh(torch.tensor(max(z, 0.0), dtype=torch.float64) / 16))
        .square()
        .item()
    )
    return dict(
        kernel_fc1=z,
        reference_fc1_fp64=z64,
        kernel_activation=activated,
        reference_activation_fp64=reference_activation,
        fp64_activation_at_kernel_fc1=activation_at_kernel_z,
        kernel_activation_error_at_same_fc1=activated - activation_at_kernel_z,
        kernel_quantized=actual_q,
        reference_quantized=reference_q,
        scale=scale,
        requantized_kernel_activation=requantized,
        quantizer_reproduced=requantized == actual_q,
        adjacent_codes_at_kernel_scale=adjacent,
        midpoint=midpoint if adjacent else None,
        kernel_distance_from_midpoint=activated - midpoint if adjacent else None,
        reference_distance_from_midpoint=(
            reference_activation - midpoint if adjacent else None
        ),
    )


def sf_byte_offsets(row, width):
    """Linear byte offsets for the kernel's 128-row, four-scale SF atom."""
    if width % 128:
        raise ValueError("snapshot requires a multiple of 128 intermediate columns")
    return [
        ((row // 128) * (width // 128) + k // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + k % 4
        for k in range(width // 32)
    ]


def snapshot_fc1(workspace, counts, local_expert, source_rank, row, slot):
    """Read a completed kernel's retained pool; never launch another MoE call.

    counts must be the all-source token counts for this owner's local experts.
    Restrict metadata lookup to valid rows of the selected expert (padding can
    contain stale metadata). Data and scale pools have independent padding.
    """
    mega = workspace._frontend._mega
    kernel = mega.kernel

    def region(name):
        spec = kernel._local_region_by_name[name]
        return mega.local_workspace.narrow(0, kernel._local_offsets[name], spec.nbytes)

    def prefix(block):
        return sum((n + block - 1) // block * block for n in counts[:local_expert])

    data_start = prefix(kernel.token_padding_block)
    sf_start = prefix(kernel.sf_padding_block)
    count = counts[local_expert]
    metadata = region("token_src_metadata").view(torch.int64)
    target = (source_rank << 48) | (slot << 32) | row
    matches = (metadata[data_start : data_start + count] == target).nonzero().flatten()
    if matches.numel() != 1:
        raise RuntimeError(
            f"snapshot expected one valid source record, found {matches.numel()}"
        )
    relative_row = int(matches[0])
    shape = kernel._local_region_by_name["fc1_output"].shape
    width = shape[1]
    q = (
        region("fc1_output")
        .view(torch.uint8)
        .reshape(shape)[data_start + relative_row]
        .cpu()
    )
    offsets = torch.tensor(
        sf_byte_offsets(sf_start + relative_row, width),
        device=mega.local_workspace.device,
    )
    scales = region("fc1_output_sf").view(torch.uint8)[offsets].cpu()
    a = q.view(torch.float8_e4m3fn).float() * torch.pow(
        2.0, scales.float() - 127
    ).repeat_interleave(32)
    if not torch.isfinite(a).all():
        raise RuntimeError("nonfinite kernel intermediate snapshot")
    return dict(
        source_rank=source_rank,
        pool_row=data_start + relative_row,
        sf_row=sf_start + relative_row,
        intermediate=a.tolist(),
        fp8_bytes=q.tolist(),
        scale_bytes=scales.tolist(),
    )


def diagnose_expert(x, up, down, z, a, roundtrip, index, channel, weights=None):
    """Compare reference arithmetic variants with identical expert batch shape.

    These are reference intermediates, not snapshots of the fused kernel.
    FP64 FC2 with the original quantized intermediate isolates FC2 reference
    roundoff. Recomputing FC1 in FP64 tests sensitivity before quantization.
    """
    z64 = x.double() @ up.double().T

    def activate(value):
        result = (16 * torch.tanh(torch.relu(value) / 16)).square()
        return result if weights is None else result * weights[:, None]

    a_fc1_64 = roundtrip(activate(z64.float()))
    a_both_64 = roundtrip(activate(z64).float())
    products = a[index].double() * down[channel].double()
    dot = products.sum().item()
    abs_sum = products.abs().sum().item()
    variants = {}
    for name, value in (("fp64_fc1", a_fc1_64), ("fp64_fc1_activation", a_both_64)):
        changed = (value[index] != a[index]).nonzero().flatten()
        delta = (value[index].double() - a[index].double()) * down[channel].double()
        variants[name] = dict(
            changed_intermediate_elements=changed.numel(),
            fc2_fp64_dot=(value[index].double() * down[channel].double()).sum().item(),
            fc2_bf16=(value[index].double() * down[channel].double())
            .sum()
            .to(torch.bfloat16)
            .item(),
            fc2_dot_delta=delta.sum().item(),
            changed_elements=[
                dict(
                    channel=j,
                    fc1_fp32=z[index, j].item(),
                    fc1_fp64=z64[index, j].item(),
                    intermediate_reference=a[index, j].item(),
                    intermediate_variant=value[index, j].item(),
                    fc2_product_delta=delta[j].item(),
                )
                for j in changed[:16].tolist()
            ],
        )
    return dict(
        scope="reference_variants_not_kernel_intermediate_snapshots",
        reference_fp32_fc2=(a @ down.T)[index, channel].item(),
        reference_fp64_fc2=dot,
        reference_fp64_fc2_bf16=products.sum().to(torch.bfloat16).item(),
        fc2_abs_product_sum=abs_sum,
        fc2_cancellation_ratio=abs_sum / max(abs(dot), 1e-30),
        fc1_fp32_fp64_max_abs=(z[index].double() - z64[index]).abs().max().item(),
        variants=variants,
    )
