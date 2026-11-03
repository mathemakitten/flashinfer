# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Late FP32 routing-weight multiplication and reduction of BF16 expert terms."""

import triton
import triton.language as tl


@triton.jit
def _reduce(
    terms, ids, probs, output, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for slot in range(K):
        expert = tl.load(ids + row * K + slot)
        prob = tl.load(probs + row * K + slot)
        value = tl.load(
            terms + (row * K + slot) * H + col, mask=(col < H) & (expert >= 0), other=0
        ).to(tl.float32)
        total += value * tl.where(expert >= 0, prob, 0.0)
    tl.store(output + row * H + col, total, mask=col < H)


def weighted_reduce(terms, ids, probs, output):
    n, k, h = terms.shape
    if n:
        _reduce[(n, triton.cdiv(h, 256))](
            terms, ids, probs, output, h, k, 256, enable_fp_fusion=False
        )
