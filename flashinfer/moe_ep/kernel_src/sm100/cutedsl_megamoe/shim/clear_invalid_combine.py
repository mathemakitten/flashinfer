# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Clear unwritten BF16 combine slots without touching peer-written slots."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


class ClearInvalidCombine:
    def __init__(self, num_experts: int):
        self.num_experts = num_experts

    @cute.kernel
    def kernel(self, ids: cute.Tensor, combine: cute.Tensor):
        lane, _, _ = cute.arch.thread_idx()
        slot, _, _ = cute.arch.block_idx()
        row = slot // ids.shape[1]
        k = slot % ids.shape[1]
        expert = ids[row, k]
        if (expert < 0) | (expert >= self.num_experts):
            for h in cutlass.range(lane, combine.shape[2], 128):
                combine[row, k, h] = cutlass.BFloat16(0)

    @cute.jit
    def __call__(self, ids: cute.Tensor, combine: cute.Tensor, stream: cuda.CUstream):
        self.kernel(ids, combine).launch(
            grid=(ids.shape[0] * ids.shape[1], 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )
