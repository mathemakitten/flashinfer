# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owned NT4 non-gated adaptation; vendor source remains unchanged.
"""Autonomous epilogue for the fused fc1+fc2 GLU BF16 MegaMoE kernel."""

from typing import Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from common.megamoe_constants import Log2E
from cutlass.cute.typing import Float32
from .epilogue_mxfp8 import GluMxfp8Epilogue

# Gate/up interleave granularity on the fc1 feature axis: each subtile folds
# one 32-col gate block with its adjacent 32-col up block into 32 downproj
# outputs.  (Exported for the host runner / reference.)
Fc1GateUpInterleave = 32
Fc1CTMAStages = 1


# =============================================================================
# GluBf16Epilogue
# =============================================================================


class GluBf16Epilogue(GluMxfp8Epilogue):
    """NT4 BF16 non-gated epilogue, using the owned NT4 activation/layout.

    The whole task-tile machinery is inherited from ``GluMxfp8Epilogue``: the
    fc1/fc2 chains, the raw-C store path and the TMEM/SMEM layouts are all
    parametric over ``fc1_output_dtype`` / ``combine_format`` (the fc1
    hand-off quantisation is const_expr-gated on ``fc1_output_dtype.width ==
    8`` and collapses to a plain f32 -> bf16 cast here).  This class only
    narrows construction (no scale factors, no overlapping accumulation) and
    inherits the NT4 clamped squared-ReLU hook without a gate/up fold.
    """

    def __init__(
        self,
        *,
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        use_2cta_instrs: bool,
        fc1_output_dtype: Type[cutlass.Numeric],
        fc1_output_layout: utils.LayoutEnum,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        c_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        glu_clamp: Optional[float] = None,
        epilog_sync_bar_id: int = 1,
        epilogue_warp_ids: Tuple[int, ...] = (0, 1, 2, 3),
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        fc2_in_kernel_topk_reduce: bool = False,
        token_back_by_dispatch: bool = False,
        epi_flag_batch: Tuple[int, int] = (1, 1),
        apply_topk_in_fc1: bool = False,
        generate_c: bool = False,
        use_stg_fc1: bool = False,
    ) -> None:
        super().__init__(
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mn=cluster_shape_mn,
            use_2cta_instrs=use_2cta_instrs,
            # Placeholder: the parent ctor requires an SF vector size.  Every
            # SF attribute it derives is cleared right below, and the SF
            # consumers in the fc1 chain are const_expr-gated on
            # ``fc1_output_dtype.width == 8``.
            sf_vec_size=32,
            fc1_output_dtype=fc1_output_dtype,
            fc1_output_layout=fc1_output_layout,
            acc_dtype=acc_dtype,
            c_dtype=c_dtype,
            glu_clamp=glu_clamp,
            allow_overlap_acc=False,
            epilog_sync_bar_id=epilog_sync_bar_id,
            epilogue_warp_ids=epilogue_warp_ids,
            static_expert_shape=static_expert_shape,
            fc2_in_kernel_topk_reduce=fc2_in_kernel_topk_reduce,
            token_back_by_dispatch=token_back_by_dispatch,
            epi_flag_batch=epi_flag_batch,
            apply_topk_in_fc1=apply_topk_in_fc1,
            generate_c=generate_c,
            use_stg_fc1=use_stg_fc1,
        )

        # BF16 stores the fc1 output as plain data: clear the scale-factor
        # state the parent ctor derived from the placeholder.
        self.sf_dtype = None
        self._sf_vec_size = None
        self._num_sfa_tmem_cols = 0
        self._num_sfb_tmem_cols = 0
        self._num_sf_tmem_cols = 0

        # No overlapping accumulation: with zero SF TMEM columns the two acc
        # stages tile TMEM back-to-back (2 x 256 = the full 512-col budget).
        # The parent ctor currently hardcodes ``_overlapping_accum = True``
        # (its computed ``allow_overlap_acc`` value is shadowed), so passing
        # ``allow_overlap_acc=False`` above is not sufficient on its own.
        self._overlapping_accum = False
        self._num_acc_pipeline_stages = self._num_acc_stage
        self._num_accumulator_tmem_cols = self._cta_tile_n * self._num_acc_stage
        self._iter_acc_early_release = 0
