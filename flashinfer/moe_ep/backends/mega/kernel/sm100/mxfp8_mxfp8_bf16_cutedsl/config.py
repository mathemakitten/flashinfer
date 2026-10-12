"""CuTeDSL MXFP8 mega-MoE kernel config."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.sm100.cutedsl_megamoe.mxfp8_mega_moe``.

    ``intermediate_size`` is the post-SwiGLU width, matching NVFP4 and SGLang.
    The MXFP8 kernel's full FC1 gate+up width is derived internally as
    ``2 * intermediate_size``.

    Expert weights must be MXFP8 at kernel launch; supply bf16 ``MoEWeightPack``
    and enable ``MegaConfig.preprocess_weights`` (default), or pass pre-quantized
    fp8 weights with ``w13_scale`` / ``w2_scale``.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm100_mxfp8_mxfp8_bf16_cutedsl"
    kind: Literal["mxfp8_e4m3", "mxfp8_e5m2"] = "mxfp8_e4m3"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    # Enables in_kernel_fc2_reduce, autotune may still disable this if it is faster
    enable_in_kernel_fc2_reduce: bool = False
    # Kernel tuning knobs (see kernel_src.sm100.cutedsl_megamoe.shim.tuner); overrides
    # the token-count default heuristic entirely when set, e.g. a winner from the
    # kernel repo's tester sweep. None -> tuner.default_knobs(..., dtype="mxfp8").
    # "auto" -> online autotune at the first forward: collectively time the
    # shim.autotune candidate set on the live problem and keep the winner
    # (one cute.compile per candidate, paid once per session).
    knobs: dict | str | None = None

    # Explicit development variant; never selected by the default SwiGLU path.
    activation: Literal["swiglu", "clamped_relu2"] = "swiglu"
    clamped_relu2_limit: float | None = None

    routing_weight_placement: Literal["before_quantization", "after_fc2"] = (
        "before_quantization"
    )

    def __post_init__(self) -> None:
        if self.routing_weight_placement not in ("before_quantization", "after_fc2"):
            raise ValueError("invalid routing_weight_placement")
        if (
            self.routing_weight_placement == "after_fc2"
            and self.activation != "clamped_relu2"
        ):
            raise ValueError("after_fc2 is qualified only for clamped_relu2")
        if self.activation not in ("swiglu", "clamped_relu2"):
            raise ValueError("activation must be swiglu or clamped_relu2")
        if self.activation == "clamped_relu2":
            limit = self.clamped_relu2_limit
            if limit is None or not math.isfinite(limit) or limit <= 0:
                raise ValueError(
                    "clamped_relu2 requires a positive finite clamped_relu2_limit"
                )
            if self.gate_up_clamp is not None or self.activation_clamp is not None:
                raise ValueError("clamped_relu2_limit is separate from gate_up_clamp")
            if self.intermediate_size % 256:
                raise ValueError(
                    "clamped_relu2 currently requires intermediate_size divisible by 256"
                )
            if self.enable_in_kernel_fc2_reduce:
                raise ValueError(
                    "clamped_relu2 initially requires separate deterministic top-k reduction"
                )
            if isinstance(self.knobs, str):
                raise ValueError(
                    "clamped_relu2 has no qualified tuning cache; use None or explicit knobs"
                )
        elif self.clamped_relu2_limit is not None:
            raise ValueError("clamped_relu2_limit requires activation='clamped_relu2'")
