"""CuTeDSL BF16 MegaMoE kernel configuration."""

from __future__ import annotations

from dataclasses import dataclass

from ..common.bf16_config import Sm100_Bf16_Cutedsl_MegaMoeConfigBase


@dataclass
class Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(Sm100_Bf16_Cutedsl_MegaMoeConfigBase):
    """Parameters for the SM100 BF16 CuTeDSL MegaMoE kernel."""

    kernel_name: str = "sm100_bf16_bf16_bf16_cutedsl"

    activation: str = "swiglu"
    clamped_relu2_limit: float | None = None

    def __post_init__(self):
        if self.activation not in ("swiglu", "clamped_relu2"):
            raise ValueError("unsupported BF16 activation")
        if self.activation == "clamped_relu2":
            if self.clamped_relu2_limit != 16.0 or self.intermediate_size % 256:
                raise ValueError(
                    "NT4 BF16 requires limit=16 and intermediate divisible by 256"
                )
            if self.gate_up_clamp is not None or self.activation_clamp is not None:
                raise ValueError("use clamped_relu2_limit for NT4")
            if self.enable_in_kernel_fc2_reduce or self.knobs == "auto":
                raise ValueError(
                    "NT4 BF16 requires separate reduction and explicit knobs"
                )
            if isinstance(self.knobs, dict) and (
                self.knobs.get("apply_topk_in_fc1", False)
                or self.knobs.get("in_kernel_fc2_reduce", False)
            ):
                raise ValueError("NT4 BF16 requires late weighting")
        elif self.clamped_relu2_limit is not None:
            raise ValueError("clamped_relu2_limit requires clamped_relu2")
