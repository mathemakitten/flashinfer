"""Execute the host config validation without importing CUDA dependencies."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/shim/mxfp8.py"
tree = ast.parse(SOURCE.read_text())
validate = next(
    n
    for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
)
namespace = {"math": math, "_KIND_TO_TORCH_DTYPE": {"mxfp8_e4m3": None}}
exec(
    compile(ast.Module(body=[validate], type_ignores=[]), str(SOURCE), "exec"),
    namespace,
)


class GeometryTest(unittest.TestCase):
    def config(self, **overrides):
        fields = dict(
            ab_stage_reduction=0,
            max_active_clusters=None,
            return_contributions_only=False,
            local_only=False,
            routing_weight_placement="after_fc2",
            activation="clamped_relu2",
            clamped_relu2_limit=16,
            gate_up_clamp=None,
            intermediate=4096,
            enable_in_kernel_fc2_reduce=False,
            in_kernel_fc2_reduce=False,
            token_back_by_dispatch=False,
            kind="mxfp8_e4m3",
            world_size=4,
            rank=0,
            num_tokens_per_rank=128,
            num_topk=10,
            num_total_experts=512,
            hidden=768,
            mma_tiler_mnk=(256, 256, 128),
            cluster_shape_mnk=(2, 1, 1),
            use_2cta_instrs=True,
            load_balance_mode="static",
            group_hint=1024,
            flag_batch=1,
            epi_flag_batch=(1, 1),
        )
        return SimpleNamespace(**(fields | overrides))

    def test_default_and_m128(self):
        namespace["__post_init__"](self.config())
        namespace["__post_init__"](
            self.config(
                mma_tiler_mnk=(128, 256, 128),
                cluster_shape_mnk=(1, 1, 1),
                use_2cta_instrs=False,
            )
        )

    def test_m128_requires_nt4_and_matching_geometry(self):
        base = dict(
            mma_tiler_mnk=(128, 256, 128),
            cluster_shape_mnk=(1, 1, 1),
            use_2cta_instrs=False,
        )
        for wrong in (
            dict(cluster_shape_mnk=(2, 1, 1)),
            dict(use_2cta_instrs=True),
            dict(mma_tiler_mnk=(128, 128, 128)),
            dict(
                activation="swiglu",
                clamped_relu2_limit=None,
                routing_weight_placement="before_quantization",
            ),
        ):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                namespace["__post_init__"](self.config(**(base | wrong)))


if __name__ == "__main__":
    unittest.main()
