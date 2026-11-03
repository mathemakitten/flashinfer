"""Exercise the actual dataclass/knob merge without CUDA-only module imports."""

import ast
import dataclasses
import math
from pathlib import Path
import typing
import unittest

SHIM = (
    Path(__file__).resolve().parents[2]
    / "flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/shim"
)


def load_definition(filename, name, namespace):
    tree = ast.parse((SHIM / filename).read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


class WorkerConfigTest(unittest.TestCase):
    def test_pipeline_stage_reduction(self):
        ns = {}
        reduce = load_definition("nt4/pipeline_config.py", "reduce_ab_stages", ns)
        self.assertEqual(reduce(5, 6, 0), (5, 6))
        self.assertEqual(reduce(5, 6, 1), (4, 5))
        self.assertEqual(reduce(5, 6, 2), (3, 4))
        for a, b, n in ((3, 4, 2), (4, 4, -1), (4, 4, True)):
            with self.assertRaises(ValueError):
                reduce(a, b, n)

    def test_knob_reaches_config_and_invalid_caps_fail(self):
        ns = dict(
            vars(typing),
            dataclasses=dataclasses,
            math=math,
            _KIND_TO_TORCH_DTYPE={"mxfp8_e4m3": None},
        )
        config = load_definition("mxfp8.py", "MegaMoEMxfp8Config", ns)
        merge = load_definition("tuner.py", "with_knobs", ns)
        original = config(
            rank=0,
            world_size=4,
            num_tokens_per_rank=2048,
            num_topk=10,
            num_total_experts=512,
            hidden=768,
            intermediate=4096,
            activation="clamped_relu2",
            clamped_relu2_limit=16.0,
            mma_tiler_mnk=(128, 256, 128),
            cluster_shape_mnk=(1, 1, 1),
            use_2cta_instrs=False,
            group_hint=1024,
            flag_batch=1,
        )
        for cap in (152, 144, 136):
            candidate = merge(original, {"max_active_clusters": cap})
            self.assertEqual(candidate.max_active_clusters, cap)
            self.assertEqual(candidate.group_hint, 1024)
            self.assertEqual(candidate.cluster_shape_mnk, (1, 1, 1))
        self.assertEqual(
            merge(original, {"ab_stage_reduction": 2}).ab_stage_reduction, 2
        )
        for invalid in (-1, 3, True):
            with self.assertRaises(ValueError):
                merge(original, {"ab_stage_reduction": invalid})
        self.assertIsNone(original.max_active_clusters)
        for cap in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                merge(original, {"max_active_clusters": cap})


if __name__ == "__main__":
    unittest.main()
