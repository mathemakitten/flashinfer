"""CPU-only contract checks; also runnable directly without FlashInfer imports."""

import importlib.util
from pathlib import Path
import sys
import unittest


def load_config():
    path = Path(__file__).resolve().parents[2] / (
        "flashinfer/moe_ep/backends/mega/kernel/sm100/"
        "mxfp8_mxfp8_bf16_cutedsl/config.py"
    )
    spec = importlib.util.spec_from_file_location("nt4_config_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig


Config = load_config()


class TestNt4Config(unittest.TestCase):
    def make(self, **kwargs):
        args = dict(
            intermediate_size=4096,
            top_k=10,
            activation="clamped_relu2",
            clamped_relu2_limit=16.0,
        )
        args.update(kwargs)
        return Config(**args)

    def test_nt4_shape_and_explicit_selection(self):
        self.assertEqual(self.make().activation, "clamped_relu2")
        self.assertEqual(Config(4096, 10).activation, "swiglu")

    def test_soft_clamp_is_not_gate_clamp(self):
        for field in ("gate_up_clamp", "activation_clamp"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.make(**{field: 16.0})

    def test_routing_placement_explicit_and_restricted(self):
        self.assertEqual(self.make().routing_weight_placement, "before_quantization")
        self.assertEqual(
            self.make(routing_weight_placement="after_fc2").routing_weight_placement,
            "after_fc2",
        )
        with self.assertRaises(ValueError):
            self.make(routing_weight_placement="invalid")
        with self.assertRaises(ValueError):
            Config(4096, 10, routing_weight_placement="after_fc2")

    def test_limit_must_be_finite_positive(self):
        for value in (None, 0, -16, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make(clamped_relu2_limit=value)

    def test_no_silent_swiglu_cache_or_atomic_reduction(self):
        for kwargs in (
            {"knobs": "auto"},
            {"knobs": "cache"},
            {"enable_in_kernel_fc2_reduce": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)

    def test_partial_fc1_tile_rejected(self):
        with self.assertRaises(ValueError):
            self.make(intermediate_size=320)

    def test_swiglu_rejects_nt4_only_limit(self):
        with self.assertRaises(ValueError):
            self.make(activation="swiglu")


if __name__ == "__main__":
    unittest.main()
