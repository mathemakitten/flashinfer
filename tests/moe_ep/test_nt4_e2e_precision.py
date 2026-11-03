"""CPU checks for experiment precision selection; no GPU execution."""

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2] / "examples/moe_ep"
sys.path.insert(0, str(ROOT))
from nt4_e2e import precision_options, trial_modes


class PrecisionTest(unittest.TestCase):
    def test_bf16_removes_all_fp8_activation_flags(self):
        args = [
            "driver.py",
            "--bf16",
            "--fp8-recipe=mxfp8",
            "--fp8-format",
            "e4m3",
            "--fp8-param-gather",
            "--load",
            "/checkpoint",
            "--num-experts",
            "512",
        ]
        result = precision_options(args, "bf16")
        self.assertEqual(
            result,
            ["driver.py", "--bf16", "--load", "/checkpoint", "--num-experts", "512"],
        )

    def test_both_precisions_launch_direct(self):
        self.assertIn(("direct", "direct"), trial_modes("bf16"))
        self.assertIn(("direct", "direct"), trial_modes("mxfp8"))
        self.assertEqual(trial_modes("bf16")[-1], ("baseline-end", "baseline"))

    def test_mxfp8_sets_recipe_and_preserves_storage_gather(self):
        result = precision_options(
            ["driver.py", "--bf16", "--fp8-param-gather", "--fp8-recipe", "delayed"],
            "mxfp8",
        )
        self.assertEqual(result.count("--fp8-recipe"), 1)
        self.assertEqual(result[result.index("--fp8-recipe") + 1], "mxfp8")
        self.assertIn("--fp8-param-gather", result)


if __name__ == "__main__":
    unittest.main()
