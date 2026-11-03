"""CPU contracts for the real-route CUTLASS sweep."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/moe_ep"))
from nt4_cutlass_decode_tune import (
    pack_scale,
    error_metrics,
    select_pair,
    exact_comparison,
)
from nt4_checkpoint_artifact import canonical_scale


class DecodeTuneTest(unittest.TestCase):
    def test_scale_layout_roundtrip_nt4_shapes(self):
        for rows, cols in ((4096, 24), (768, 128)):
            scales = torch.randint(0, 255, (2, rows, cols), dtype=torch.uint8)
            packed = pack_scale(scales)
            self.assertEqual(packed.shape, (2, rows, cols // 4))
            self.assertEqual(packed.dtype, torch.int32)
            for expert in range(2):
                self.assertTrue(
                    torch.equal(
                        canonical_scale(packed[expert], rows, cols * 32), scales[expert]
                    )
                )

    def test_selector_requires_cross_sample_improvement(self):
        records = [((1, 4), [50.0, 110.0]), ((2, 4), [90.0, 90.0])]
        self.assertEqual(select_pair(records, [100.0, 100.0])["profile_ids"], [2, 4])

    def test_exact_check_reports_small_changes_without_accepting_them(self):
        expected = torch.ones((2, 8), dtype=torch.bfloat16)
        actual = expected.clone()
        actual[1, 3] = 1.0078125
        report = exact_comparison(actual, expected)
        self.assertFalse(report["exact"])
        self.assertEqual(report["mismatched_elements"], 1)
        self.assertEqual(report["examples"][0]["index"], [1, 3])
        self.assertEqual(report["max_abs"], 0.0078125)
        self.assertTrue(exact_comparison(expected.clone(), expected)["exact"])

    def test_output_gate_rejects_nonfinite_and_empty_row_leak(self):
        expected = torch.ones((2, 768), dtype=torch.bfloat16)
        expected[0] = 0
        self.assertEqual(error_metrics(expected, expected)["global_rel_l2"], 0.0)
        for bad in (float("nan"), 1e-10):
            actual = expected.clone()
            actual[0, 0] = bad
            with self.assertRaises(AssertionError):
                error_metrics(actual, expected)


if __name__ == "__main__":
    unittest.main()
