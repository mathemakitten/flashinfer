"""Independent correctness must not be replaced by backend agreement."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/moe_ep"))
import nt4_checkpoint_replay as replay


class Tests(unittest.TestCase):
    def setUp(self):
        self.emit = patch.object(replay, "emit").start()
        self.addCleanup(patch.stopall)

    def test_independent_pass_with_baseline_disagreement(self):
        reference = torch.ones(3, 4)
        errors, agreement = replay.qualification_comparisons(
            reference.clone(), reference, reference[:2] * 1.04, 2
        )
        self.assertEqual(errors, [])
        self.assertTrue(all(v["status"] == "DIFFER" for v in agreement.values()))
        self.assertTrue(
            any(
                c.kwargs.get("numerical_gate") is False
                for c in self.emit.call_args_list
            )
        )

    def test_agreeing_backends_cannot_hide_independent_failure(self):
        reference = torch.ones(2, 4)
        actual = reference * 1.04
        errors, _ = replay.qualification_comparisons(
            actual, reference, actual.clone(), 2
        )
        self.assertTrue(any("megamoe_vs_independent_reference" in e for e in errors))

    def test_zero_reference_row_still_gates(self):
        reference = torch.tensor([[100.0, 100.0], [0.0, 0.0]])
        actual = reference.clone()
        actual[1, 0] = 1e-12
        errors, _ = replay.qualification_comparisons(
            actual, reference, actual.clone(), 2
        )
        self.assertTrue(any("exact-zero reference row" in e for e in errors))

    def test_nonfinite_baseline_still_fails(self):
        reference = torch.ones(2, 4)
        baseline = reference.clone()
        baseline[0, 0] = float("nan")
        errors, _ = replay.qualification_comparisons(reference, reference, baseline, 2)
        self.assertTrue(any("nonfinite backend output" in e for e in errors))

    def test_nonfinite_candidate_still_fails(self):
        reference = torch.ones(2, 4)
        actual = reference.clone()
        actual[0, 0] = float("inf")
        errors, _ = replay.qualification_comparisons(actual, reference, reference, 2)
        self.assertTrue(
            any("megamoe_vs_independent_reference: nonfinite" in e for e in errors)
        )

    def test_regular_compare_retains_graph_and_hybrid_gate(self):
        with self.assertRaises(AssertionError):
            replay.compare(
                torch.ones(2, 4) * 1.04, torch.ones(2, 4), "nvls_hybrid_vs_direct"
            )

    def test_backend_shape_mismatch_still_fails(self):
        with self.assertRaisesRegex(AssertionError, "shape mismatch"):
            replay.report_backend_agreement(
                torch.ones(2, 4), torch.ones(1, 4), "baseline"
            )


if __name__ == "__main__":
    unittest.main()
