"""CPU tests of the bring-up acceptance policy; run directly with Python."""

import importlib.util
from pathlib import Path
import unittest

import torch

_path = Path(__file__).resolve().parents[2] / "examples/moe_ep/nt4_megamoe_bringup.py"
_spec = importlib.util.spec_from_file_location("nt4_bringup", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate = _module.validate_contributions


class ContributionValidationTest(unittest.TestCase):
    def test_late_weighting_checks_raw_terms_and_weighted_output(self):
        ref = torch.tensor([[[8.0, -4.0], [2.0, 6.0], [0.0, 0.0]]])
        ids = torch.tensor([[1, 2, -1]])
        weights = torch.tensor([[0.25, 0.5, 0.0]])
        expected = torch.tensor([[3.0, 2.0]], dtype=torch.bfloat16)
        validate(expected, expected, ref, ref, ids, routing_weights=weights)
        with self.assertRaises(AssertionError):
            validate(
                ref.sum(1).bfloat16(), expected, ref, ref, ids, routing_weights=weights
            )
        corrupted = ref.clone()
        corrupted[0, 0, 0] += 1
        with self.assertRaises(AssertionError):
            validate(expected, expected, corrupted, ref, ids, routing_weights=weights)

    def test_late_reducer_preserves_fma_product_before_cancellation(self):
        terms = torch.tensor([[[-1.0], [1.0078125]]])
        weights = torch.tensor([[1.0, 0.9922480583190918]])
        got = _module.weighted_bf16_recombine(terms, weights)
        expected = (
            (terms[:, 1].double() * weights[:, 1, None].double() - 1).float().bfloat16()
        )
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def fixture(self):
        # A well-conditioned second channel keeps the global L2 check active
        # while channel zero reproduces the observed cancellation case.
        ref = (
            torch.tensor(
                [
                    1.9609375,
                    -6.75,
                    -2.625,
                    2.265625,
                    -4.4375,
                    7.65625,
                    -0.35546875,
                    -2.421875,
                    2.0625,
                    2.453125,
                ]
            )
            .view(1, 10, 1)
            .expand(1, 10, 2)
            .clone()
        )
        ref[:, :, 1] = 10
        return ref, torch.arange(10).view(1, 10)

    def check(self, ref, terms, ids, actual=None):
        expected = ref.sum(1).to(torch.bfloat16)
        if actual is None:
            actual = terms.sum(1).to(torch.bfloat16)
        return validate(actual, expected, terms, ref, ids)

    def test_observed_one_step_cancellation_passes(self):
        ref, ids = self.fixture()
        terms = ref.clone()
        terms[0, 1, 0] = -6.71875
        metrics = self.check(ref, terms, ids)
        self.assertEqual(metrics["expert_max_budget_fraction"], 1)

    def test_fc2_cancellation_uses_product_magnitudes(self):
        ref = torch.tensor([[[-2.3126602172851562e-05, 1.0]]])
        terms = ref.clone()
        terms[0, 0, 0] = -2.4437904357910156e-05
        products = torch.tensor([[[97.76946947351098, 1.0]]])
        ids = torch.tensor([[387]])
        validate(
            terms.sum(1).bfloat16(),
            ref.sum(1).bfloat16(),
            terms,
            ref,
            ids,
            products,
            4096,
        )
        terms[0, 0, 0] = 0.1
        with self.assertRaisesRegex(AssertionError, "rounding budget"):
            validate(
                terms.sum(1).bfloat16(),
                ref.sum(1).bfloat16(),
                terms,
                ref,
                ids,
                products,
                4096,
            )

    def test_roundoff_bound_does_not_replace_vector_gate(self):
        ref = torch.tensor([[[1.0, 1.0]]])
        terms = ref + 0.125
        with self.assertRaisesRegex(AssertionError, "per-expert vector"):
            validate(
                terms.sum(1).bfloat16(),
                ref.sum(1).bfloat16(),
                terms,
                ref,
                torch.tensor([[0]]),
                torch.full_like(ref, 10000),
                4096,
            )

    def test_opposing_errors_cannot_hide(self):
        ref, ids = self.fixture()
        terms = ref.clone()
        terms[0, 1, 0] += 0.125
        terms[0, 5, 0] -= 0.125
        with self.assertRaisesRegex(AssertionError, "expert term"):
            self.check(ref, terms, ids)

    def test_power_of_two_boundary_rejects_two_steps(self):
        ref, ids = self.fixture()
        ref[0, 0, 0] = 1
        terms = ref.clone()
        terms[0, 0, 0] = 0.9921875
        with self.assertRaisesRegex(AssertionError, "expert term"):
            self.check(ref, terms, ids)

    def test_masked_stale_term_rejected(self):
        ref, ids = self.fixture()
        ref.zero_()
        ids.fill_(-1)
        terms = ref.clone()
        self.check(ref, terms, ids)
        terms[0, 0, 0] = 0.001
        with self.assertRaises(AssertionError):
            self.check(ref, terms, ids)

    def test_wrong_reduction_rejected(self):
        ref, ids = self.fixture()
        actual = ref.sum(1).to(torch.bfloat16)
        actual[0, 0] += 0.01
        with self.assertRaises(AssertionError):
            self.check(ref, ref, ids, actual)

    def test_nonfinite_rejected(self):
        ref, ids = self.fixture()
        terms = ref.clone()
        terms[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            self.check(ref, terms, ids)

    def test_global_l2_still_rejects_large_cancelled_error(self):
        ref, ids = self.fixture()
        ref = ref[:, :, :1]
        terms = ref.clone()
        terms[0, 1, 0] = -6.71875
        with self.assertRaisesRegex(AssertionError, "relative L2"):
            self.check(ref, terms, ids)


if __name__ == "__main__":
    unittest.main()
