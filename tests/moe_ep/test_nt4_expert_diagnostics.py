"""CPU checks for untimed expert reference diagnostics."""

import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

path = Path(__file__).resolve().parents[2] / "examples/moe_ep/nt4_expert_diagnostics.py"
spec = importlib.util.spec_from_file_location("nt4_expert_diagnostics", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DiagnosticsTest(unittest.TestCase):
    def test_exact_e8m0_all_finite_exponents(self):
        import math

        codes = torch.arange(255, dtype=torch.uint8)
        expected = torch.tensor(
            [math.ldexp(1.0, b - 127) for b in range(255)], dtype=torch.float64
        )
        self.assertTrue(torch.equal(module.decode_e8m0_exact(codes), expected))
        # Exercise noncontiguous inputs and preserve their shape.
        matrix = codes[:12].reshape(3, 4).T
        self.assertTrue(
            torch.equal(module.decode_e8m0_exact(matrix), expected[:12].reshape(3, 4).T)
        )
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            module.decode_e8m0_exact(torch.tensor([255], dtype=torch.uint8))

    def test_sample2_scale_decode_false_rejection(self):
        # Full 32-code block from expert121/source row95/slot0. The actual
        # scale byte is123; CUDA double pow returned one ULP below 2^-4.
        codes = torch.tensor(
            [
                [
                    0,
                    0,
                    65,
                    0,
                    0,
                    0,
                    0,
                    87,
                    85,
                    80,
                    0,
                    0,
                    0,
                    0,
                    75,
                    47,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    119,
                    0,
                    0,
                    21,
                    36,
                    0,
                    32,
                ]
            ],
            dtype=torch.uint8,
        )
        scale = module.decode_e8m0_exact(torch.tensor([[123]], dtype=torch.uint8))
        self.assertEqual(scale.item(), 0.0625)
        actual = codes.view(torch.float8_e4m3fn).double() * scale
        reference = actual.clone()
        reference[0, 25] = 16.0
        # Other coordinates agree and cannot explain the rejection. The
        # changed coordinate's independently computed bounds come from log.
        lower, upper = reference.clone(), reference.clone()
        lower[0, 25] = 15.467991788537296
        upper[0, 25] = 15.532042378980043
        distorted = torch.nextafter(scale, torch.zeros_like(scale))
        self.assertEqual(distorted.item(), 0.06249999999999999)
        self.assertFalse(
            bool(
                module.boundary_blocks(
                    actual, reference, lower, upper, distorted, self.quantize
                ).all()
            )
        )
        self.assertTrue(
            bool(
                module.boundary_blocks(
                    actual, reference, lower, upper, scale, self.quantize
                ).all()
            )
        )
        # Correct decoding does not permit an actual wrong scale byte.
        wrong = module.decode_e8m0_exact(torch.tensor([[124]], dtype=torch.uint8))
        self.assertFalse(
            bool(
                module.boundary_blocks(
                    actual, reference, lower, upper, wrong, self.quantize
                ).all()
            )
        )

    @staticmethod
    def quantize(value):
        blocks = value.reshape(*value.shape[:-1], -1, 32)
        maximum = blocks.abs().amax(-1).clamp_min(1e-30)
        scales = 2.0 ** torch.ceil(torch.log2(maximum / 448.0))
        q = (value / scales.repeat_interleave(32, -1)).to(torch.float8_e4m3fn)
        return q, scales

    def test_boundary_gate_observed_cases_and_corruption(self):
        # FP64 FC1 values from the two captured failures. Build independent
        # operands yielding these dots; do not feed measured errors to bounds.
        for z, actual_value, expected_value in [
            (0.9195698201656342, 0.8125, 0.875),
            (0.953097403049469, 0.875, 0.9375),
        ]:
            x = torch.zeros(1, 768, dtype=torch.float64)
            x[0, 0] = 1
            up = torch.zeros(32, 768, dtype=torch.float64)
            up[:, 0] = 2
            up[0, 0] = z
            z64 = x @ up.T
            activation = (16 * torch.tanh(z64 / 16)).square()
            q, scales = self.quantize(activation.float())
            reference = q.float() * scales.repeat_interleave(32, -1)
            self.assertEqual(reference[0, 0], expected_value)
            actual = reference.clone()
            actual[0, 0] = actual_value
            lower, upper = module.fc1_activation_interval(x, up, z64)

            def check(value=actual, scale=scales, lo=lower, hi=upper):
                return module.boundary_blocks(
                    value, reference, lo, hi, scale, self.quantize
                )

            self.assertTrue(bool(check().all()))
            result = module.validate_intermediate(actual, [reference], boundary=check)
            self.assertEqual(result["boundary_blocks"], 1)
            # Adjacent but far from a boundary is not sufficient.
            self.assertFalse(bool(check(lo=activation, hi=activation).all()))
            bad = actual.clone()
            bad[0, 0] -= 0.0625  # skip a code
            self.assertFalse(bool(check(value=bad).all()))
            self.assertFalse(bool(check(scale=scales * 2).all()))
            bad = actual.clone()
            bad[0, 4] = 0  # unrelated corruption in same block
            with self.assertRaisesRegex(AssertionError, "matches no"):
                module.validate_intermediate(
                    bad, [reference], boundary=lambda: check(value=bad)
                )
            # Uncertainty spanning more than two adjacent scales fails closed.
            self.assertFalse(bool(check(lo=lower * 0.5, hi=upper * 2).all()))

    def test_boundary_diagnostics_identify_rejection_without_changing_gate(self):
        reference = torch.full((2, 32), 16.0)
        actual = reference.clone()
        actual[:, 25] = 15.0
        # Row 0: activation excludes 15.5. Row 1: midpoint reachable but
        # block amax crosses the independent E8M0 scale threshold at 14.
        lower = torch.full((2, 32), 15.9, dtype=torch.float64)
        upper = torch.full_like(lower, 16.1)
        lower[1] = 13.9
        # Code16 cannot be produced by this coordinate even though the block
        # scale and the changed coordinate individually look admissible.
        lower[1, 0] = upper[1, 0] = 14.0
        _, scales = self.quantize(reference)
        records = []
        plain = module.boundary_blocks(
            actual, reference, lower, upper, scales, self.quantize
        )
        reported = module.boundary_blocks(
            actual, reference, lower, upper, scales, self.quantize, diagnostics=records
        )
        self.assertTrue(torch.equal(plain, reported))
        self.assertFalse(bool(plain.any()))
        self.assertEqual(records[0]["reasons"], ["element_outside_boundary_rule"])
        self.assertEqual(records[0]["details"][0]["channel"], 25)
        self.assertEqual(records[0]["details"][0]["midpoint"], 15.5)
        self.assertFalse(records[0]["details"][0]["interval_reaches_midpoint"])
        self.assertIn("ambiguous_scale_interval", records[1]["reasons"])
        self.assertTrue(records[1]["details"][0]["interval_reaches_midpoint"])
        records = []
        masked = module.boundary_blocks(
            actual,
            reference,
            lower,
            upper,
            scales,
            self.quantize,
            diagnostics=records,
            diagnostic_mask=torch.tensor([False, True]),
        )
        self.assertTrue(torch.equal(plain, masked))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["expert_row"], 1)

    def test_joint_scale_observed_256_boundary_and_corruptions(self):
        # Raw codes and the two relevant intervals from the expert154 log.
        # Other coordinates are synthetic exact intervals, so this exercises
        # the reported boundary geometry, not a replay of the full capture.
        codes = torch.tensor(
            [
                [
                    0,
                    0,
                    104,
                    125,
                    99,
                    0,
                    0,
                    0,
                    61,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    84,
                    0,
                    0,
                    0,
                    126,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ],
            dtype=torch.uint8,
        )
        scale = torch.tensor([[2.0**-8]], dtype=torch.float64)
        actual = codes.view(torch.float8_e4m3fn).double() * scale
        reference = actual.clone()
        reference[0, 8] = 0.005859375
        lo, hi = actual.clone(), actual.clone()
        lo[0, 8], hi[0, 8] = 0.005629418052026363, 0.006596764640921462
        lo[0, 23], hi[0, 23] = 1.7382664611905498, 1.7540345977431842

        def check(a=actual, l=lo, h=hi, s=scale):
            return module.boundary_blocks(a, reference, l, h, s, self.quantize)

        self.assertTrue(bool(check().all()))
        # No value in the block can produce the stored smaller scale now.
        bad_lo = lo.clone()
        bad_lo[0, 23] = 1.751
        self.assertFalse(bool(check(l=bad_lo).any()))
        # Adjacent value, but its rounding cell is outside the independent box.
        bad_hi = hi.clone()
        bad_hi[0, 8] = 0.006
        self.assertFalse(bool(check(h=bad_hi).any()))
        # Unchanged coordinates must be feasible too, not just changed ones.
        bad_lo, bad_hi = lo.clone(), hi.clone()
        bad_lo[0, 2] = bad_hi[0, 2] = actual[0, 2] * 0.5
        self.assertFalse(bool(check(l=bad_lo, h=bad_hi).any()))
        self.assertFalse(bool(check(s=scale * 4).any()))

    def test_joint_witness_upper_scale_and_tie_fail_closed(self):
        actual = torch.zeros(1, 32, dtype=torch.float64)
        actual[0, 0] = 1.75
        lo, hi = actual.clone(), actual.clone()
        lo[0, 0], hi[0, 0] = 1.74, 1.76
        scale = torch.tensor([[2.0**-7]], dtype=torch.float64)
        self.assertTrue(
            bool(
                module.boundary_blocks(
                    actual, actual, lo, hi, scale, self.quantize
                ).all()
            )
        )
        # Upper scale requires an amax above1.75, unavailable here.
        hi[0, 0] = 1.75
        self.assertFalse(
            bool(module.joint_scale_witness(actual, lo, hi, scale, self.quantize).any())
        )
        # A singleton at an FP8 midpoint cannot serve as our open-cell witness.
        actual[0, 1] = 0.00634765625
        lo[0, 1] = hi[0, 1] = 0.006103515625
        self.assertFalse(
            bool(
                module.joint_scale_witness(
                    actual, lo, hi, scale / 2, self.quantize
                ).any()
            )
        )

    def test_boundary_gate_preserves_l2_limit(self):
        ref = torch.full((1, 32), 0.875)
        actual = torch.full_like(ref, 0.8125)
        lo = torch.full((1, 32), 0.84374, dtype=torch.float64)
        hi = torch.full_like(lo, 0.84376)
        _, scale = self.quantize(lo.float())
        with self.assertRaisesRegex(AssertionError, "relative L2"):
            module.validate_intermediate(
                actual,
                [ref],
                boundary=lambda: module.boundary_blocks(
                    actual, ref, lo, hi, scale, self.quantize
                ),
            )

    def test_interval_cancellation_relu_and_nonfinite(self):
        x = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
        up = torch.tensor([[100.0, -100.0, 0.0], [1.0, -2.0, 0.0]], dtype=torch.float64)
        z = x @ up.T
        lower, upper = module.fc1_activation_interval(x, up, z)
        self.assertTrue(bool((lower == 0).all()))
        self.assertGreater(upper[0, 0], upper[0, 1])
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            module.fc1_activation_interval(x * float("nan"), up, z)

    def test_prequant_midpoint_and_quantizer_attribution(self):
        result = module.prequant_record(
            1.0, 0.84375 - 2**-24, 1.0, 0.84375 + 2**-24, 0.8125, 0.875, 2**-8
        )
        self.assertTrue(result["adjacent_codes_at_kernel_scale"])
        self.assertTrue(result["quantizer_reproduced"])
        self.assertEqual(result["midpoint"], 0.84375)
        self.assertLess(result["kernel_distance_from_midpoint"], 0.0)
        self.assertGreater(result["reference_distance_from_midpoint"], 0.0)
        wrong = module.prequant_record(
            1.0, 0.84375 + 2**-24, 1.0, 0.84375 + 2**-24, 0.8125, 0.875, 2**-8
        )
        self.assertFalse(wrong["quantizer_reproduced"])
        with self.assertRaisesRegex(AssertionError, "instrumentation"):
            module.prequant_record(float("nan"), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    def test_block_gate_accepts_explained_boundary_only(self):
        original = torch.zeros(2, 64)
        original[:, 9] = 6.5
        precise = original.clone()
        precise[:, 9] = 6.0
        actual = original.clone()
        actual[1, 9] = 6.0
        # A single-element row gives a >1.5% relative change, so provide
        # representative nonzero context while retaining the midpoint case.
        original[:, 10:] = precise[:, 10:] = actual[:, 10:] = 10.0
        metrics = module.validate_intermediate(actual, [original, precise])
        self.assertEqual(metrics["changed_elements"], 1)
        corrupt = actual.clone()
        corrupt[0, 9] = 5.5
        with self.assertRaisesRegex(AssertionError, "matches no"):
            module.validate_intermediate(corrupt, [original, precise])
        corrupt = actual.clone()
        corrupt[0, 0] = float("nan")
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            module.validate_intermediate(corrupt, [original, precise])
        with self.assertRaisesRegex(AssertionError, "matches no"):
            module.validate_intermediate(torch.ones(1, 32), [torch.zeros(1, 32)])

    def test_block_gate_rejects_elementwise_cherry_picking(self):
        first = torch.ones(1, 32) * 10
        second = first.clone()
        second[0, :2] = 9.5
        mixed = first.clone()
        mixed[0, 0] = 9.5
        with self.assertRaisesRegex(AssertionError, "matches no"):
            module.validate_intermediate(mixed, [first, second])

    def test_fc2_gate_still_rejects_corrupt_terms_and_combine(self):
        source = path.parent / "nt4_megamoe_bringup.py"
        spec2 = importlib.util.spec_from_file_location("nt4_bringup_gate", source)
        gate = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(gate)
        terms = torch.ones(1, 2, 4)
        ids = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.5, 0.5]])
        output = torch.ones(1, 4, dtype=torch.bfloat16)
        gate.validate_contributions(
            output, output, terms, terms, ids, torch.ones_like(terms), 4096, weights
        )
        corrupt = terms.clone()
        corrupt[0, 1, 0] += 0.125
        with self.assertRaisesRegex(AssertionError, "rounding budget"):
            gate.validate_contributions(
                output,
                output,
                corrupt,
                terms,
                ids,
                torch.ones_like(terms),
                4096,
                weights,
            )
        with self.assertRaises(AssertionError):
            gate.validate_contributions(
                output + 1,
                output,
                terms,
                terms,
                ids,
                torch.ones_like(terms),
                4096,
                weights,
            )

    def test_snapshot_independent_padding_and_source_metadata(self):
        width = 4096
        sizes = {
            "token_src_metadata": 128 * 8,
            "fc1_output": 128 * width,
            "fc1_output_sf": 256 * (width // 32),
        }
        offsets = {}
        total = 0
        for name, size in sizes.items():
            offsets[name] = total
            total += size
        storage = torch.zeros(total, dtype=torch.uint8)

        def view(name):
            return storage[offsets[name] : offsets[name] + sizes[name]]

        target = (2 << 48) | (6 << 32) | 30
        view("token_src_metadata").view(torch.int64)[64] = target
        # Stale metadata in a padding row must not be matched.
        view("token_src_metadata").view(torch.int64)[65] = target
        q = (
            torch.full((width,), 6.0, dtype=torch.float32)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        view("fc1_output").reshape(128, width)[64] = q
        # Build expected SF atoms independently as [M atom,K atom,32,4,4].
        sf = view("fc1_output_sf").reshape(2, 32, 32, 4, 4)
        sf[1, :, 0, 0, :] = 127
        kernel = SimpleNamespace(
            token_padding_block=64,
            sf_padding_block=128,
            _local_offsets=offsets,
            _local_region_by_name={
                name: SimpleNamespace(nbytes=size, shape=(128, width))
                for name, size in sizes.items()
            },
        )
        workspace = SimpleNamespace(
            _frontend=SimpleNamespace(
                _mega=SimpleNamespace(kernel=kernel, local_workspace=storage)
            )
        )
        result = module.snapshot_fc1(workspace, [2, 1], 1, 2, 30, 6)
        self.assertEqual(result["pool_row"], 64)
        self.assertEqual(result["sf_row"], 128)
        self.assertEqual(result["intermediate"], [6.0] * width)
        meta, bytes_, scales = module.snapshot_fc1_expert(workspace, [2, 1], 1)
        self.assertEqual(meta.tolist(), [target])
        self.assertEqual(bytes_.shape, (1, width))
        self.assertTrue(bool((scales == 127).all()))
        self.assertEqual(bytes_[0].tolist(), result["fp8_bytes"])
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            module.snapshot_fc1(workspace, [2, 1], 1, 1, 30, 6)

    def test_cancellation_and_late_weight_semantics(self):
        x = torch.tensor([[1.0, 1.0]])
        up = torch.eye(2)
        down = torch.tensor([[1.0, -1.0]])
        z = x @ up.T
        a = (16 * torch.tanh(z / 16)).square()
        before = a.clone()
        result = module.diagnose_expert(x, up, down, z, a, lambda v: v, 0, 0)
        self.assertEqual(result["reference_fp64_fc2"], 0.0)
        self.assertGreater(result["fc2_abs_product_sum"], 1.0)
        self.assertEqual(
            result["variants"]["fp64_fc1"]["changed_intermediate_elements"], 0
        )
        torch.testing.assert_close(a, before, rtol=0, atol=0)

    def test_early_routing_weights_preserved(self):
        x = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        up = torch.eye(2)
        down = torch.tensor([[1.0, 1.0]])
        weights = torch.tensor([0.25, 0.5])
        z = x @ up.T
        a = (16 * torch.tanh(z / 16)).square() * weights[:, None]
        result = module.diagnose_expert(x, up, down, z, a, lambda v: v, 1, 0, weights)
        self.assertEqual(
            result["variants"]["fp64_fc1"]["changed_intermediate_elements"], 0
        )
        self.assertEqual(result["reference_fp64_fc2"], a[1].double().sum().item())


if __name__ == "__main__":
    unittest.main()
