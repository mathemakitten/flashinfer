"""GPU checks for masking, FP32 cancellation and invalid-slot poison handling."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/moe_ep"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA + Triton")
class FusedGlueTest(unittest.TestCase):
    def test_routes_and_partial_with_poisoned_padding(self):
        from nt4_nvls_kernels import route_mask, weighted_partial

        # Non-block-aligned route count; every owner boundary and padding.
        ids = torch.tensor(
            [[0, 127, 128, 255, 256, 383, 384, 511, -1, -1]], device="cuda"
        ).repeat(7, 1)
        probs = torch.linspace(0.1, 1.0, 10, device="cuda").expand(7, -1).contiguous()
        torch.manual_seed(91)
        terms = torch.randn(7, 10, 768, device="cuda", dtype=torch.bfloat16)
        terms[:, 0] = 64
        terms[:, 1] = -32
        local_ids, local_probs = torch.empty_like(ids), torch.empty_like(probs)
        output = torch.empty(7, 768, device="cuda", dtype=torch.float32)
        for rank in range(4):
            route_mask(ids, probs, local_ids, local_probs, rank)
            owned = (ids >= rank * 128) & (ids < (rank + 1) * 128)
            torch.testing.assert_close(
                local_ids, torch.where(owned, ids - rank * 128, -1), rtol=0, atol=0
            )
            torch.testing.assert_close(
                local_probs, torch.where(owned, probs, 0.0), rtol=0, atol=0
            )
            poisoned = torch.where(owned[:, :, None], terms, float("nan"))
            weighted_partial(poisoned, local_ids, local_probs, output)
            expected = torch.zeros_like(output)
            for slot in range(10):
                expected = (
                    expected
                    + torch.where(owned[:, slot, None], terms[:, slot].float(), 0.0)
                    * local_probs[:, slot, None]
                )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        local_ids.fill_(-1)
        local_probs.zero_()
        weighted_partial(terms, local_ids, local_probs, output)
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
