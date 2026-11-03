"""CPU coverage of decode gating and instance isolation."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/moe_ep"))
from nt4_cutlass_decode_override import install_decode_override


def cutlass_fused_moe(x, *, profile_ids=None):
    return x, profile_ids


def flashinfer_cutlass_mxfp8_moe(x, *, out=None):
    return cutlass_fused_moe(x)


class Experts:
    def _flashinfer_forward(self, x):
        return flashinfer_cutlass_mxfp8_moe(x)


class OverrideTest(unittest.TestCase):
    def test_decode_only_and_restore(self):
        expert, other = Experts(), Experts()
        active = False
        restore = install_decode_override(expert, [39, 428], enabled=lambda: active)
        self.assertEqual(expert._flashinfer_forward(1), (1, None))
        active = True
        self.assertEqual(expert._flashinfer_forward(2), (2, [39, 428]))
        self.assertEqual(other._flashinfer_forward(2), (2, None))
        self.assertEqual(flashinfer_cutlass_mxfp8_moe(2), (2, None))
        active = False
        self.assertEqual(expert._flashinfer_forward(3), (3, None))
        restore()
        active = True
        self.assertEqual(expert._flashinfer_forward(4), (4, None))
        self.assertNotIn("_flashinfer_forward", vars(expert))

    def test_invalid_ids(self):
        for ids in ([39], [39, -1], [39, True]):
            with self.assertRaises(ValueError):
                install_decode_override(Experts(), ids, enabled=lambda: True)


if __name__ == "__main__":
    unittest.main()
