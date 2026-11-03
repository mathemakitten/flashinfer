"""CPU tests for preserving baseline-agreement evidence."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

scripts = Path(__file__).resolve().parents[2] / "examples/moe_ep"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "baseline_diag", scripts / "nt4_baseline_diagnostic.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Tests(unittest.TestCase):
    def test_snapshot_scale_atom_and_tile_boundaries(self):
        # Independent fixed offsets from SfAtom's strides ((16,4),(0,1)).
        expected = {0: 0, 31: 496, 32: 4, 127: 508, 128: 16384}
        for row, first in expected.items():
            indices = m.snapshot_scale_indices(row, 4096)
            self.assertEqual(indices[:4].tolist(), list(range(first, first + 4)))
            self.assertEqual(indices[4].item(), first + 512)
            self.assertEqual(indices[-1].item(), first + 31 * 512 + 3)
        all_indices = torch.cat([m.snapshot_scale_indices(r, 4096) for r in range(128)])
        self.assertEqual(all_indices.unique().numel(), 128 * 128)
        self.assertEqual(all_indices.max().item(), 128 * 128 - 1)

    def test_boundary_candidates_require_reachable_rounding_boundary(self):
        def quant(v):
            return v.float().to(torch.float8_e4m3fn), torch.ones(
                v.shape[0], v.shape[1] // 32, dtype=torch.float64
            )

        ref = torch.ones(1, 64, dtype=torch.float64)
        lo, hi = ref.clone(), ref.clone()
        self.assertEqual(list(m.boundary_candidates(ref, lo, hi, quant)), [])
        lo[0, 3], hi[0, 3] = 1.06249, 1.06251
        candidates = list(m.boundary_candidates(ref, lo, hi, quant))
        self.assertEqual(len(candidates), 1)
        endpoint, block, candidate = candidates[0]
        self.assertEqual((endpoint, block), ("upper", 0))
        self.assertEqual(candidate[0, 3].item(), 1.125)
        self.assertTrue(torch.equal(candidate[:, 32:], ref[:, 32:]))
        self.assertEqual(int((candidate != ref).sum()), 1)
        self.assertEqual(ref[0, 3].item(), 1.0)

    def test_slot_interventions_preserve_and_reconstruct_weights(self):
        probs = torch.tensor([[0.2, 0.3, 0.5], [0.0, 0.7, 0.3]])
        before = probs.clone()
        parts = [m.isolate_slot(probs, i) for i in range(3)]
        self.assertTrue(torch.equal(sum(parts), probs))
        self.assertTrue(torch.equal(probs, before))
        for i, part in enumerate(parts):
            self.assertEqual(part.shape, probs.shape)
            self.assertTrue(torch.equal(part[:, i], probs[:, i]))
            self.assertEqual(
                int(torch.count_nonzero(part[:, [j for j in range(3) if j != i]])), 0
            )

    def test_comparison(self):
        b = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        a = b.clone()
        a[1, 0] = 1.1
        result = m.comparison(a, b)
        self.assertEqual(result["worst_row"], 1)
        self.assertAlmostEqual(result["max_row_rel_l2"], 0.1, places=6)
        self.assertFalse(result["exact"])

    def test_capture_preserves_three_outputs(self):
        from nt4_checkpoint_artifact import load_payload

        with tempfile.TemporaryDirectory() as tmp:
            replay = SimpleNamespace(
                group=None,
                layer_number=14,
                capacity=8,
                n=2,
                knobs={},
                weights={"w1": torch.ones(2, dtype=torch.uint8)},
            )
            sample = dict(
                x=torch.ones(2, 3),
                ids=torch.zeros(2, 1, dtype=torch.int64),
                probs=torch.ones(2, 1),
                baseline=torch.ones(2, 3),
            )
            actual = torch.ones(8, 3) * 2
            reference = torch.ones(8, 3) * 3

            def gather(out, obj, group=None):
                out[0] = obj

            with (
                patch.dict(os.environ, {"NT4_BASELINE_FAILURE_ROOT": tmp}),
                patch.object(m.dist, "get_rank", return_value=0),
                patch.object(m.dist, "get_world_size", return_value=1),
                patch.object(m.dist, "broadcast_object_list"),
                patch.object(m.dist, "all_gather_object", side_effect=gather),
            ):
                m.capture_failure(replay, sample, actual, reference, [["mismatch"]])
            manifest_path = next(Path(tmp).glob("*/manifest.json"))
            manifest = json.loads(manifest_path.read_text())
            self.assertTrue(manifest["capture_complete"])
            self.assertEqual(manifest["validation_errors"], [["mismatch"]])
            payload = load_payload(
                manifest_path.parent / "sample0-rank0.pt",
                manifest["ranks"][0]["samples"][0],
            )
            self.assertEqual(payload["megamoe"].shape, (2, 3))
            self.assertTrue(torch.equal(payload["baseline"], sample["baseline"]))
            self.assertTrue(torch.equal(payload["megamoe"], actual[:2]))
            self.assertTrue(torch.equal(payload["reference"], reference[:2]))


if __name__ == "__main__":
    unittest.main()
