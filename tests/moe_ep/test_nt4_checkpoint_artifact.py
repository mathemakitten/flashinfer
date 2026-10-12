"""Run directly on CPU; verifies artifact integrity and canonical scale layout."""

import json
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/moe_ep"))
from nt4_checkpoint_artifact import (
    SCHEMA,
    canonical_scale,
    load_payload,
    save_manifest,
    save_payload,
    validate_sample,
)
from nt4_checkpoint_replay import compare
from nt4_checkpoint_capture import revision
from nt4_token_sweep import make_sweep_sample, token_census


class TokenSweepTest(unittest.TestCase):
    def pool(self):
        return dict(
            x=torch.arange(3 * 768).reshape(3, 768).bfloat16(),
            ids=torch.tensor([[1] * 10, [-1] * 10, [2] * 5 + [-1] * 5]),
            probs=torch.tensor([[0.0] * 10, [1.0] * 10, [0.5] * 10]),
        )

    def test_census_distinguishes_padding_and_zero_weight(self):
        result = token_census(self.pool())
        self.assertEqual(result["active_routed_rows"], 2)
        self.assertEqual(result["positive_weight_rows"], 1)
        self.assertEqual(result["padding_rows"], 1)
        self.assertEqual(result["valid_expert_slots"], 15)

    def test_resample_excludes_padding_preserves_partial_routes(self):
        sample = make_sweep_sample(self.pool(), 5, 8, 0)
        self.assertEqual(token_census(sample)["active_routed_rows"], 5)
        self.assertTrue(bool((sample["ids"][5:] == -1).all()))
        self.assertTrue(bool((sample["probs"][sample["ids"] < 0] == 0).all()))
        torch.testing.assert_close(sample["x"][1], self.pool()["x"][2])

    def test_empty_and_oversized_sweep_rejected(self):
        pool = self.pool()
        pool["ids"].fill_(-1)
        with self.assertRaises(ValueError):
            make_sweep_sample(pool, 1, 64, 0)
        with self.assertRaises(ValueError):
            make_sweep_sample(self.pool(), 65, 64, 0)


class ArtifactTest(unittest.TestCase):
    def test_revision_without_git_or_git_directory(self):
        failures = [
            FileNotFoundError("git"),
            subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"]),
            subprocess.TimeoutExpired("git", 10),
        ]
        with tempfile.TemporaryDirectory() as root:
            for failure in failures:
                with patch(
                    "nt4_checkpoint_capture.subprocess.check_output",
                    side_effect=failure,
                ):
                    result = revision(Path(root), Path(root), "mcore")
                self.assertFalse(result["git_available"])
                self.assertIsNone(result["head"])
                self.assertFalse((Path(root) / "mcore-tracked.patch").exists())

    def test_revision_records_git_when_available(self):
        with tempfile.TemporaryDirectory() as root:
            with patch(
                "nt4_checkpoint_capture.subprocess.check_output",
                side_effect=["abc\n", " M x\n", "patch\n"],
            ):
                result = revision(Path(root), Path(root), "mcore")
            self.assertTrue(result["git_available"])
            self.assertEqual(result["head"], "abc")
            self.assertEqual(
                (Path(root) / "mcore-tracked.patch").read_text(), "patch\n"
            )

    def sample(self, n=2):
        return dict(
            x=torch.zeros(n, 768, dtype=torch.bfloat16),
            ids=torch.arange(10).expand(n, 10).clone(),
            probs=torch.ones(n, 10, dtype=torch.float32),
            baseline=torch.zeros(n, 768, dtype=torch.bfloat16),
        )

    def test_scale_mapping_matches_independent_index_formula(self):
        # Nontrivial row and column groups, including padded scale columns.
        rows, cols = 256, 160
        raw = (torch.arange(rows * 8) % 251).to(torch.uint8)
        logical = canonical_scale(raw, rows, cols)
        for r in range(rows):
            for c in range(cols // 32):
                offset = (
                    (((r // 128) * 2 + c // 4) * 32 + r % 32) * 4 + (r % 128) // 32
                ) * 4 + c % 4
                self.assertEqual(logical[r, c], raw[offset])

    def test_reject_malformed_scale(self):
        with self.assertRaises(ValueError):
            canonical_scale(torch.zeros(17, dtype=torch.uint8), 128, 768)

    def test_save_load_preserves_raw_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "weights.pt"
            payload = dict(q=torch.arange(256).to(torch.uint8))
            hashes = save_payload(path, payload)
            self.assertTrue(torch.equal(load_payload(path, hashes)["q"], payload["q"]))
            with self.assertRaises(FileExistsError):
                save_payload(path, payload)

    def test_corruption_detected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sample.pt"
            hashes = save_payload(path, self.sample())
            changed = self.sample()
            changed["x"][0, 0] = 1
            torch.save(changed, path)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_payload(path, hashes)

    def test_missing_tensor_detected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sample.pt"
            hashes = save_payload(path, self.sample())
            del hashes["ids"]
            with self.assertRaisesRegex(ValueError, "keys"):
                load_payload(path, hashes)

    def test_empty_and_masked_samples(self):
        validate_sample(self.sample(0))
        sample = self.sample()
        sample["ids"].fill_(-1)
        sample["probs"].zero_()
        validate_sample(sample)
        sample["probs"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "probabilities"):
            validate_sample(sample)

    def test_out_of_range_route_and_nonfinite_sample(self):
        sample = self.sample()
        sample["ids"][0, 0] = 512
        with self.assertRaisesRegex(ValueError, "expert IDs"):
            validate_sample(sample)
        sample = self.sample()
        sample["x"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_sample(sample)

    def test_manifest_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "manifest.json"
            save_manifest(path, dict(schema=SCHEMA, capture_complete=False))
            save_manifest(path, dict(schema=SCHEMA, capture_complete=True))
            self.assertTrue(json.loads(path.read_text())["capture_complete"])
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    @patch("nt4_checkpoint_replay.emit")
    def test_direct_comparison_rejects_local_error_hidden_by_global_norm(self, emit):
        ref = torch.ones(2, 768, dtype=torch.bfloat16)
        ref[0] *= 10000
        actual = ref.clone()
        actual[1] *= 2
        with self.assertRaisesRegex(AssertionError, "per-token"):
            compare(actual, ref, "test")
        compare(ref[:0], ref[:0], "empty")


if __name__ == "__main__":
    unittest.main()
