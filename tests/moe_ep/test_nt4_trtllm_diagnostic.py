"""CPU checks for failure payloads, metrics and MXFP8 intermediate layouts."""

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
    "diag", scripts / "nt4_trtllm_diagnostic.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RuntimeArray:
    def __iter__(self):
        return iter([39, 428])


class DiagnosticTest(unittest.TestCase):
    def test_workspace_ownership_is_acquired_at_record(self):
        tap_spec = importlib.util.spec_from_file_location(
            "tap", scripts.parents[1] / "flashinfer/fused_moe/nt4_diagnostic.py"
        )
        tap = importlib.util.module_from_spec(tap_spec)
        tap_spec.loader.exec_module(tap)
        tensors = [torch.arange(4) + i for i in range(5)]
        capsules = [torch.utils.dlpack.to_dlpack(t) for t in tensors]
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            tap.record(capsules, [39, 428])
        # No second DLPack conversion is allowed after record returns.
        with patch.object(
            torch, "from_dlpack", side_effect=AssertionError("converted twice")
        ):
            captured, tactic = tap.take()
        self.assertEqual(tactic, [39, 428])
        for original, retained in zip(tensors, captured):
            self.assertEqual(original.data_ptr(), retained.data_ptr())
            self.assertTrue(torch.equal(original, retained))
        with self.assertRaisesRegex(RuntimeError, "No TRTLLM workspace"):
            tap.take()
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            tap.record(tensors[:4] + [None], [39, 428])
            retained, _ = tap.take()
            self.assertIsNone(retained[4])
            for index in range(4):
                missing = list(tensors)
                missing[index] = None
                with self.assertRaisesRegex(
                    RuntimeError, "required workspace is missing"
                ):
                    tap.record(missing, [39, 428])

    def test_tactic_serialization(self):
        self.assertEqual(m.plain_tactic(RuntimeArray()), [39, 428])
        self.assertEqual(m.plain_tactic(-1), -1)
        self.assertIsNone(m.plain_tactic(None))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "legacy.pt"
            torch.save({"tactic": RuntimeArray()}, p)
            with self.assertRaises(Exception):
                m.load_snapshot(p)
            self.assertEqual(
                m.load_snapshot(p, trusted_legacy=True)["tactic"], [39, 428]
            )

    def test_metrics_and_cancellation(self):
        r = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
        a = r.clone()
        a[0, 0] += 0.5
        result = m.metrics(a, r)
        self.assertAlmostEqual(result["global_rel_l2"], 0.1)
        self.assertAlmostEqual(result["max_row_rel_l2"], 0.1)
        self.assertEqual(result["worst_row"], 0)
        a[1, 0] = 0.001
        self.assertEqual(m.metrics(a, r)["worst_row"], 1)
        a[1, 0] = float("nan")
        self.assertFalse(m.metrics(a, r)["finite"])

    def test_swizzled_scales_cross_tile_boundaries(self):
        # Build layout by explicit five-axis indexing, independently of decoder.
        raw = torch.zeros((2, 2, 32, 4, 4), dtype=torch.uint8)
        rows = torch.tensor([0, 31, 32, 127, 128, 255])
        expected = []
        for row in rows.tolist():
            values = []
            for k in range(8):
                byte = 120 + (row + k) % 10
                raw[row // 128, k // 4, row % 32, (row % 128) // 32, k % 4] = byte
                values.append(2.0 ** (byte - 127))
            expected.append(values)
        self.assertTrue(
            torch.equal(
                m.sf_rows(raw, rows, 256), torch.tensor(expected, dtype=torch.float64)
            )
        )

    def test_8x4_scales_and_export_selection(self):
        raw = torch.zeros((17, 2, 8, 4), dtype=torch.uint8)
        rows = torch.tensor([0, 7, 8, 31, 32, 127, 128])
        expected = []
        for row in rows.tolist():
            values = []
            for k in range(8):
                byte = 115 + (row + k) % 10
                raw[row // 8, k // 4, row % 8, k % 4] = byte
                values.append(2.0 ** (byte - 127))
            expected.append(values)
        self.assertTrue(
            torch.equal(
                m.sf_rows(raw, rows, 256, "8x4"),
                torch.tensor(expected, dtype=torch.float64),
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "expanded-config.json"
            configs = [
                dict(eltwiseActType="clampedrelu2", sfLayoutC="8x4"),
                dict(
                    eltwiseActType="none",
                    dtypeA="mxe4m3",
                    dtypeB="mxe4m3",
                    dtypeC="bf16",
                    sfLayoutB="8x4",
                ),
            ]
            p.write_text(json.dumps(dict(configs=configs)))
            self.assertEqual(m.intermediate_sf_layout(tmp), "8x4")
            configs[1]["sfLayoutB"] = "128x4"
            p.write_text(json.dumps(dict(configs=configs)))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                m.intermediate_sf_layout(tmp)

    def test_failure_snapshot_is_self_contained(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = {
                "w1": torch.ones(1, 32, 32, dtype=torch.uint8),
                "s1": torch.ones(1, 32, 1, dtype=torch.uint8),
            }
            h = SimpleNamespace(
                rank=2,
                x=torch.ones(2, 32),
                ids=torch.zeros(2, 10, dtype=torch.int64),
                probs=torch.ones(2, 10),
                clamp_limit=torch.tensor([16.0]),
                weights=w,
                direct=SimpleNamespace(weights=w, layer_number=14),
            )
            with (
                patch.dict(os.environ, {"NT4_TRT_FAILURE_ROOT": tmp}),
                patch.dict(
                    sys.modules,
                    {
                        "flashinfer.fused_moe.nt4_diagnostic": SimpleNamespace(
                            last_tactic=RuntimeArray()
                        )
                    },
                ),
            ):
                m.save_failure(h, torch.ones(2, 32), torch.ones(2, 32) * 2)
            manifest = next(Path(tmp).glob("numerical-failures/*/manifest.json"))
            self.assertTrue(json.loads(manifest.read_text())["complete"])
            saved = torch.load(manifest.parent / "failure.pt", weights_only=True)
            self.assertEqual(saved["tactic"], [39, 428])
            self.assertEqual(saved["rank"], 2)
            self.assertEqual(saved["layer"], 14)
            self.assertEqual(
                saved["packed_weight_hashes"]["w1"], m.tensor_hash(w["w1"])
            )
            w["w1"][0, 0, 0] = 2
            self.assertNotEqual(
                saved["packed_weight_hashes"]["w1"], m.tensor_hash(w["w1"])
            )
            self.assertEqual(saved["weights"]["w1"][0, 0, 0], 1)


if __name__ == "__main__":
    unittest.main()
