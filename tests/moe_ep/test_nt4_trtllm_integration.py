"""CPU tests for artifact isolation and graph-safe activation argument validation."""

import ast
from enum import IntEnum
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "nt4_export", ROOT / "flashinfer/jit/trtllm_nt4_export.py"
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class Activation(IntEnum):
    Swiglu = 3
    ClampedRelu2 = 12


class Quantization(IntEnum):
    MxFp8 = 1
    DeepSeekFp8 = 2


class IntegrationTest(unittest.TestCase):
    def test_export_requires_same_build_fc2_and_has_content_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact"
            (artifact / "cubins").mkdir(parents=True)
            (root / "validation.json").write_text(
                json.dumps({"status": "PASS", "consumer_compile": "PASS"})
            )
            (artifact / "Enums.h").write_text("enum {ClampedRelu2};")
            (artifact / "flashinferMetaInfo.h").write_text("manifest")
            cubin = artifact / "cubins/test.cubin"
            cubin.write_bytes(b"first")
            configs = [{"eltwiseActType": "clampedrelu2"}]
            config_path = root / "expanded-config.json"
            config_path.write_text(json.dumps({"configs": configs}))
            with self.assertRaisesRegex(ValueError, "both clamped FC1"):
                export.inspect_nt4_export(tmp, ("Enums.h",))
            configs.append(
                {"eltwiseActType": "none", "dtypeC": "bf16", "routeAct": False}
            )
            config_path.write_text(json.dumps({"configs": configs}))
            _, first = export.inspect_nt4_export(tmp, ("Enums.h",))
            # New process semantics: cache is deliberately immutable within a process.
            export.inspect_nt4_export.cache_clear()
            cubin.write_bytes(b"second")
            _, second = export.inspect_nt4_export(tmp, ("Enums.h",))
            self.assertNotEqual(first, second)
            export.inspect_nt4_export.cache_clear()
            (artifact / "Enums.h").unlink()
            with self.assertRaises(FileNotFoundError):
                export.inspect_nt4_export(tmp, ("Enums.h",))

    def test_tiles_only_advertise_matched_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "expanded-config.json"
            configs = [
                {"eltwiseActType": "clampedrelu2", "tileN": n} for n in (16, 32, 128)
            ] + [
                {
                    "eltwiseActType": "none",
                    "dtypeC": "bf16",
                    "routeAct": False,
                    "tileN": n,
                }
                for n in (16, 32, 64)
            ]
            path.write_text(json.dumps({"configs": configs}))
            self.assertEqual(export.nt4_export_tiles(tmp), [16, 32])
            path.write_text(json.dumps({"configs": configs[:3]}))
            with self.assertRaisesRegex(ValueError, "matching"):
                export.nt4_export_tiles(tmp)

    def test_argument_contract_without_importing_cuda_package(self):
        tree = ast.parse((ROOT / "flashinfer/fused_moe/core.py").read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_validate_fp8_block_scale_gemm1_activation_params"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                node,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        namespace = {
            "os": os,
            "torch": SimpleNamespace(float32="float32"),
            "ActivationType": Activation,
            "Fp8QuantizationType": Quantization,
        }
        exec(compile(module, "activation_contract", "exec"), namespace)
        validate = namespace[node.name]
        scale = SimpleNamespace(
            dtype="float32", ndim=1, is_cuda=True, is_contiguous=lambda: True
        )
        with patch.dict(os.environ, {"FLASHINFER_TRTLLM_NT4_EXPORT": "/test/export"}):
            validate(1, 12, None, None, scale)
            for quant, alpha, beta, limit in (
                (2, None, None, scale),
                (1, None, None, None),
                (1, scale, None, scale),
                (1, None, scale, scale),
            ):
                with self.assertRaises(ValueError):
                    validate(quant, 12, alpha, beta, limit)
            scale.dtype = "bfloat16"
            with self.assertRaisesRegex(ValueError, "float32"):
                validate(1, 12, None, None, scale)
            scale.dtype = "float32"
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "FLASHINFER_TRTLLM_NT4_EXPORT"):
                validate(1, 12, None, None, scale)
            # Existing unparameterized activations keep their behavior.
            validate(1, 3, None, None, None)


if __name__ == "__main__":
    unittest.main()
