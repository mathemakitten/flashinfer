"""CPU regression for epilogue construction order (not a CuTe compile test)."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = (
    Path(__file__).resolve().parents[2]
    / "flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/shim/nt4"
)


class DebugLifecycleTest(unittest.TestCase):
    def test_debug_configuration_follows_each_epilogue_creation(self):
        tree = ast.parse((ROOT / "kernel_mxfp8_glu_fc12.py").read_text())
        setup = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_setup_attributes"
        )
        creation = next(
            i
            for i, n in enumerate(setup.body)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "epilogue" for t in n.targets
            )
        )
        # Execute the real construction/configuration statements with an inert
        # epilogue; the rest of setup requires CUDA/CuTe, unavailable on this host.
        code = compile(
            ast.Module(body=setup.body[creation : creation + 2], type_ignores=[]),
            str(ROOT),
            "exec",
        )
        for enabled in (False, True):
            kernel = SimpleNamespace(
                debug_prequant=enabled,
                _local_offsets={"debug_prequant": 8192, "fc1_output": 1024},
                _local_region_by_name={
                    "fc1_output": SimpleNamespace(shape=(128, 4096))
                },
            )
            env = {
                "self": kernel,
                "_epi_common": {},
                "GluMxfp8Epilogue": SimpleNamespace,
            }
            for _ in range(2):
                exec(code, env)
                if enabled:
                    self.assertEqual(kernel.epilogue._debug_prequant_offset, 8192)
                    self.assertEqual(kernel.epilogue._debug_prequant_width, 4096)
                    self.assertEqual(kernel.epilogue._debug_prequant_rows, 128)
                else:
                    self.assertFalse(hasattr(kernel.epilogue, "_debug_prequant_offset"))
        tree = ast.parse((ROOT / "megamoe_kernel_mxfp8.py").read_text())
        constructor = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        self.assertFalse(
            any(
                isinstance(n, ast.Attribute) and n.attr == "epilogue"
                for n in ast.walk(constructor)
            )
        )


if __name__ == "__main__":
    unittest.main()
