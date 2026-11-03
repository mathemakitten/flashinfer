"""CPU regression checks for the BF16 port's weight-layout and configuration API."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "flashinfer/moe_ep/backends/mega/kernel/sm100/bf16_bf16_bf16_cutedsl"


def functions(path):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(MoEWeightPack=SimpleNamespace, TransformedMegaWeights=tuple, torch=torch)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


class Bf16PortTest(unittest.TestCase):
    def test_nt4_weights_keep_every_channel_in_order(self):
        ns = functions(BACKEND / "weights.py")
        w1 = torch.arange(2 * 256 * 32).reshape(2, 256, 32).bfloat16()
        w2 = torch.arange(2 * 32 * 256).reshape(2, 32, 256).bfloat16()
        result = ns["preprocess_mega_weights"](
            SimpleNamespace(w13=w1, w2=w2),
            intermediate_size=256,
            hidden_size=32,
            activation="clamped_relu2",
        )
        torch.testing.assert_close(result[0][0].transpose(1, 2), w1, rtol=0, atol=0)
        self.assertEqual(result[0][0].stride(1), 1)
        ns["validate_transformed_mega_weights"](
            result,
            intermediate_size=256,
            hidden_size=32,
            activation="clamped_relu2",
            world_size=1,
            num_experts=2,
        )
        with self.assertRaises(ValueError):
            ns["validate_transformed_mega_weights"](
                result,
                intermediate_size=256,
                hidden_size=32,
                world_size=1,
                num_experts=2,
            )

    def test_swiglu_default_still_interleaves(self):
        ns = functions(BACKEND / "weights.py")
        w1 = torch.cat(
            [torch.ones(1, 64, 32), torch.full((1, 64, 32), 2.0)], dim=1
        ).bfloat16()
        result = ns["preprocess_mega_weights"](
            SimpleNamespace(w13=w1, w2=torch.zeros(1, 32, 64, dtype=torch.bfloat16)),
            intermediate_size=64,
            hidden_size=32,
        )
        rows = result[0][0].transpose(1, 2)[0, :, 0]
        self.assertEqual(
            rows.tolist(), [1.0] * 32 + [2.0] * 32 + [1.0] * 32 + [2.0] * 32
        )

    def test_invalid_slots_cannot_leak_nan_into_recombine(self):
        tree = ast.parse((ROOT / "examples/moe_ep/nt4_bf16_replay.py").read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "recombine"
        )
        ns = dict(torch=torch)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<reference>", "exec"), ns)
        terms = torch.tensor([[[2.0, 4.0], [float("nan"), float("nan")]]])
        result = ns["recombine"](
            terms, torch.tensor([[4, -1]]), torch.tensor([[0.5, 1.0]])
        )
        torch.testing.assert_close(
            result, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
