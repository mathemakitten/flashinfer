"""CPU routing contract for the experimental hybrid; no GPU qualification."""

import ast
from pathlib import Path
import unittest

import torch

root = Path(__file__).resolve().parents[2]
tree = ast.parse((root / "examples/moe_ep/nt4_nvls_hybrid.py").read_text())
# Load the pure tensor helper without loading distributed replay dependencies.
helper = next(
    n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "local_routes"
)
scope = {"torch": torch}
exec(compile(ast.Module(body=[helper], type_ignores=[]), "local_routes", "exec"), scope)
local_routes = scope["local_routes"]


class RoutingTest(unittest.TestCase):
    def test_each_slot_owned_once_in_original_order(self):
        ids = torch.tensor([[0, 127, 128, 255, 256, 383, 384, 511, -1, -1]])
        probs = torch.arange(10).float().view(1, 10)
        ownership = torch.zeros_like(ids)
        for rank in range(4):
            local, weights = local_routes(ids, probs, rank)
            valid = local >= 0
            ownership += valid
            torch.testing.assert_close(local[valid] + rank * 128, ids[valid])
            torch.testing.assert_close(weights[valid], probs[valid])
            self.assertTrue(bool((weights[~valid] == 0).all()))
        torch.testing.assert_close(ownership, (ids >= 0).long())

    def test_empty_owner_and_padding(self):
        ids = torch.tensor([[1, 2] + [-1] * 8, [-1] * 10])
        local, weights = local_routes(ids, torch.ones_like(ids).float(), 3)
        self.assertTrue(bool((local == -1).all()))
        self.assertTrue(bool((weights == 0).all()))


if __name__ == "__main__":
    unittest.main()
