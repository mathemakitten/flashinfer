"""CPU tests of stream ownership; no FlashInfer/TE import or GPU required."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[2] / "examples/moe_ep/nt4_te_workspace.py"
spec = importlib.util.spec_from_file_location("nt4_te_workspace", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestStreamWorkspaces(unittest.TestCase):
    def test_streams_have_independent_scratch_and_reuse_their_own(self):
        current = {0: 10, 1: 10}
        pool = module.StreamWorkspaces(
            lambda *args: object(),
            lambda device: SimpleNamespace(cuda_stream=current[device]),
        )
        first = pool(0, False, False)
        self.assertIs(first, pool(0, False, False))
        current[0] = 11
        second = pool(0, False, False)
        self.assertIsNot(first, second)
        current[0] = 10
        self.assertIs(first, pool(0, False, False))
        self.assertIsNot(first, pool(1, False, False))
        self.assertIsNot(first, pool(0, True, False))

    def test_grouped_workspace_lists_are_not_shared_between_streams(self):
        current = [10]
        pool = module.StreamWorkspaces(
            lambda *args: [object(), object()],
            lambda device: SimpleNamespace(cuda_stream=current[0]),
        )
        first = pool(0, False, True)
        current[0] = 11
        second = pool(0, False, True)
        self.assertTrue(all(a is not b for a, b in zip(first, second)))
        self.assertEqual(sum(row["calls"] for row in pool.report()), 2)


if __name__ == "__main__":
    unittest.main()
