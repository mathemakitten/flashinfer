"""CPU-only graph/symmetric-workspace destruction ordering regression."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class ResourcesTest(unittest.TestCase):
    def test_graphs_destroyed_before_any_hybrid_workspace(self):
        path = Path(__file__).resolve().parents[2] / "examples/moe_ep/nt4_e2e_graph.py"
        tree = ast.parse(path.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "GraphResources"
        )
        namespace = {}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        events = []
        resources = namespace["GraphResources"](
            SimpleNamespace(close=lambda: events.append("direct"))
        )
        for i in range(2):
            resources.buckets[i] = dict(
                graph=SimpleNamespace(reset=lambda i=i: events.append(f"graph{i}")),
                hybrid=SimpleNamespace(close=lambda i=i: events.append(f"hybrid{i}")),
            )
        resources.close()
        self.assertEqual(events, ["graph0", "graph1", "hybrid0", "hybrid1", "direct"])
        self.assertEqual(resources.buckets, {})


if __name__ == "__main__":
    unittest.main()
