"""CPU check of ordered explicit teardown; does not exercise NVSHMEM."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2] / "examples/moe_ep"


class CleanupTest(unittest.TestCase):
    def test_workspace_release_precedes_final_barrier(self):
        tree = ast.parse((ROOT / "nt4_e2e_adapter.py").read_text())
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "close"
        )
        events = []
        ns = dict(
            torch=SimpleNamespace(
                cuda=SimpleNamespace(synchronize=lambda: events.append("sync"))
            ),
            dist=SimpleNamespace(
                barrier=lambda: events.append("barrier"), get_rank=lambda: 0
            ),
            json=SimpleNamespace(dumps=lambda value: ""),
            print=lambda *args, **kwargs: None,
        )
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), "<cleanup>", "exec"), ns
        )
        obj = SimpleNamespace(
            restorers=[lambda: events.append("restore")],
            layers={
                1: (9, SimpleNamespace(close=lambda: events.append("close9"))),
                2: (7, SimpleNamespace(close=lambda: events.append("close7"))),
            },
        )
        ns["close"](obj)
        self.assertEqual(
            events,
            ["sync", "barrier", "restore", "close7", "close9", "sync", "barrier"],
        )
        self.assertEqual(obj.layers, {})


class GraphCleanupTest(unittest.TestCase):
    def test_graphs_released_before_symmetric_workspace(self):
        tree = ast.parse((ROOT / "nt4_e2e_graph.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "GraphResources"
        )
        ns = {}
        exec(
            compile(ast.Module(body=[cls], type_ignores=[]), "<graph_cleanup>", "exec"),
            ns,
        )
        events = []
        resource = ns["GraphResources"](
            SimpleNamespace(close=lambda: events.append("workspace"))
        )
        resource.buckets["shape"] = {
            "graph": SimpleNamespace(reset=lambda: events.append("graph"))
        }
        resource.close()
        self.assertEqual(events, ["graph", "workspace"])
        self.assertEqual(resource.buckets, {})


if __name__ == "__main__":
    unittest.main()
