"""CPU checks for diagnostic timing boundary and native continuation ownership."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2] / "examples/moe_ep"
sys.path.insert(0, str(ROOT))
from nt4_branch_native_output import install_native_output


class Tensor:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return Tensor(self.value)

    def record_stream(self, stream):
        pass


class NVLSAllGatherVDispatcher:
    pass


class IsolationTest(unittest.TestCase):
    def setUp(self):
        self.trace = []
        trace = self.trace

        class Mlp:
            _nt4_e2e_active = True

            def dispatch(self, x, probs):
                trace.append(
                    (
                        "dispatch",
                        x.value,
                        probs.value,
                        self.token_dispatcher.routing_map.value,
                    )
                )
                return x, probs

            def routed_experts_compute(self, x, probs):
                trace.append("native_clamped")
                return Tensor(x.value + probs.value), None

            def combine(self, out):
                trace.append("combine")
                return out

        self.mlp = Mlp()
        self.mlp.token_dispatcher = NVLSAllGatherVDispatcher()
        self.x, self.probs, self.routes = Tensor(10), Tensor(2), Tensor(7)

        def route():
            self.mlp.token_dispatcher.routing_map = self.routes
            return self.x, self.probs

        def forbidden(*args):
            raise AssertionError(
                "patched measured backend must not run in continuation"
            )

        self.mlp.dispatch = self.mlp.routed_experts_compute = self.mlp.combine = (
            forbidden
        )
        self.block = SimpleNamespace(
            moe_layer=SimpleNamespace(mlp=self.mlp),
            _moe_router_preprocess=route,
            _postprocess=lambda **kw: kw["combined_output"],
        )
        self.probe = SimpleNamespace()
        install_native_output(self.block, self.probe)

    def join(self):
        with patch.dict(
            sys.modules,
            {"torch": SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: 1))},
        ):
            return self.probe.after_join()

    def test_install_before_inference_dispatcher_selection(self):
        # Reinstall on a fresh block whose initial dispatcher is the training
        # implementation. MCore selects NVLS before invoking preprocess.
        native = self.mlp.token_dispatcher
        training = SimpleNamespace()
        self.mlp.token_dispatcher = training

        def route():
            self.mlp.token_dispatcher = native
            native.routing_map = self.routes
            return self.x, self.probs

        block = SimpleNamespace(
            moe_layer=SimpleNamespace(mlp=self.mlp),
            _moe_router_preprocess=route,
            _postprocess=lambda **kw: kw["combined_output"],
        )
        install_native_output(block, self.probe)
        self.assertIs(self.mlp.token_dispatcher, training)
        block._moe_router_preprocess()
        self.assertEqual(self.join().value, 12)
        self.assertEqual(block._postprocess(combined_output=Tensor(-999)).value, 12)

    def test_wrong_active_dispatcher_is_rejected_at_decode(self):
        self.mlp.token_dispatcher = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "active NVLS during decode"):
            self.block._moe_router_preprocess()

    def test_dispatcher_cannot_change_after_route(self):
        self.block._moe_router_preprocess()
        self.mlp.token_dispatcher = NVLSAllGatherVDispatcher()
        with self.assertRaisesRegex(RuntimeError, "dispatcher changed"):
            self.join()

    def test_saved_local_inputs_and_native_output_survive_measured_mutation(self):
        self.block._moe_router_preprocess()
        self.x.value = 999
        self.probs.value = 999
        self.routes.value = -1
        self.mlp.token_dispatcher.routing_map = Tensor(999)
        native = self.join()
        result = self.block._postprocess(combined_output=Tensor(-999))
        self.assertIs(result, native)
        self.assertEqual(result.value, 12)
        self.assertEqual(
            self.trace, [("dispatch", 10, 2, 7), "native_clamped", "combine"]
        )

    def test_new_route_requires_new_join_and_prefill_is_untouched(self):
        self.block._moe_router_preprocess()
        self.join()
        self.block._moe_router_preprocess()
        with self.assertRaisesRegex(RuntimeError, "timed join"):
            self.block._postprocess(combined_output=Tensor(0))
        self.mlp._nt4_e2e_active = False
        self.block._moe_router_preprocess()
        original = Tensor(123)
        self.assertIs(self.block._postprocess(combined_output=original), original)
        with self.assertRaisesRegex(RuntimeError, "no saved"):
            self.join()

    def test_native_capture_uses_local_bucket_not_gather_capacity(self):
        tree = ast.parse((ROOT / "nt4_e2e_graph.py").read_text())
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "observe_prepare"
        )

        class Dispatcher:
            _per_rank_worst_case_token_count = 2048
            _local_tokens = 128

        adapter = SimpleNamespace(model_capture_layers={})
        layer = SimpleNamespace(token_dispatcher=Dispatcher(), layer_number=9)
        ns = dict(
            self=adapter,
            layer=layer,
            isolate=True,
            torch=SimpleNamespace(
                cuda=SimpleNamespace(is_current_stream_capturing=lambda: True)
            ),
            original_prepare=lambda *args, **kwargs: "prepared",
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), "observe", "exec"), ns)
        for rows in (128, 64):
            layer.token_dispatcher._local_tokens = rows
            self.assertEqual(ns["observe_prepare"](object(), None, None), "prepared")
        self.assertEqual(adapter.model_capture_layers, {128: {9}, 64: {9}})
        ns["isolate"] = False
        ns["observe_prepare"](object(), None, None)
        self.assertEqual(adapter.model_capture_layers[2048], {9})

    def test_probe_stamps_join_before_native_rerun(self):
        # Execute the real method without importing GPU-only probe dependencies.
        tree = ast.parse((ROOT / "nt4_branch_probe.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "BranchProbe"
        )
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "join"
        )
        ns = {"PHASES": ("compute",)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "join", "exec"), ns)
        trace = []
        probe = SimpleNamespace(
            tail_enabled=False,
            seen={"compute"},
            clock="device",
            timestamps=None,
            stamp=lambda buf, slot: trace.append(("stamp", slot)),
            after_join=lambda: trace.append("native") or 42,
            native_outputs={},
            current_tag=0,
        )
        ns["join"](probe)
        self.assertEqual(trace, [("stamp", 13), "native"])
        self.assertEqual(probe.native_outputs, {0: 42})


if __name__ == "__main__":
    unittest.main()
