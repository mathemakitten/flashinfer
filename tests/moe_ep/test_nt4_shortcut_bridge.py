"""CPU tests for staged shortcut dispatch; GPU stream validation is remote."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/moe_ep"))
from nt4_shortcut_bridge import install_shortcut_bridge, install_cutlass_shortcut_bridge


class BridgeTest(unittest.TestCase):
    def test_cutlass_uses_gathered_routes_and_preserves_native_prefill(self):
        calls = []
        active = [True]
        experts = SimpleNamespace(
            _uses_mxfp8_weights=False,
            _resolve_flashinfer_activation_type=lambda: "ClampedRelu2",
            _ensure_concatenated_weights=lambda: calls.append("weights"),
            _flashinfer_forward=lambda x, r, p: (
                calls.append(("cutlass", x, r, p)) or "bf16-output",
                None,
            ),
            prepare_shortcut_overlap=lambda x, p, r, **kw: ("native", x, p, r),
            compute_shortcut_overlap=lambda prepared: prepared,
            finish_shortcut_overlap=lambda computed: (computed, None),
        )
        original_prepare = experts.prepare_shortcut_overlap
        restore = install_cutlass_shortcut_bridge(experts, enabled=lambda: active[0])
        prepared = experts.prepare_shortcut_overlap(
            "gathered-x", "probs", "routes", max_blocks=304
        )
        result = experts.finish_shortcut_overlap(
            experts.compute_shortcut_overlap(prepared)
        )
        self.assertEqual(result, ("bf16-output", None))
        self.assertEqual(
            calls, ["weights", ("cutlass", "gathered-x", "routes", "probs")]
        )
        self.assertEqual(experts._flashinfer_activation_type, "ClampedRelu2")
        active[0] = False
        prepared = experts.prepare_shortcut_overlap("x", "p", "r")
        self.assertEqual(
            experts.finish_shortcut_overlap(experts.compute_shortcut_overlap(prepared)),
            (("native", "x", "p", "r"), None),
        )
        self.assertEqual(len(calls), 2)
        restore()
        self.assertIs(experts.prepare_shortcut_overlap, original_prepare)
        self.assertFalse(hasattr(experts, "_flashinfer_activation_type"))

    def test_direct_and_prefill_preserve_stage_contract(self):
        calls = []
        native_prepared, native_computed = object(), object()

        def prepare(x, p, r, *, max_blocks=0):
            calls.append(("prepare", max_blocks))
            return native_prepared

        def compute(p):
            self.assertIs(p, native_prepared)
            calls.append("compute")
            return native_computed

        def finish(c):
            self.assertIs(c, native_computed)
            calls.append("finish")
            return "native", None

        experts = SimpleNamespace(
            prepare_shortcut_overlap=prepare,
            compute_shortcut_overlap=compute,
            finish_shortcut_overlap=finish,
        )
        active = [True]

        def direct(x, p, r):
            calls.append(("direct", x, p, r))
            return "megamoe", None

        restore = install_shortcut_bridge(
            experts, enabled=lambda: active[0], execute=direct
        )
        p = experts.prepare_shortcut_overlap("x", "p", "r", max_blocks=304)
        c = experts.compute_shortcut_overlap(p)
        self.assertEqual(experts.finish_shortcut_overlap(c), ("megamoe", None))
        self.assertEqual(calls, [("direct", "x", "p", "r")])
        active[0] = False
        p = experts.prepare_shortcut_overlap("x", "p", "r", max_blocks=304)
        c = experts.compute_shortcut_overlap(p)
        self.assertEqual(experts.finish_shortcut_overlap(c), ("native", None))
        self.assertEqual(calls[1:], [("prepare", 304), "compute", "finish"])
        restore()
        self.assertIs(experts.prepare_shortcut_overlap, prepare)
        self.assertIs(experts.compute_shortcut_overlap, compute)
        self.assertIs(experts.finish_shortcut_overlap, finish)


if __name__ == "__main__":
    unittest.main()
