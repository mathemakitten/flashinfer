"""CPU lifecycle checks for the NT4 whole-model graph audit."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import contextlib
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/moe_ep"))
from nt4_model_graph import ModelGraphSession


class ModelGraphTest(unittest.TestCase):
    def make_session(self):
        runner = SimpleNamespace(
            fwd_graph_recorded=True, replay_graph_capture=lambda: 42
        )
        manager = SimpleNamespace(
            cudagraph_runners=[], get_cudagraph_runner=lambda: runner
        )
        model = SimpleNamespace(
            cudagraph_manager=manager, _should_call_local_cudagraph=lambda: True
        )
        context = SimpleNamespace(
            is_decode_only=lambda: True, get_active_request_count=lambda: 128
        )

        def forward():
            if model._should_call_local_cudagraph():
                return manager.get_cudagraph_runner().replay_graph_capture()
            return 42

        controller = SimpleNamespace(
            _language_model=model, _dynamic_step_forward_logits=forward
        )
        engine = SimpleNamespace(controller=controller, context=context)
        adapter = SimpleNamespace(decode_occupancy={})
        session = ModelGraphSession(engine, adapter)
        return session, runner, controller, adapter

    def test_mixed_native_and_experimental_graph_coverage(self):
        for bridge_layers, captured_layers, valid in (
            ({7}, {7, 9}, True),
            (set(), {7, 9}, False),
            ({7}, {7}, False),
        ):
            session, _, _, adapter = self.make_session()
            adapter.layers = {
                1: (7, SimpleNamespace(replay=object())),
                2: (9, SimpleNamespace(replay=None)),
            }
            adapter.shortcut_overlap = True
            adapter.mode = "direct"
            adapter.shortcut_capture_layers = bridge_layers
            adapter.model_capture_layers = {128: captured_layers}
            session.engine.create_cuda_graphs = (
                lambda: session.manager.cudagraph_runners.append(object())
            )
            session.engine.reset = lambda: None
            dist = SimpleNamespace(barrier=lambda: None, get_rank=lambda: 0)
            torch = SimpleNamespace(
                distributed=dist, cuda=SimpleNamespace(synchronize=lambda: None)
            )
            with (
                patch.dict(
                    sys.modules,
                    {
                        "torch": torch,
                        "torch.distributed": dist,
                        "nt4_capture_state": SimpleNamespace(
                            set_context=lambda ctx: None
                        ),
                    },
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if valid:
                    session.build()
                    self.assertFalse(adapter.allow_setup)
                else:
                    with self.assertRaises(RuntimeError):
                        session.build()

    def test_eager_reference_then_actual_replay(self):
        session, _, controller, adapter = self.make_session()
        self.assertEqual(controller._dynamic_step_forward_logits(), 42)
        self.assertEqual(session.replays, 0)
        session.force_eager = False
        session.begin_round(True)
        self.assertEqual(controller._dynamic_step_forward_logits(), 42)
        self.assertEqual(session.replays, 1)
        self.assertEqual(session.decode_steps, 1)
        self.assertEqual(adapter.decode_occupancy, {128: 2})

    def test_eager_fallback_is_rejected(self):
        session, _, controller, _ = self.make_session()
        session.begin_round(True)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            controller._dynamic_step_forward_logits()

    def test_new_capture_in_measured_round_is_rejected(self):
        session, runner, controller, _ = self.make_session()
        runner.fwd_graph_recorded = False
        session.force_eager = False
        session.begin_round(True)
        with self.assertRaisesRegex(RuntimeError, "new model graph"):
            controller._dynamic_step_forward_logits()

    def test_prefill_does_not_require_a_graph(self):
        session, _, controller, _ = self.make_session()
        session.engine.context.is_decode_only = lambda: False
        session.begin_round(True)
        self.assertEqual(controller._dynamic_step_forward_logits(), 42)
        self.assertEqual(session.decode_steps, 0)


if __name__ == "__main__":
    unittest.main()
