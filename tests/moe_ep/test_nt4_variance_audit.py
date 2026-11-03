"""CPU checks for diagnostic output and cleanup; no GPU timing claims."""

import json
import gc
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/moe_ep"))
from nt4_variance_audit import FirstForwardAudit, VarianceAudit, sha256


class AuditTest(unittest.TestCase):
    def test_gc_records_after_first_forward_and_removes_callback(self):
        previous = list(gc.callbacks)
        audit = FirstForwardAudit()
        try:
            audit.begin()
            audit.finish()
            gc.collect(2)
            self.assertTrue(any(r["generation"] == 2 for r in audit.gc_records))
            self.assertTrue(all(r["stage"] == "python_gc" for r in audit.gc_records))
            audit.gc_active = False
            count = len(audit.gc_records)
            gc.collect(2)
            self.assertEqual(len(audit.gc_records), count)
        finally:
            audit.close()
        self.assertEqual(gc.callbacks, previous)

    def test_constructor_hook_preserves_class_and_nested_calls(self):
        class Request:
            def __init__(self, value):
                self.value = value
                self.__post_init__()

            def __post_init__(self):
                self.value += 1

        original = Request.__init__
        audit = FirstForwardAudit()
        try:
            audit.install(Request, "__init__", "request.construct")
            audit.install(Request, "__post_init__", "request.post_init")
            audit.begin()
            request = Request(4)
            audit.finish()
            self.assertIsInstance(request, Request)
            self.assertEqual(request.value, 5)
            self.assertEqual(
                [r["stage"] for r in audit.records[:2]],
                ["request.post_init", "request.construct"],
            )
        finally:
            audit.close()
        self.assertIs(Request.__init__, original)

    def test_first_forward_scope_and_restore(self):
        class Context:
            def prepare(self, value):
                return value + 1

        context = Context()
        audit = FirstForwardAudit()
        audit.install(context, "prepare", "context.prepare")
        self.assertEqual(context.prepare(1), 2)
        self.assertEqual(audit.records, [])
        audit.begin()
        self.assertEqual(context.prepare(2), 3)
        audit.finish()
        self.assertEqual(context.prepare(3), 4)
        self.assertEqual(
            [r["stage"] for r in audit.records],
            ["context.prepare", "generation_to_first_forward"],
        )
        for span in audit.records:
            self.assertGreaterEqual(span["end"]["wall_ns"], span["start"]["wall_ns"])
            self.assertIn("thread_cpu_ns", span["end"])
        audit.close()
        self.assertNotIn("prepare", vars(context))
        self.assertEqual(context.prepare(4), 5)

    def test_first_forward_records_exception_and_preserves_override(self):
        def fail():
            raise ValueError("test")

        context = NS(prepare=fail)
        audit = FirstForwardAudit()
        audit.install(context, "prepare", "context.prepare")
        audit.begin()
        with self.assertRaisesRegex(ValueError, "test"):
            context.prepare()
        self.assertEqual(len(audit.records), 1)
        audit.close()
        self.assertIs(context.prepare, fail)

    def test_report_preserves_rank_phase_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = VarianceAudit.__new__(VarianceAudit)
            audit.destination = Path(directory)
            audit.rank = 2
            audit.index = 3
            audit.active = True
            audit.telemetry = None
            audit.telemetry_file = None
            audit.records = [
                dict(decode=True, requests=128, host_start_ns=10, host_end_ns=20)
            ]
            end = object()
            start = NS(elapsed_time=lambda e: 1.25 if e is end else None)
            audit.pool = [(start, end)]
            audit.configs = {
                json.dumps(dict(config={"BLOCK_SIZE_M": 64}, capturing=True)): 28
            }
            audit.end()
            report = json.loads(
                (Path(directory) / "audit-round3-rank2.json").read_text()
            )
            self.assertEqual(report["forward_intervals"][0]["gpu_elapsed_ms"], 1.25)
            self.assertTrue(report["forward_intervals"][0]["decode"])
            self.assertEqual(report["launch_configs"][0]["observations"], 28)
            self.assertFalse(audit.active)

    def test_close_stops_sampler_and_restores_hooks(self):
        audit = VarianceAudit.__new__(VarianceAudit)
        process, output = Mock(), Mock()
        audit.telemetry = process
        audit.telemetry_file = output
        audit.experts = NS(prepare_vllm_moe=None)
        audit.engine = NS(controller=NS(_dynamic_step_forward_logits=None))
        audit.original_prepare = object()
        audit.original_forward = object()
        audit.close()
        process.terminate.assert_called_once()
        output.close.assert_called_once()
        self.assertIs(audit.experts.prepare_vllm_moe, audit.original_prepare)
        self.assertIs(
            audit.engine.controller._dynamic_step_forward_logits, audit.original_forward
        )

    def test_input_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "prompts"
            p.write_text("a")
            first = sha256(p)
            p.write_text("b")
            self.assertNotEqual(first, sha256(p))


if __name__ == "__main__":
    unittest.main()
