"""CPU-only restart tests; no model/GPU/import of FlashInfer required."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

source = Path(__file__).resolve().parents[2] / "examples/moe_ep/nt4_e2e_resume.py"
spec = importlib.util.spec_from_file_location("resume", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "manifest.json").write_text("{}")
        (self.root / "prompts").write_text("prompt")
        self.out = self.root / "out"
        self.out.mkdir()
        self.args = SimpleNamespace(
            output=self.out,
            source_artifact=self.root,
            prompt_file=self.root / "prompts",
            resume=True,
            resume_legacy=True,
            timeout_seconds=100,
            rounds=2,
            precision="mxfp8",
            model_graphs=True,
            shortcut_overlap=True,
            request_count=2,
            num_tokens_to_generate=3,
        )
        self.command = ["python", "model.py"]

    def runner(self):
        r = m.ResumeRun(self.args, source.parents[2])
        self.addCleanup(r.lock.close)
        return r

    def complete(self, name, speed=100):
        d = self.out / name
        d.mkdir(exist_ok=True)
        rows = []
        for i in (1, 2):
            rows.append(
                dict(
                    round=i,
                    warmup=False,
                    precision="mxfp8",
                    model_graphs=True,
                    shortcut_overlap=True,
                    model_graph_vs_eager_outputs="EXACT",
                    requested_batch_reached=True,
                    peak_decode_active_requests=2,
                    total_output_tokens=6,
                    output_tokens_per_second=speed,
                    gc_policy={"policy": "collect_before_round_disable_during_round"},
                )
            )
            m.write_json(d / f"outputs-round{i}.json", [[1, 2, 3], [1, 2, 3]])
        m.write_json(d / "results.json", rows)
        m.write_json(self.out / f"{name}-command.json", self.command)
        (self.out / f"{name}.log").write_text('{"stage": "e2e_cleanup_complete"}\n' * 4)

    def test_legacy_needs_explicit_acknowledgment(self):
        self.args.resume_legacy = False
        with self.assertRaisesRegex(RuntimeError, "no resume provenance"):
            self.runner()

    def test_reuse_success_archive_failure_refresh_control(self):
        self.complete("baseline-start")
        self.complete("hybrid")
        m.write_json(self.out / "hybrid-exit.json", {"returncode": 1})
        self.complete("baseline-end")
        r = self.runner()
        self.assertFalse(r.prepare("baseline-start", self.command))
        self.assertTrue(r.prepare("hybrid", self.command))
        self.assertTrue((r.archive / "hybrid/results.json").exists())
        self.assertFalse((self.out / "hybrid").exists())
        self.assertTrue(r.prepare("baseline-end", self.command))
        self.assertTrue((r.archive / "baseline-end.log").exists())

    def test_partial_crash_and_changed_command_never_reused(self):
        self.complete("hybrid")
        (self.out / "hybrid/outputs-round2.json").unlink()
        r = self.runner()
        self.assertFalse(r.reusable("hybrid", self.command))
        self.complete("hybrid")
        with (self.out / "hybrid.log").open("a") as f:
            f.write("Segfault encountered")
        self.assertFalse(r.reusable("hybrid", self.command))
        with self.assertRaisesRegex(RuntimeError, "command differs"):
            r.reusable("hybrid", ["changed"])

    def test_configuration_and_lock_guards(self):
        r = self.runner()
        with self.assertRaisesRegex(RuntimeError, "owns this output"):
            self.runner()
        r.lock.close()
        self.args.request_count = 4
        with self.assertRaisesRegex(RuntimeError, "configuration differs"):
            self.runner()

    def test_failed_correctness_and_bad_measurements(self):
        self.complete("hybrid")
        r = self.runner()
        for key, value in [
            ("model_graph_vs_eager_outputs", "MISMATCH"),
            ("requested_batch_reached", False),
            ("output_tokens_per_second", float("nan")),
        ]:
            self.complete("hybrid")
            p = self.out / "hybrid/results.json"
            rows = json.loads(p.read_text())
            rows[0][key] = value
            m.write_json(p, rows)
            self.assertFalse(r.reusable("hybrid", self.command))

    def test_control_drift_blocks_comparison(self):
        self.complete("baseline-start", 100)
        self.complete("baseline-end", 110)
        r = self.runner()
        self.assertFalse(r.finish()["comparison_valid"])
        self.complete("baseline-end", 101)
        self.assertTrue(r.finish()["comparison_valid"])
        r.finished("trtllm-clamped", 124)
        self.assertFalse(r.finish()["comparison_valid"])


if __name__ == "__main__":
    unittest.main()
