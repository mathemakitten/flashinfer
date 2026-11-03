"""CPU-only validation of matched-input branch reports; runnable with unittest."""

import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/moe_ep/nt4_branch_backend_summary.py"
)
spec = importlib.util.spec_from_file_location("branch_backend_summary", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for arm in module.ARMS:
            backend, schedule = arm.split("-")
            folder = self.root / arm / backend
            folder.mkdir(parents=True)
            for rank in range(4):
                rows = []
                for round_ in (1, 2):
                    for sample in range(16):
                        for layer in (7, 32, 68):
                            rows.append(
                                dict(
                                    rank=rank,
                                    round=round_,
                                    sample=sample,
                                    layer=layer,
                                    schedule=schedule,
                                    clock="device",
                                    input_mode=(
                                        "record" if arm == module.ARMS[0] else "replay"
                                    ),
                                    input_sha256=f"{rank}-{round_}-{sample}-{layer}",
                                    decode_step=sample,
                                    active_requests=128,
                                    join_us=110,
                                    phase_offsets_us={
                                        "experts_combine": [10, 100],
                                        "mixer_post_shared": [20, 80],
                                    },
                                    phase_us={
                                        "experts_combine": 90,
                                        "mixer_post_shared": 60,
                                    },
                                    tail_phase_us={"shared_fc1": 10},
                                )
                            )
                (folder / f"branch-rank{rank}.jsonl").write_text(
                    "\n".join(map(json.dumps, rows))
                )

    def test_mxfp8_uses_cutlass_baseline(self):
        for schedule in ("overlap", "serial"):
            old = self.root / f"vllm-{schedule}"
            new = self.root / f"baseline-{schedule}"
            old.rename(new)
            (new / "vllm").rename(new / "baseline")
        with contextlib.redirect_stdout(io.StringIO()):
            result = module.summarize(self.root, "mxfp8")
        self.assertEqual(result["precision"], "mxfp8")
        self.assertEqual(result["native_backend"], "FlashInfer CUTLASS + NVLS")
        self.assertTrue(result["valid_matched_block_inputs"])
        self.assertEqual(len(result["results"]), 48)

    def test_hybrid_sweep_and_provenance(self):
        for schedule in ("serial", "overlap"):
            baseline = self.root / f"baseline-{schedule}"
            shutil.copytree(self.root / f"vllm-{schedule}", baseline)
            (baseline / "vllm").rename(baseline / "baseline")
            shutil.copytree(
                self.root / f"direct-{schedule}", self.root / f"hybrid-{schedule}"
            )
            for backend in ("baseline", "direct", "hybrid"):
                (self.root / f"{backend}-{schedule}" / "summary.json").write_text(
                    json.dumps(
                        dict(
                            precision="mxfp8",
                            mxfp8_nvls_hybrid=backend == "hybrid",
                            branch_backend=(
                                "baseline" if backend == "baseline" else "direct"
                            ),
                            mxfp8_worker_clusters=None,
                            mxfp8_ab_stage_reduction=None,
                            cutlass_decode_profile_ids=(
                                [39, 428] if backend == "baseline" else None
                            ),
                            model_graphs=True,
                            shortcut_overlap=True,
                        )
                    )
                )
        with contextlib.redirect_stdout(io.StringIO()):
            result = module.summarize(self.root, "mxfp8", hybrid_sweep=True)
        self.assertEqual(len(result["results"]), 72)
        self.assertTrue(result["hybrid_sweep"])
        path = self.root / "hybrid-overlap/summary.json"
        metadata = json.loads(path.read_text())
        metadata["mxfp8_nvls_hybrid"] = False
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "incorrect hybrid"):
            module.summarize(self.root, "mxfp8", hybrid_sweep=True)

    def test_worker_sweep_and_provenance(self):
        for schedule in ("serial", "overlap"):
            folder = self.root / f"baseline-{schedule}"
            shutil.copytree(self.root / f"vllm-{schedule}", folder)
            (folder / "vllm").rename(folder / "baseline")
            (folder / "summary.json").write_text(
                json.dumps(
                    dict(
                        precision="mxfp8",
                        mxfp8_worker_clusters=None,
                        cutlass_decode_profile_ids=[39, 428],
                    )
                )
            )
            for cap in (152, 144, 136):
                folder = self.root / f"direct{cap}-{schedule}"
                shutil.copytree(self.root / f"direct-{schedule}", folder)
                (folder / "summary.json").write_text(
                    json.dumps(
                        dict(
                            precision="mxfp8",
                            mxfp8_worker_clusters=cap,
                            cutlass_decode_profile_ids=None,
                        )
                    )
                )
        with contextlib.redirect_stdout(io.StringIO()):
            result = module.summarize(self.root, "mxfp8", True)
        self.assertEqual(len(result["results"]), 96)
        p = self.root / "direct144-overlap/summary.json"
        p.write_text(p.read_text().replace("144", "152"))
        with self.assertRaisesRegex(ValueError, "incorrect worker"):
            module.summarize(self.root, "mxfp8", True)

    def test_pipeline_sweep_and_provenance(self):
        for schedule in ("serial", "overlap"):
            folder = self.root / f"baseline-{schedule}"
            shutil.copytree(self.root / f"vllm-{schedule}", folder)
            (folder / "vllm").rename(folder / "baseline")
            (folder / "summary.json").write_text(
                json.dumps(
                    dict(
                        precision="mxfp8",
                        mxfp8_worker_clusters=None,
                        mxfp8_ab_stage_reduction=None,
                        cutlass_decode_profile_ids=[39, 428],
                    )
                )
            )
            for reduction in (0, 1, 2):
                folder = self.root / f"directstage{reduction}-{schedule}"
                shutil.copytree(self.root / f"direct-{schedule}", folder)
                (folder / "summary.json").write_text(
                    json.dumps(
                        dict(
                            precision="mxfp8",
                            mxfp8_worker_clusters=152,
                            mxfp8_ab_stage_reduction=reduction,
                            cutlass_decode_profile_ids=None,
                        )
                    )
                )
        with contextlib.redirect_stdout(io.StringIO()):
            result = module.summarize(self.root, "mxfp8", pipeline_sweep=True)
        self.assertEqual(len(result["results"]), 96)
        p = self.root / "directstage1-overlap/summary.json"
        p.write_text(p.read_text().replace("152", "144"))
        with self.assertRaisesRegex(ValueError, "incorrect pipeline"):
            module.summarize(self.root, "mxfp8", pipeline_sweep=True)

    def test_matching_and_completion_times(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = module.summarize(self.root)
        self.assertTrue(result["valid_matched_block_inputs"])
        self.assertEqual(len(result["results"]), 48)
        self.assertEqual(result["results"][0]["experts_after_tail_us"], 20)
        self.assertEqual(result["results"][0]["tail_after_experts_us"], 0)

    def test_reject_mismatched_inputs(self):
        path = self.root / "direct-overlap/direct/branch-rank0.jsonl"
        path.write_text(
            path.read_text().replace(
                '"input_sha256": "0-1-0-7"', '"input_sha256": "wrong"'
            )
        )
        with self.assertRaisesRegex(ValueError, "unmatched inputs"):
            module.summarize(self.root)

    def test_reject_missing_rank_samples(self):
        path = self.root / "direct-overlap/direct/branch-rank0.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[1:]))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            module.summarize(self.root)


if __name__ == "__main__":
    unittest.main()
