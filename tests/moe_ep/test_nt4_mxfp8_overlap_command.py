"""Exercise complete E2E command generation without loading a model or CUDA."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/moe_ep"))
import nt4_e2e


class OverlapCommandTest(unittest.TestCase):
    def test_hybrid_alternating_order(self):
        self.assertEqual(
            nt4_e2e.hybrid_overlap_trials(3),
            [
                ("baseline-start", "baseline"),
                ("00-hybrid", "direct"),
                ("01-cutlass", "baseline"),
                ("01-hybrid", "direct"),
                ("02-cutlass", "baseline"),
                ("02-hybrid", "direct"),
            ],
        )

    def test_mxfp8_and_bf16_select_distinct_native_baselines(self):
        scenarios = [("mxfp8", None, None, None), ("bf16", None, None, None)]
        scenarios += [
            ("mxfp8", backend, schedule, None)
            for backend in ("baseline", "direct")
            for schedule in ("serial", "overlap")
        ]
        scenarios += [
            ("mxfp8", backend, schedule, "workers")
            for backend in ("baseline", "direct")
            for schedule in ("serial", "overlap")
        ]
        scenarios += [
            ("mxfp8", "direct", schedule, "pipeline")
            for schedule in ("serial", "overlap")
        ]
        scenarios += [("mxfp8", "direct", schedule, "hybrid_pipeline") for schedule in ("serial", "overlap")]
        scenarios.append(("mxfp8", None, None, "confirm"))
        scenarios += [
            ("mxfp8", "direct", schedule, "hybrid")
            for schedule in ("serial", "overlap")
        ]
        scenarios.append(("mxfp8", None, None, "hybrid_confirm"))
        scenarios += [
            ("mxfp8", "direct", schedule, "surrogate")
            for schedule in ("serial", "overlap")
        ]
        for precision, branch_backend, schedule, experiment in scenarios:
            with (
                self.subTest(
                    precision=precision, backend=branch_backend, schedule=schedule
                ),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = root / "source"
                source.mkdir()
                (source / "manifest.json").write_text(
                    json.dumps(dict(capture_complete=True, world_size=4))
                )
                mcore = root / "mcore"
                arguments = mcore / "megatron/training/arguments.py"
                arguments.parent.mkdir(parents=True)
                arguments.write_text("--inference-shortcut-moe-overlap")
                experts = mcore / "megatron/core/transformer/moe/experts.py"
                experts.parent.mkdir(parents=True)
                experts.write_text(
                    "class PreparedFlashInferMoE:\n    def prepare_shortcut_overlap(self): pass\n"
                )
                shortcut = mcore / "megatron/core/models/hybrid/shortcut_block.py"
                shortcut.parent.mkdir(parents=True)
                shortcut.write_text('probe.start("experts_combine")')
                launches = []

                def launch(command, *, env, **kwargs):
                    launches.append((command, env.copy()))
                    out = Path(env["NT4_E2E_OUTPUT"])
                    out.mkdir(parents=True)
                    (out / "results.json").write_text("[]")
                    (out / "outputs-round1.json").write_text("[[1, 2]]")
                    return type("Process", (), {"wait": lambda self, **kw: 0})()

                argv = [
                    "nt4_e2e.py",
                    "--source-artifact",
                    str(source),
                    "--mcore-root",
                    str(mcore),
                    "--prompt-file",
                    str(root / "prompts.jsonl"),
                    "--output",
                    str(root / "out"),
                    "--precision",
                    precision,
                    "--model-graphs",
                    "--shortcut-overlap",
                    "--controlled-gc",
                    "--rounds",
                    "1",
                ]
                if branch_backend:
                    argv += [
                        "--request-count",
                        "128",
                        "--branch-contention",
                        schedule,
                        "--branch-backend",
                        branch_backend,
                        "--branch-input-mode",
                        (
                            "record"
                            if branch_backend == "baseline" and schedule == "overlap"
                            else "replay"
                        ),
                        "--branch-inputs",
                        str(root / "inputs"),
                    ]
                if experiment and experiment not in (
                    "hybrid",
                    "hybrid_confirm",
                    "surrogate",
                ):
                    argv += (
                        ["--cutlass-decode-profile-ids", "39", "428"]
                        if branch_backend == "baseline"
                        else ["--mxfp8-worker-clusters", "144"]
                    )
                if experiment == "hybrid_confirm":
                    argv += [
                        "--mxfp8-nvls-hybrid",
                        "--cutlass-decode-profile-ids",
                        "39",
                        "428",
                    ]
                if experiment in ("hybrid", "surrogate"):
                    argv += ["--mxfp8-nvls-hybrid"]
                if experiment == "hybrid_pipeline":
                    argv += ["--mxfp8-nvls-hybrid", "--branch-native-output"]
                if experiment == "surrogate":
                    argv += ["--trtllm-relu2-surrogate", "--branch-native-output"]
                if experiment == "confirm":
                    argv += [
                        "--mxfp8-worker-confirm",
                        "--cutlass-decode-profile-ids",
                        "39",
                        "428",
                    ]
                if experiment in ("pipeline", "hybrid_pipeline"):
                    argv[argv.index("--mxfp8-worker-clusters") + 1] = "152"
                    argv += ["--mxfp8-ab-stage-reduction", "1"]
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(
                        nt4_e2e,
                        "build_command",
                        return_value=["--", "generate.py", "--cuda-graph-impl", "none"],
                    ),
                    patch.object(nt4_e2e, "check_source_headers"),
                    patch.object(nt4_e2e.subprocess, "Popen", side_effect=launch),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    nt4_e2e.main()
                self.assertEqual(
                    launches[0][1]["NT4_MXFP8_NVLS_HYBRID"],
                    "1" if experiment in ("hybrid", "surrogate", "hybrid_pipeline") else "0",
                )
                if experiment in ("hybrid", "surrogate"):
                    self.assertNotIn("NT4_MXFP8_WORKER_CLUSTERS", launches[0][1])
                    self.assertNotIn("NT4_CUTLASS_DECODE_PROFILE_IDS", launches[0][1])
                    self.assertNotIn("NT4_MXFP8_AB_STAGE_REDUCTION", launches[0][1])
                elif experiment == "hybrid_confirm":
                    self.assertEqual(
                        [env["NT4_MXFP8_NVLS_HYBRID"] for _, env in launches],
                        ["0", "1", "0"],
                    )
                    for _, env in launches:
                        if env["NT4_E2E_MODE"] == "baseline":
                            self.assertEqual(
                                json.loads(env["NT4_CUTLASS_DECODE_PROFILE_IDS"]),
                                [39, 428],
                            )
                        else:
                            self.assertNotIn("NT4_CUTLASS_DECODE_PROFILE_IDS", env)
                        self.assertNotIn("NT4_MXFP8_WORKER_CLUSTERS", env)
                        self.assertNotIn("NT4_MXFP8_AB_STAGE_REDUCTION", env)
                elif experiment == "confirm":
                    for _, env in launches:
                        if env["NT4_E2E_MODE"] == "baseline":
                            self.assertEqual(
                                json.loads(env["NT4_CUTLASS_DECODE_PROFILE_IDS"]),
                                [39, 428],
                            )
                            self.assertNotIn("NT4_MXFP8_WORKER_CLUSTERS", env)
                        else:
                            self.assertEqual(env["NT4_MXFP8_WORKER_CLUSTERS"], "144")
                            self.assertNotIn("NT4_CUTLASS_DECODE_PROFILE_IDS", env)
                        self.assertNotIn("NT4_MXFP8_AB_STAGE_REDUCTION", env)
                elif experiment:
                    env = launches[0][1]
                    if branch_backend == "baseline":
                        self.assertEqual(
                            json.loads(env["NT4_CUTLASS_DECODE_PROFILE_IDS"]), [39, 428]
                        )
                        self.assertNotIn("NT4_MXFP8_WORKER_CLUSTERS", env)
                    else:
                        self.assertEqual(
                            env["NT4_MXFP8_WORKER_CLUSTERS"],
                            "152" if experiment in ("pipeline", "hybrid_pipeline") else "144",
                        )
                        if experiment in ("pipeline", "hybrid_pipeline"):
                            self.assertEqual(env["NT4_MXFP8_AB_STAGE_REDUCTION"], "1")
                        self.assertNotIn("NT4_CUTLASS_DECODE_PROFILE_IDS", env)
                self.assertEqual(
                    launches[0][1]["NT4_TRTLLM_RELU2_SURROGATE"],
                    "1" if experiment == "surrogate" else "0",
                )
                self.assertEqual(
                    launches[0][1]["NT4_BRANCH_NATIVE_OUTPUT"],
                    "1" if experiment in ("surrogate", "hybrid_pipeline") else "0",
                )
                expected = (
                    ["baseline", "direct", "baseline"]
                    if precision == "mxfp8"
                    else ["vllm", "direct", "vllm"]
                )
                if branch_backend:
                    expected = [branch_backend]
                self.assertEqual([env["NT4_E2E_MODE"] for _, env in launches], expected)
                if branch_backend:
                    self.assertEqual(launches[0][1]["NT4_BRANCH_CONTENTION"], schedule)
                    self.assertEqual(
                        launches[0][1]["NT4_BRANCH_INPUT_MODE"],
                        argv[argv.index("--branch-input-mode") + 1],
                    )
                for command, env in launches:
                    backend = command[
                        command.index("--inference-grouped-gemm-backend") + 1
                    ]
                    self.assertEqual(
                        backend, "flashinfer" if precision == "mxfp8" else "vllm"
                    )
                    self.assertIn("--inference-shortcut-moe-overlap", command)
                    self.assertEqual(env["NT4_E2E_MODEL_GRAPHS"], "1")
                    if precision == "mxfp8":
                        self.assertEqual(
                            command[
                                command.index(
                                    "--inference-shortcut-moe-expert-max-blocks"
                                )
                                + 1
                            ],
                            "0",
                        )
                report = json.loads((root / "out/summary.json").read_text())
                self.assertEqual(
                    report["comparison_reference"],
                    branch_backend
                    or ("baseline-start" if precision == "mxfp8" else "vllm-start"),
                )


if __name__ == "__main__":
    unittest.main()
