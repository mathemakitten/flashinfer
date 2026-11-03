"""CPU checks for reconstructing a live run without shell argument corruption."""

import json
import importlib.util
from pathlib import Path
import unittest
import tempfile

path = Path(__file__).resolve().parents[2] / "examples/moe_ep/nt4_three_way.py"
spec = importlib.util.spec_from_file_location("nt4_three_way", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LaunchTest(unittest.TestCase):
    def test_explicit_production_capacity_budget(self):
        manifest = {
            "layer": 7,
            "provenance": {
                "command": [
                    "nt4_checkpoint_generate.py",
                    "--inference-dynamic-batching-max-tokens=1024",
                ]
            },
        }
        kwargs = dict(
            mcore_root=Path("/m"),
            artifact=Path("/a"),
            samples=3,
            counts="128",
            prompt_file=Path("/p"),
            request_count=128,
        )
        command = module.build_command(manifest, **kwargs, model_token_budget=128)
        flag = "--inference-dynamic-batching-max-tokens"
        self.assertEqual(command.count(flag), 1)
        self.assertEqual(command[command.index(flag) + 1], "128")
        self.assertNotIn(flag + "=1024", command)
        with self.assertRaises(ValueError):
            module.build_command(manifest, **kwargs, model_token_budget=64)
        default = module.build_command(manifest, **kwargs)
        self.assertEqual(default[default.index(flag) + 1], "1024")

    def test_matched_capacity_preserves_engine_budget(self):
        manifest = {
            "layer": 7,
            "provenance": {
                "command": [
                    "nt4_checkpoint_generate.py",
                    "--inference-dynamic-batching-max-tokens",
                    "1024",
                ]
            },
        }
        kwargs = dict(
            mcore_root=Path("/m"),
            artifact=Path("/a"),
            samples=3,
            counts="128",
            prompt_file=Path("/p"),
            request_count=128,
        )
        command = module.build_command(
            manifest, **kwargs, match_production_capacity=True
        )
        self.assertLess(
            command.index("--match-production-capacity"), command.index("--")
        )
        flag = "--inference-dynamic-batching-max-tokens"
        self.assertEqual(command[command.index(flag) + 1], "1024")
        with self.assertRaises(ValueError):
            module.build_command(
                manifest,
                **(kwargs | {"prompt_file": None}),
                match_production_capacity=True,
            )

    def test_geometry_knobs_stay_in_wrapper(self):
        knobs = dict(
            mma_tiler_mnk=[128, 256, 128],
            cluster_shape_mnk=[1, 1, 1],
            use_2cta_instrs=False,
            group_hint=1024,
            flag_batch=1,
        )
        command = module.build_command(
            {
                "layer": 7,
                "provenance": {"command": ["nt4_checkpoint_generate.py", "--bf16"]},
            },
            mcore_root=Path("/m"),
            artifact=Path("/a"),
            samples=3,
            counts="128",
            knobs=knobs,
        )
        index = command.index("--knobs-json")
        self.assertLess(index, command.index("--"))
        self.assertEqual(json.loads(command[index + 1]), knobs)

    def test_profile_flags_stay_in_capture_wrapper(self):
        command = module.build_command(
            {
                "layer": 7,
                "provenance": {"command": ["nt4_checkpoint_generate.py", "--bf16"]},
            },
            mcore_root=Path("/m"),
            artifact=Path("/a"),
            samples=1,
            counts="128",
            profile_dir=Path("/profiles with spaces"),
            profile_iterations=12,
        )
        split = command.index("--")
        self.assertEqual(
            command[command.index("--profile-dir") + 1], "/profiles with spaces"
        )
        self.assertEqual(command[command.index("--profile-iterations") + 1], "12")
        self.assertLess(command.index("--profile-dir"), split)
        self.assertNotIn("--profile-dir", command[split + 1 :])

    def test_empty_submodules_fail_before_launch(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "cutlass/numeric_types.h"):
                module.check_source_headers(Path(root))

    def test_preserves_model_arguments_and_uses_current_scripts(self):
        model_args = [
            "--load",
            "/checkpoint with spaces",
            "--prompts",
            "math $x; `literal`",
            "--bf16",
        ]
        manifest = {
            "layer": 7,
            "provenance": {"command": ["/old/nt4_checkpoint_generate.py", *model_args]},
        }
        command = module.build_command(
            manifest,
            mcore_root=Path("/mcore"),
            artifact=Path("/new"),
            samples=1,
            counts="1,8,64",
        )
        split = command.index("--")
        self.assertEqual(command[split + 2 :], model_args)
        self.assertEqual(
            command[split + 1], str(path.parent / "nt4_checkpoint_generate.py")
        )
        self.assertIn(str(path.parent / "nt4_checkpoint_capture.py"), command[:split])
        self.assertEqual(command[command.index("--layer") + 1], "7")
        self.assertIn("--three-way", command[:split])
        self.assertEqual(
            command[command.index("--sweep-bucket-rows") + 1], "64,128,256,512"
        )

    def test_real_mode_replaces_admission_and_disables_sweep(self):
        argv = [
            "/old/nt4_checkpoint_generate.py",
            "--prompts",
            "one",
            "two",
            "--prompt-file=old.jsonl",
            "--prompt-file-num-truncate",
            "50",
            "--incoming-requests-per-sec",
            "-1",
            "--inference-dynamic-batching-max-requests",
            "64",
            "--inference-dynamic-batching-max-tokens",
            "1024",
            "--load",
            "/original",
        ]
        command = module.build_command(
            {"layer": 7, "provenance": {"command": argv}},
            mcore_root=Path("/m"),
            artifact=Path("/n"),
            samples=3,
            counts="1,512",
            prompt_file=Path("/new.jsonl"),
            request_count=256,
        )
        split = command.index("--")
        wrapper, model = command[:split], command[split + 2 :]
        self.assertNotIn("--sweep-counts", wrapper)
        self.assertIn("--decode-only", wrapper)
        self.assertEqual(wrapper[wrapper.index("--min-active-tokens") + 1], "256")
        self.assertNotIn("--prompts", model)
        self.assertNotIn("--prompt-file=old.jsonl", model)
        for flag, value in {
            "--prompt-file": "/new.jsonl",
            "--prompt-file-num-truncate": "256",
            "--inference-dynamic-batching-max-requests": "256",
            "--inference-dynamic-batching-max-tokens": "1024",
            "--load": "/original",
        }.items():
            self.assertEqual(model.count(flag), 1)
            self.assertEqual(model[model.index(flag) + 1], value)

    def test_rejects_unknown_driver(self):
        with self.assertRaises(ValueError):
            module.build_command(
                {"provenance": {"command": ["unknown.py"]}},
                mcore_root=Path("/m"),
                artifact=Path("/n"),
                samples=1,
                counts="1",
            )

    def test_ignore_eos_is_explicit_and_removes_stop_words(self):
        manifest = {
            "layer": 7,
            "provenance": {
                "command": [
                    "nt4_checkpoint_generate.py",
                    "--termination-id",
                    "42",
                    "--stop-words",
                    "stop here",
                    "--num-tokens-to-generate",
                    "128",
                ]
            },
        }
        kwargs = dict(
            mcore_root=Path("/m"),
            artifact=Path("/n"),
            samples=3,
            counts="128",
            prompt_file=Path("/p"),
            request_count=128,
        )
        natural = module.build_command(manifest, **kwargs)
        self.assertEqual(natural[natural.index("--termination-id") + 1], "42")
        fixed = module.build_command(manifest, **kwargs, ignore_eos=True)
        self.assertEqual(fixed.count("--termination-id"), 1)
        self.assertEqual(fixed[fixed.index("--termination-id") + 1], "-1")
        self.assertNotIn("--stop-words", fixed)
        self.assertEqual(fixed[fixed.index("--min-active-tokens") + 1], "128")


if __name__ == "__main__":
    unittest.main()
