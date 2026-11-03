"""Checkpoint E2E comparison of automatic GC and collection outside generation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def source_metadata(checkout):
    """Best-effort provenance; synced directories need neither Git nor .git."""
    result = {"path": str(checkout.resolve()), "head": None, "status": None}
    try:
        for key, command in (
            ("head", ["rev-parse", "HEAD"]),
            ("status", ["status", "--porcelain", "--untracked-files=no"]),
        ):
            process = subprocess.run(
                ["git", "-C", str(checkout), *command],
                capture_output=True, text=True, timeout=10, check=True,
            )
            result[key] = process.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        result["git_metadata_unavailable"] = type(exc).__name__
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mcore-root", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest_path = args.source_artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model = manifest["provenance"]["command"]
    for flag in ("--load", "--tokenizer-model"):
        value = model[model.index(flag) + 1]
        if not Path(value).exists():
            parser.error(f"{flag} path from source manifest is unavailable: {value}")
    if not args.prompt_file.is_file():
        parser.error(f"missing prompts: {args.prompt_file}")
    if not (args.mcore_root / "megatron").is_dir():
        parser.error("--mcore-root must contain megatron/")
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    # Prevent an unrelated experiment's NT4 environment from changing either arm.
    env = {k: v for k, v in os.environ.items() if not k.startswith("NT4_")}
    env["NT4_ALLOW_SOURCE_JIT"] = "1"
    env["NT4_AUDIT_GPU_TELEMETRY"] = "0"  # No periodic writes to Lustre.
    env["PYTHONPATH"] = os.pathsep.join((str(root), str(args.mcore_root.resolve())))
    env.pop("FLASHINFER_TRTLLM_NT4_EXPORT", None)
    common = [sys.executable, str(root / "examples/moe_ep/nt4_e2e.py"),
              "--mcore-root", str(args.mcore_root.resolve()),
              "--source-artifact", str(args.source_artifact.resolve()),
              "--prompt-file", str(args.prompt_file.resolve()),
              "--precision", "bf16", "--request-count", "128",
              "--model-token-budget", "2048", "--num-tokens-to-generate", "128",
              "--model-graphs", "--shortcut-overlap", "--rounds", "3",
              "--vllm-variance-audit", "2", "--timeout-seconds", "7200"]
    commands = []
    for arm in ("automatic", "controlled"):
        command = common + ["--output", str(args.output.resolve() / arm)]
        if arm == "controlled":
            command += ["--controlled-gc"]
        commands.append((arm, command))
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    args.output.mkdir(parents=True)
    versions = {}
    for name, checkout in (("flashinfer", root), ("mcore", args.mcore_root)):
        versions[name] = source_metadata(checkout)
    (args.output / "provenance.json").write_text(json.dumps(versions, indent=2))
    with (args.output / "packages.txt").open("w") as log:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=log, check=True)
    for arm, command in commands:
        print(f"Starting {arm}: {args.output / (arm + '.log')}", flush=True)
        with (args.output / (arm + ".log")).open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Complete: {args.output}")


if __name__ == "__main__":
    main()
