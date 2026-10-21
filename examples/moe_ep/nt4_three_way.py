"""Launch a live three-way capture using a previous artifact's model arguments.

This reloads the checkpoint. Saved outputs alone cannot time live CUTLASS.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def check_source_headers(checkout):
    required = (
        "3rdparty/cutlass/include/cutlass/numeric_types.h",
        "3rdparty/cutlass/include/cutlass/arch/barrier.h",
        "3rdparty/cccl/cub/cub/cub.cuh",
        "3rdparty/cccl/libcudacxx/include/cuda/std/version",
        "3rdparty/cccl/thrust/thrust/version.h",
        "3rdparty/spdlog/include/spdlog/spdlog.h",
    )
    missing = [name for name in required if not (checkout / name).is_file()]
    if missing:
        raise RuntimeError(
            "Incomplete FlashInfer source checkout; sync populated 3rdparty/cutlass, "
            "3rdparty/cccl and 3rdparty/spdlog from the matching local checkout. "
            "Missing: " + ", ".join(missing)
        )


def replace_model_options(argv, updates):
    result = []
    i = 0
    while i < len(argv):
        name = argv[i].split("=", 1)[0]
        if name in updates:
            inline = "=" in argv[i]
            i += 1
            if not inline:
                while i < len(argv) and not argv[i].startswith("--"):
                    i += 1
        else:
            result.append(argv[i])
            i += 1
    for name, value in updates.items():
        if value is not None:
            result.extend([name, str(value)])
    return result


def build_command(
    manifest,
    *,
    mcore_root,
    artifact,
    samples,
    counts,
    buckets="64,128,256,512",
    max_local_tokens=64,
    prompt_file=None,
    request_count=None,
    min_active_tokens=None,
    capture_interval=8,
    ignore_eos=False,
    profile_dir=None,
    profile_iterations=10,
    knobs=None,
    model_token_budget=None,
    match_production_capacity=False,
):
    command = manifest["provenance"]["command"]
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(s, str) for s in command)
    ):
        raise ValueError(
            "artifact provenance must contain the original model argument list"
        )
    root = Path(__file__).resolve().parent
    if Path(command[0]).name != "nt4_checkpoint_generate.py":
        raise ValueError("expected model arguments from nt4_checkpoint_generate.py")
    model_args = command[1:]
    if ignore_eos and prompt_file is None:
        raise ValueError("--ignore-eos requires real prompt-file capture")
    if model_token_budget is not None and (
        prompt_file is None or model_token_budget < (request_count or 1)
    ):
        raise ValueError(
            "model token budget requires real prompts and must fit all requests"
        )
    if match_production_capacity and prompt_file is None:
        raise ValueError("matched capacity requires real prompt capture")
    capture_args = ["--sweep-counts", counts, "--sweep-bucket-rows", buckets]
    if prompt_file is not None:
        if request_count is None or request_count < 1:
            raise ValueError("real capture requires a positive request count")
        minimum = request_count if min_active_tokens is None else min_active_tokens
        if not 1 <= minimum <= request_count:
            raise ValueError("minimum active tokens must lie within request count")
        max_local_tokens = max(max_local_tokens, request_count)
        # Preserve the production token budget unless it cannot fit the batch.
        current_budget = 1024
        for i, arg in enumerate(model_args):
            if arg == "--inference-dynamic-batching-max-tokens":
                current_budget = int(model_args[i + 1])
            elif arg.startswith("--inference-dynamic-batching-max-tokens="):
                current_budget = int(arg.split("=", 1)[1])
        if model_token_budget is not None:
            current_budget = model_token_budget
        model_args = replace_model_options(
            model_args,
            {
                "--prompts": None,
                "--prompt-file": prompt_file,
                "--prompt-file-num-truncate": request_count,
                "--incoming-requests-per-sec": -1,
                "--inference-dynamic-batching-max-requests": request_count,
                "--inference-dynamic-batching-max-tokens": max(
                    current_budget, request_count
                ),
            },
        )
        capture_args = [
            "--decode-only",
            "--min-active-tokens",
            str(minimum),
            "--capture-interval",
            str(capture_interval),
        ]
        if ignore_eos:
            model_args = replace_model_options(
                model_args, {"--termination-id": -1, "--stop-words": None}
            )
    if profile_iterations < 1:
        raise ValueError("profile iterations must be positive")
    profile_args = (
        []
        if profile_dir is None
        else [
            "--profile-dir",
            str(profile_dir),
            "--profile-iterations",
            str(profile_iterations),
        ]
    )
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(root / "nt4_checkpoint_capture.py"),
        "--mcore-root",
        str(mcore_root),
        "--artifact",
        str(artifact),
        "--layer",
        str(manifest["layer"]),
        "--samples",
        str(samples),
        "--skip-calls",
        "1",
        "--max-local-tokens",
        str(max_local_tokens),
        "--three-way",
        "--graph",
        "--benchmark",
        "--rounds",
        "6",
        "--warmup",
        "20",
        "--iterations",
        "100",
        "--timeout-seconds",
        "7200",
        "--routing-weight-placement",
        "after_fc2",
        "--knobs-json",
        json.dumps({"group_hint": 512} if knobs is None else knobs),
        *capture_args,
        *profile_args,
        *(["--match-production-capacity"] if match_production_capacity else []),
        "--",
        str(root / "nt4_checkpoint_generate.py"),
        *model_args,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--mcore-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sweep-counts", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--sweep-bucket-rows", default="64,128,256,512")
    parser.add_argument("--max-local-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-file", type=Path, help="real decode capture; disables synthetic sweep"
    )
    parser.add_argument("--match-production-capacity", action="store_true")
    parser.add_argument("--request-count", type=int)
    parser.add_argument(
        "--model-token-budget",
        type=int,
        help="explicit production token budget for capacity comparisons",
    )
    parser.add_argument("--min-active-tokens", type=int)
    parser.add_argument("--capture-interval", type=int, default=8)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="controlled fixed-length workload: disable EOS and stop words to retain the decode cohort",
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--knobs-json", default='{"group_hint":512}')
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from nt4_gb300_tune import validate_knobs

    try:
        knobs = validate_knobs(json.loads(args.knobs_json))
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    if (args.prompt_file is None) != (args.request_count is None):
        parser.error(
            "supply both --prompt-file and --request-count for real decode capture"
        )
    if args.prompt_file is not None and not args.prompt_file.is_file():
        parser.error("prompt file does not exist")
    if args.ignore_eos and args.prompt_file is None:
        parser.error("--ignore-eos requires --prompt-file")
    if args.capture_interval < 1:
        parser.error("capture interval must be positive")
    if args.samples < 1:
        parser.error("samples must be positive")
    if args.artifact.exists():
        parser.error("artifact must be a new directory")
    manifest = json.loads((args.source_artifact / "manifest.json").read_text())
    if not manifest.get("capture_complete") or manifest.get("world_size") != 4:
        parser.error("source artifact must be a completed EP4 capture")
    command = build_command(
        manifest,
        mcore_root=args.mcore_root,
        artifact=args.artifact,
        samples=args.samples,
        counts=args.sweep_counts,
        buckets=args.sweep_bucket_rows,
        max_local_tokens=args.max_local_tokens,
        prompt_file=args.prompt_file,
        request_count=args.request_count,
        min_active_tokens=args.min_active_tokens,
        capture_interval=args.capture_interval,
        ignore_eos=args.ignore_eos,
        profile_dir=args.profile_dir,
        profile_iterations=args.profile_iterations,
        knobs=knobs,
        model_token_budget=args.model_token_budget,
        match_production_capacity=args.match_production_capacity,
    )
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        checkout = Path(__file__).resolve().parents[2]
        check_source_headers(checkout)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(checkout), str(args.mcore_root), env.get("PYTHONPATH", "")]
        )
        # MCore catches ImportError and reports a generic minimum-version error.
        # Import directly first so missing dependencies retain their traceback.
        preflight = """
import json
from pathlib import Path
import flashinfer
from packaging.version import Version
print(json.dumps(dict(stage='flashinfer_preflight', source=flashinfer.__file__, version=flashinfer.__version__)), flush=True)
assert Path(flashinfer.__file__).resolve().parent.parent == Path.cwd().resolve(), 'wrong FlashInfer checkout imported'
if Version(flashinfer.__version__) < Version('0.6.4'):
    raise RuntimeError('FlashInfer version is below 0.6.4 or build metadata is missing. Install THIS checkout with: python -m pip install --no-deps --no-build-isolation -e . ; do not bypass the version check.')
"""
        subprocess.run(
            [sys.executable, "-c", preflight], cwd=checkout, env=env, check=True
        )
        sys.exit(subprocess.call(command, env=env, cwd=checkout))


if __name__ == "__main__":
    main()
