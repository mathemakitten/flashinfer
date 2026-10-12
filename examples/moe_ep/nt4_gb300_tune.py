"""Isolated GB300 NT4 scheduling experiments on saved checkpoint artifacts.

No production defaults are changed. Every trial validates correctness before
CUDA-graph timing. Separate torchrun jobs contain collective failures/hangs.
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

CANDIDATES = {
    "default": {},
    "atomic": {"load_balance_mode": "atomic_counter"},
    "group32": {"group_hint": 32},
    "group128": {"group_hint": 128},
    "flag1": {"flag_batch": 1},
    "flag8": {"flag_batch": 8},
    "epilogue2": {"epi_flag_batch": [2, 2]},
    "dispatch_return": {"token_back_mode": "reuse_dispatch_warps"},
    "atomic_group128": {"load_balance_mode": "atomic_counter", "group_hint": 128},
}
INITIAL_CANDIDATES = tuple(CANDIDATES)
for group in (16, 64, 96, 192, 256):
    CANDIDATES[f"group{group}"] = {"group_hint": group}
for batch in (1, 8):
    CANDIDATES[f"group128_flag{batch}"] = {"group_hint": 128, "flag_batch": batch}
for fc1, fc2 in ((2, 1), (1, 2), (2, 2)):
    CANDIDATES[f"group128_epi{fc1}{fc2}"] = {
        "group_hint": 128,
        "epi_flag_batch": (fc1, fc2),
    }
REFINE_CANDIDATES = (
    "group128",
    "group16",
    "group32",
    "group64",
    "group96",
    "group192",
    "group256",
    "group128_flag1",
    "group128_flag8",
    "group128_epi21",
    "group128_epi12",
    "group128_epi22",
)


for group in (384, 512, 768):
    CANDIDATES[f"group{group}"] = {"group_hint": group}
for batch in (1, 2, 8):
    CANDIDATES[f"group256_flag{batch}"] = {"group_hint": 256, "flag_batch": batch}
GROUP256_CANDIDATES = (
    "group256",
    "group384",
    "group512",
    "group768",
    "group256_flag1",
    "group256_flag2",
    "group256_flag8",
)

# Reuse the existing explicit FC2-ready counters and staged token-return path.
# Both arms retain the cross-rank completion/retirement barriers. This measures
# staging + return scheduling, NOT the isolated cost of destination signaling.
CANDIDATES["group512_direct_return"] = {
    "group_hint": 512,
    "token_back_mode": "epi_warps",
}
CANDIDATES["group512_staged_return"] = {
    "group_hint": 512,
    "token_back_mode": "reuse_dispatch_warps",
}
RETURN_CONTROL_SEQUENCE = (
    "group512_direct_return",
    "group512_staged_return",
    "group512_direct_return",
)


def trial_sequence(names, refine=False, control=None):
    sequence = ["default"] + [n for n in names if n != "default"]
    if control or refine:
        sequence.append(control or "group128")
    return sequence + ["default"]


def validate_knobs(knobs):
    if not isinstance(knobs, dict):
        raise ValueError("knobs must be a JSON object")
    for key, value in knobs.items():
        if key == "load_balance_mode":
            valid = value in ("static", "atomic_counter")
        elif key == "group_hint":
            valid = type(value) is int and value > 0
        elif key == "flag_batch":
            valid = type(value) is int and 1 <= value <= 16
        elif key == "epi_flag_batch":
            valid = (
                isinstance(value, (tuple, list))
                and len(value) == 2
                and all(type(v) is int and 1 <= v <= 32 for v in value)
            )
        elif key == "token_back_mode":
            valid = value in ("epi_warps", "reuse_dispatch_warps")
        else:
            raise ValueError(f"unsupported NT4 experiment knob: {key}")
        if not valid:
            raise ValueError(f"invalid {key}: {value!r}")
    normalized = dict(knobs)
    if "epi_flag_batch" in normalized:
        normalized["epi_flag_batch"] = tuple(normalized["epi_flag_batch"])
    return normalized


def records(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{"rank":', text):
        try:
            item, _ = decoder.raw_decode(text[match.start() :])
        except ValueError:
            continue
        yield item


def trial_summary(text, returncode):
    items = list(records(text))
    checks = [
        r
        for r in items
        if r.get("stage") == "correctness" and r.get("status") == "PASS"
    ]
    timing = [r for r in items if r.get("stage") == "benchmark" and r.get("rank") == 0]
    passed = (
        returncode == 0 and {r["rank"] for r in checks} == {0, 1, 2, 3} and bool(timing)
    )
    return dict(
        status="PASS" if passed else "FAIL",
        returncode=returncode,
        benchmarks=timing if passed else [],
        config=next(
            (
                r
                for r in items
                if r.get("stage") == "tuning_config" and r.get("rank") == 0
            ),
            None,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        required=True,
        help="repeat for diverse primary and narrow-routing control",
    )
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--candidates")
    selection.add_argument(
        "--preset",
        choices=("initial", "refine", "group256", "return_control"),
        default="initial",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    names = (
        args.candidates.split(",")
        if args.candidates
        else {
            "initial": INITIAL_CANDIDATES,
            "refine": REFINE_CANDIDATES,
            "group256": GROUP256_CANDIDATES,
            "return_control": RETURN_CONTROL_SEQUENCE,
        }[args.preset]
    )
    if args.timeout_seconds < 1 or any(n not in CANDIDATES for n in names):
        parser.error("invalid timeout or candidate name")
    for artifact in args.artifact:
        if not (artifact / "manifest.json").is_file():
            parser.error(f"missing manifest: {artifact}")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    # Bracket the candidate sweep with controls to expose drift; do not promote
    # a winner automatically or compare against timings from a different node.
    sequence = trial_sequence(
        names,
        refine=args.preset == "refine",
        control="group256" if args.preset == "group256" else None,
    )
    if args.preset == "return_control" and not args.candidates:
        sequence = list(RETURN_CONTROL_SEQUENCE)
    replay = Path(__file__).with_name("nt4_checkpoint_replay.py")
    for trial, name in enumerate(sequence):
        knobs = validate_knobs(CANDIDATES[name])
        for index, artifact in enumerate(args.artifact):
            log = args.output / f"{trial:02d}-{name}-case{index}.log"
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=4",
                str(replay),
                "--artifact",
                str(artifact.resolve()),
                "--graph",
                "--benchmark-megamoe",
                "--require-gb300",
                "--routing-weight-placement",
                "after_fc2",
                "--knobs-json",
                json.dumps(knobs),
                "--timeout-seconds",
                str(args.timeout_seconds),
            ]
            if args.allow_incomplete:
                cmd.append("--allow-incomplete")
            print(
                json.dumps(
                    dict(
                        stage="trial_start",
                        candidate=name,
                        artifact=str(artifact),
                        log=str(log),
                    )
                ),
                flush=True,
            )
            start = time.monotonic()
            with log.open("w") as stream:
                process = subprocess.Popen(
                    cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
                try:
                    code = process.wait(timeout=args.timeout_seconds + 30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    code = 124
                except BaseException:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            record = dict(
                trial=trial,
                candidate=name,
                knobs=knobs,
                artifact=str(artifact),
                log=str(log),
                seconds=time.monotonic() - start,
                **trial_summary(log.read_text(), code),
            )
            if args.preset == "return_control" and not args.candidates:
                record["return_protocol"] = {
                    "producer": (
                        "fc2_local_stores_then_ready_counter"
                        if knobs["token_back_mode"] == "reuse_dispatch_warps"
                        else "fc2_direct_peer_stores"
                    ),
                    "destination_completion": "cross_rank_barrier",
                    "counted_writes": False,
                    "scope": "whole_moe_staging_and_return_comparison",
                }
            results.append(record)
            temporary = args.output / "results.tmp"
            temporary.write_text(json.dumps(results, indent=2))
            temporary.replace(args.output / "results.json")
            print(json.dumps(record), flush=True)
            if record["status"] != "PASS":
                # A device/collective error can poison the node. Preserve evidence
                # and stop rather than collecting potentially invalid later timings.
                raise SystemExit(
                    f"Trial failed; inspect {log}. Completed results are saved."
                )


if __name__ == "__main__":
    main()
