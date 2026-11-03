"""Resumable NT4 precision/batch sweep using actual checkpoint generation."""

import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-artifact", "mcore-root", "prompt-file", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--counts", default="1,2,4,8,16,32,64,128,256,512,1024")
    p.add_argument("--precisions", default="mxfp8,bf16")
    p.add_argument("--model-token-budget", type=int, default=2048)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--timeout-seconds", type=int, default=7200)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    counts = [int(x) for x in a.counts.split(",")]
    precisions = a.precisions.split(",")
    if not counts or min(counts) < 1 or max(counts) > a.model_token_budget:
        p.error("positive batch sizes must fit the fixed token budget")
    if any(x not in ("mxfp8", "bf16") for x in precisions):
        p.error("precisions must be mxfp8 and/or bf16")
    if min(a.rounds, a.timeout_seconds) < 1:
        p.error("rounds and timeout must be positive")
    with a.prompt_file.open() as f:
        prompt_count = sum(bool(line.strip()) for line in f)
    if prompt_count < max(counts):
        p.error(f"need {max(counts)} prompts, found {prompt_count}")
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
        if k != "resume"
    }
    if a.resume:
        if json.loads((a.output / "sweep-config.json").read_text()) != config:
            p.error("resume requires the same experiment configuration")
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        (a.output / "sweep-config.json").write_text(json.dumps(config, indent=2))
    # Compile and validate the new kernel before paying for full model loads.
    smoke_marker = a.output / "bf16-smoke-pass.json"
    if "bf16" in precisions and not smoke_marker.exists():
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(Path(__file__).with_name("nt4_bf16_bringup.py")),
        ]
        log = a.output / "bf16-smoke.log"
        print(json.dumps(dict(stage="bf16_smoke_start", log=str(log))), flush=True)
        with log.open("w") as f:
            proc = subprocess.Popen(
                command, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = proc.wait(timeout=a.timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise SystemExit(f"BF16 smoke timed out; see {log}")
        if code:
            raise SystemExit(
                f"BF16 smoke failed ({code}); sweep not launched; see {log}"
            )
        smoke_marker.write_text(json.dumps(dict(status="PASS", command=command)))
    results = []
    for precision in precisions:
        for count in counts:
            case = a.output / f"{precision}-requests{count}"
            case.mkdir(exist_ok=True)
            summaries = sorted(case.glob("attempt-*/summary.json"))
            if summaries:
                summary = summaries[-1]
                code = 0
            else:
                attempt = case / f'attempt-{len(list(case.glob("attempt-*"))):03d}'
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("nt4_e2e.py")),
                    "--source-artifact",
                    str(a.source_artifact),
                    "--mcore-root",
                    str(a.mcore_root),
                    "--prompt-file",
                    str(a.prompt_file),
                    "--output",
                    str(attempt),
                    "--precision",
                    precision,
                    "--request-count",
                    str(count),
                    "--model-token-budget",
                    str(a.model_token_budget),
                    "--rounds",
                    str(a.rounds),
                    "--timeout-seconds",
                    str(a.timeout_seconds),
                    "--routed-graphs",
                ]
                log = case / f"{attempt.name}.log"
                print(
                    json.dumps(
                        dict(
                            stage="sweep_start",
                            precision=precision,
                            requests=count,
                            log=str(log),
                        )
                    ),
                    flush=True,
                )
                with log.open("w") as f:
                    code = subprocess.run(
                        command, stdout=f, stderr=subprocess.STDOUT
                    ).returncode
                summary = attempt / "summary.json"
            entry = dict(
                precision=precision,
                requests=count,
                returncode=code,
                summary=str(summary),
            )
            if code == 0 and summary.exists():
                report = json.loads(summary.read_text())
                timed = [
                    r
                    for key in ("baseline-start", "direct", "vllm-nvls", "baseline-end")
                    for r in report.get(key, [])
                    if not r["warmup"]
                ]
                entry["status"] = (
                    "PASS"
                    if timed
                    and all(r.get("requested_batch_reached") is True for r in timed)
                    else "OCCUPANCY_LIMITED"
                )
                entry["results"] = report
            else:
                entry["status"] = "FAIL"
            results.append(entry)
            (a.output / "sweep-summary.json").write_text(json.dumps(results, indent=2))
            print(
                json.dumps({k: v for k, v in entry.items() if k != "results"}),
                flush=True,
            )
    if any(r["status"] != "PASS" for r in results):
        raise SystemExit(
            "Sweep finished with failed or occupancy-limited points; inspect sweep-summary.json"
        )


if __name__ == "__main__":
    main()
