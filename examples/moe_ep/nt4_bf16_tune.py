"""Correctness-gated BF16 scheduling sweep on saved real checkpoint routing.

Each trial uses a fresh EP4 process group. Bracketing unchanged controls and
per-sample results are retained; no candidate is installed automatically.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

# torchrun adds the examples directory, not the source checkout, to sys.path.
# Pin both before importing helpers that transitively import FlashInfer.
_CHECKOUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CHECKOUT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nt4_bf16_tuning_config import CANDIDATES, validate_knobs


def worker(a):
    import inspect
    import flashinfer
    from flashinfer.moe_ep import Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig

    config_source = Path(
        inspect.getfile(Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig)
    ).resolve()
    required = {"activation", "clamped_relu2_limit"}
    parameters = inspect.signature(
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig
    ).parameters
    if not config_source.is_relative_to(_CHECKOUT) or not required.issubset(parameters):
        raise RuntimeError(
            f"Incomplete or wrong NT4 FlashInfer checkout: {config_source}. "
            f"Expected NT4 activation config under {_CHECKOUT}. Sync the full "
            "flashinfer/moe_ep package, including backends/mega/kernel/sm100/"
            "bf16_bf16_bf16_cutedsl and kernel_src/sm100/cutedsl_megamoe/shim."
        )
    print(
        json.dumps(
            dict(
                stage="bf16_worker_source",
                rank=os.environ.get("RANK"),
                flashinfer_source=flashinfer.__file__,
                config_source=str(config_source),
            )
        ),
        flush=True,
    )
    import torch
    import torch.distributed as dist
    from nt4_bf16_replay import Bf16Replay
    from nt4_checkpoint_replay import compare

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    manifest = json.loads(a.artifact[0].joinpath("manifest.json").read_text())
    if (
        not manifest.get("complete")
        or manifest.get("precision") != "bf16"
        or dist.get_world_size() != 4
    ):
        raise ValueError("requires completed real BF16 EP4 capture")
    knobs = validate_knobs(CANDIDATES[a.candidate])
    results = []
    layers = [int(x) for x in a.layers.split(",")] if a.layers else manifest["layers"]
    for layer in layers:
        if layer not in manifest["layers"]:
            raise ValueError(f"layer {layer} not captured")
        root = a.artifact[0] / f"layer{layer}" / f"rank{rank}"
        meta = json.loads((root / "metadata.json").read_text())
        weights = torch.load(
            root / "weights.pt", map_location="cuda", weights_only=True
        )
        replay = Bf16Replay(weights, meta["capacity"], knobs=knobs)
        for sample_id in range(manifest["samples"]):
            sample = torch.load(
                root / f"sample{sample_id}.pt", map_location="cpu", weights_only=True
            )
            replay.check(sample)
            if sample_id == 0:
                print(
                    json.dumps(
                        dict(
                            stage="bf16_effective_config",
                            rank=rank,
                            layer=layer,
                            **replay.audit_config(),
                        )
                    ),
                    flush=True,
                )
            n = sample["x"].shape[0]
            sx, si, sp = [sample[k].cuda() for k in ("x", "ids", "probs")]

            def forward():
                # Match the E2E direct graph's capacity clearing and live staging.
                replay.x.zero_()
                replay.ids.fill_(-1)
                replay.probs.zero_()
                replay.x[:n].copy_(sx)
                replay.ids[:n].copy_(si)
                replay.probs[:n].copy_(sp)
                return replay.run()[:n]

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(a.warmup):
                    expected = forward()
                expected = expected.clone()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = forward()
            graph.replay()
            failure = None
            try:
                compare(output, expected, "bf16_tuning_graph_vs_eager")
            except (AssertionError, RuntimeError) as exc:
                failure = str(exc)
            failures = [None] * 4
            dist.all_gather_object(failures, failure)
            if any(failures):
                raise AssertionError(str(failures))
            if a.profile:
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.cudart().cudaProfilerStart()
                with torch.cuda.nvtx.range(
                    f"bf16_layer{layer}_sample{sample_id}_{a.candidate}"
                ):
                    for _ in range(20):
                        graph.replay()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                graph.reset()
                replay.close()
                dist.destroy_process_group()
                return
            times = []
            for _ in range(a.rounds):
                for _ in range(a.warmup):
                    graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(a.iterations):
                    graph.replay()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(
                    start.elapsed_time(end) * 1000 / a.iterations,
                    device="cuda",
                    dtype=torch.float64,
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                times.append(elapsed.item())
            row = dict(
                layer=layer,
                sample=sample_id,
                rounds_us=times,
                median_us=statistics.median(times),
                capacity=meta["capacity"],
                buffer_rows=n,
                active_routed_rows=int((si >= 0).any(dim=1).sum().item()),
            )
            results.append(row)
            if rank == 0:
                print(
                    json.dumps(
                        dict(stage="bf16_tune_timing", candidate=a.candidate, **row)
                    ),
                    flush=True,
                )
            graph.reset()
            del graph, output, expected, sx, si, sp, sample
        replay.close()
        del replay, weights
        torch.cuda.empty_cache()
    if rank == 0:
        a.result.write_text(
            json.dumps(
                dict(
                    candidate=a.candidate,
                    knobs=knobs,
                    status="PASS",
                    results=results,
                    gpu=torch.cuda.get_device_name(),
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    geometry=[256, 256, 64],
                    cluster=[2, 1, 1],
                    graph_pool_policy="per_bucket_private",
                    timing_scope="capacity clearing, staging, BF16 MegaMoE, late weighted reduction; max rank GPU event time",
                ),
                indent=2,
            )
        )
    dist.barrier()
    dist.destroy_process_group()


def recommend(trials):
    """Require both controls, low drift, and a win on every captured sample."""
    controls = {}
    for trial in trials:
        if trial["candidate"] == "default" and trial["status"] == "PASS":
            for r in trial["results"]:
                controls.setdefault(
                    (trial["artifact"], r["layer"], r["sample"]), []
                ).append(r["median_us"])
    if not controls or any(len(v) != 2 for v in controls.values()):
        return dict(eligible=[], reason="missing bracketing controls")
    drift = max(abs(v[1] / v[0] - 1) for v in controls.values())
    rows = []
    for candidate in sorted({t["candidate"] for t in trials} - {"default"}):
        matches = [t for t in trials if t["candidate"] == candidate]
        values = {
            (t["artifact"], r["layer"], r["sample"]): r["median_us"]
            for t in matches
            if t["status"] == "PASS"
            for r in t["results"]
        }
        if set(values) != set(controls) or any(t["status"] != "PASS" for t in matches):
            rows.append(
                dict(
                    candidate=candidate,
                    eligible=False,
                    reason="missing or failed cases",
                )
            )
            continue
        # Compare against the faster control, not an advantageous average.
        ratios = [min(controls[k]) / v for k, v in values.items()]
        rows.append(
            dict(
                candidate=candidate,
                eligible=drift <= 0.03 and min(ratios) >= 1.02,
                worst_speedup=min(ratios),
                median_speedup=statistics.median(ratios),
            )
        )
    eligible = sorted(
        [r for r in rows if r["eligible"]],
        key=lambda r: r["worst_speedup"],
        reverse=True,
    )
    return dict(
        max_control_drift=drift,
        candidates=rows,
        eligible=eligible,
        rule="<=3% control drift and >=2% improvement against faster control on every sample; E2E still required",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--candidates", default=",".join(k for k in CANDIDATES if k != "default")
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--candidate", choices=CANDIDATES, default="default")
    p.add_argument("--result", type=Path)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--layers", help="Optional captured layer subset, e.g. 7")
    a = p.parse_args()
    if min(a.warmup, a.iterations, a.rounds, a.timeout_seconds) < 1:
        p.error("counts and timeout must be positive")
    if a.worker:
        if len(a.artifact) != 1 or (not a.profile and a.result is None):
            p.error("worker requires one artifact and --result (unless profiling)")
        return worker(a)
    if a.output is None or a.profile:
        p.error("controller requires --output; profiling uses --worker --profile")
    candidates = a.candidates.split(",")
    if any(c not in CANDIDATES or c == "default" for c in candidates):
        p.error("invalid candidates; default controls are inserted automatically")
    manifests = [json.loads((r / "manifest.json").read_text()) for r in a.artifact]
    if any(not m.get("complete") or m.get("precision") != "bf16" for m in manifests):
        p.error("all artifacts must be completed BF16 captures")
    if {m["requests"] for m in manifests} != {128, 256}:
        p.error("this experiment requires both 128 and 256 request captures")
    config = dict(
        artifacts=[str(x) for x in a.artifact],
        candidates=candidates,
        warmup=a.warmup,
        iterations=a.iterations,
        rounds=a.rounds,
        layers=a.layers,
    )
    config_path = a.output / "tuning-config.json"
    if a.resume:
        if not a.output.is_dir():
            p.error("resume requires an existing tuning directory")
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                p.error("resume requires the original experiment configuration")
        else:
            # Legacy sweeps did not save config: only accept the original defaults.
            if a.candidates != ",".join(k for k in CANDIDATES if k != "default") or (
                a.warmup,
                a.iterations,
                a.rounds,
                a.layers,
            ) != (20, 100, 5, None):
                p.error("legacy resume requires the original default tuning settings")
            old = json.loads((a.output / "results.json").read_text())
            if {t["artifact"] for t in old} != set(config["artifacts"]):
                p.error("legacy artifacts differ from existing results")
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    config_path.write_text(json.dumps(config, indent=2))
    trials = []
    reused = 0
    for index, candidate in enumerate(["default", *candidates, "default"]):
        for case, artifact in enumerate(a.artifact):
            stem = a.output / f"{index:02d}-{candidate}-case{case}"
            result = stem.with_suffix(".json")
            if a.resume and result.exists():
                trial = json.loads(result.read_text())
                if (
                    trial.get("status") != "PASS"
                    or trial.get("candidate") != candidate
                    or validate_knobs(trial.get("knobs", {}))
                    != validate_knobs(CANDIDATES[candidate])
                ):
                    raise ValueError(f"incompatible saved trial: {result}")
                trial.update(
                    artifact=str(artifact),
                    returncode=0,
                    log=str(stem.with_suffix(".log")),
                )
                trials.append(trial)
                reused += 1
                print(
                    json.dumps(
                        dict(
                            stage="bf16_trial_reused",
                            candidate=candidate,
                            artifact=str(artifact),
                        )
                    ),
                    flush=True,
                )
                continue
            if a.resume and stem.with_suffix(".log").exists():
                old_log = stem.with_suffix(".log")
                old_log.rename(
                    old_log.with_name(old_log.name + f".interrupted-{time.time_ns()}")
                )
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=4",
                str(Path(__file__).resolve()),
                "--worker",
                "--artifact",
                str(artifact),
                "--candidate",
                candidate,
                "--result",
                str(result),
                "--warmup",
                str(a.warmup),
                "--iterations",
                str(a.iterations),
                "--rounds",
                str(a.rounds),
            ]
            if a.layers:
                cmd += ["--layers", a.layers]
            print(
                json.dumps(
                    dict(
                        stage="bf16_trial_start",
                        candidate=candidate,
                        artifact=str(artifact),
                        log=str(stem.with_suffix(".log")),
                    )
                ),
                flush=True,
            )
            with stem.with_suffix(".log").open("w") as log:
                proc = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
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
                    code = -1
            trial = (
                json.loads(result.read_text())
                if code == 0 and result.exists()
                else dict(candidate=candidate, status="FAIL")
            )
            trial.update(
                artifact=str(artifact),
                returncode=code,
                log=str(stem.with_suffix(".log")),
            )
            trials.append(trial)
            (a.output / "results.json").write_text(json.dumps(trials, indent=2))
    (a.output / "results.json").write_text(json.dumps(trials, indent=2))
    report = recommend(trials)
    if reused:
        report["provisional_eligible"] = report["eligible"]
        report["eligible"] = []
        report["resume_note"] = (
            "Reused trials may be from a different node/session. Ratios are exploratory; "
            "repeat finalists with fresh controls in one uninterrupted session before E2E selection."
        )
    (a.output / "recommendation.json").write_text(json.dumps(report, indent=2))
    if reused and (a.output / "candidate-knobs.json").exists():
        (a.output / "candidate-knobs.json").rename(
            a.output / f"candidate-knobs-before-resume-{time.time_ns()}.json"
        )
    if report["eligible"]:
        best = report["eligible"][0]["candidate"]
        (a.output / "candidate-knobs.json").write_text(
            json.dumps(CANDIDATES[best], indent=2)
        )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
