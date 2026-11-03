"""Calibrate branch probes on two independent one-CTA workloads; no checkpoint."""

import argparse
import json
import os
from pathlib import Path

import torch

from nt4_branch_clock import busy
from nt4_branch_probe import BranchProbe, PHASES, MOE


def intersection(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["NT4_BRANCH_TAIL"] = "0"  # Calibration has no model tail methods.
    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        raise RuntimeError("unset CUDA_LAUNCH_BLOCKING before calibrating concurrency")
    main_stream = torch.cuda.Stream()
    side = torch.cuda.Stream(priority=-1)
    ready, done = torch.cuda.Event(), torch.cuda.Event()
    x = torch.zeros((1, 1), device="cuda")
    intrinsic = torch.empty(4, dtype=torch.int64, device="cuda")
    busy(intrinsic, 0)
    busy(intrinsic, 2)
    torch.cuda.synchronize()
    records = []
    for clock in ("events", "device"):
        os.environ["NT4_BRANCH_CLOCK"] = clock
        for schedule in ("serial", "overlap"):
            probe = BranchProbe(0, schedule)

            def launch():
                probe.begin(x)
                ready.record(main_stream)
                side.wait_event(ready)
                for name in PHASES:
                    with torch.cuda.stream(side if name in MOE else main_stream):
                        probe.start(name)
                        if name == "route_dispatch":
                            busy(intrinsic, 0)
                        elif name == "projection":
                            busy(intrinsic, 2)
                        probe.end(name)
                done.record(side)
                main_stream.wait_event(done)
                probe.join()

            for graph_mode in (False, True):
                with torch.cuda.stream(main_stream):
                    launch()
                torch.cuda.synchronize()
                graph = None
                if graph_mode:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=main_stream):
                        launch()
                for repeat in range(3):
                    with torch.cuda.stream(main_stream):
                        graph.replay() if graph_mode else launch()
                    torch.cuda.synchronize()
                    row = probe.result()
                    ticks = intrinsic.cpu().tolist()
                    actual = [
                        [(t - ticks[0]) / 1000 for t in ticks[i : i + 2]]
                        for i in (0, 2)
                    ]
                    reported = intersection(
                        row["phase_offsets_us"]["route_dispatch"],
                        row["phase_offsets_us"]["projection"],
                    )
                    records.append(
                        dict(
                            clock=clock,
                            schedule=schedule,
                            graph=graph_mode,
                            repeat=repeat,
                            intrinsic_overlap_us=intersection(*actual),
                            reported_overlap_us=reported,
                            join_us=row["join_us"],
                        )
                    )
                    print(json.dumps(records[-1]), flush=True)
    # This checks concurrency independently inside the workload kernels.
    device_rows = [r for r in records if r["clock"] == "device" and r["graph"]]
    passed = all(
        (
            (r["intrinsic_overlap_us"] > 500 and r["reported_overlap_us"] > 500)
            if r["schedule"] == "overlap"
            else (r["intrinsic_overlap_us"] == 0 and r["reported_overlap_us"] == 0)
        )
        for r in device_rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(status="PASS" if passed else "FAIL", records=records), indent=2)
    )
    if not passed:
        raise SystemExit(
            "Device timestamp calibration failed; do not run the model study."
        )


if __name__ == "__main__":
    main()
