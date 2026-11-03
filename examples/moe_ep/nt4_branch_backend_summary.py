"""Compare native experts and MegaMoE shortcut timings on recorded block inputs.

Input replacement/copy and CPU I/O are outside the phase timer. These perturb
execution/cache history; use this experiment for attribution, never E2E speed.
Matching hidden states alone does not certify all recurrent/KV cache contents.
"""

import argparse
import json
import statistics
from pathlib import Path

ARMS = ("vllm-overlap", "direct-overlap", "direct-serial", "vllm-serial")


def summarize(
    root, precision="bf16", worker_sweep=False, pipeline_sweep=False, hybrid_sweep=False, trtllm_matched=False
):
    if precision not in ("bf16", "mxfp8"):
        raise ValueError(f"unsupported precision: {precision}")
    native = "baseline" if precision == "mxfp8" else "vllm"
    arms = (f"{native}-overlap", "direct-overlap", "direct-serial", f"{native}-serial")
    if sum((worker_sweep, pipeline_sweep, hybrid_sweep)) > 1:
        raise ValueError("select only one resource experiment")
    if pipeline_sweep:
        if precision != "mxfp8":
            raise ValueError("pipeline sweep requires MXFP8")
        arms = (
            "baseline-overlap",
            "baseline-serial",
            *(
                f"directstage{reduction}-{schedule}"
                for reduction in (0, 1, 2)
                for schedule in ("serial", "overlap")
            ),
        )
    if worker_sweep:
        if precision != "mxfp8":
            raise ValueError("worker sweep requires MXFP8")
        arms = (
            "baseline-overlap",
            "baseline-serial",
            *(
                f"direct{cap}-{schedule}"
                for cap in (152, 144, 136)
                for schedule in ("serial", "overlap")
            ),
        )
    if hybrid_sweep:
        if precision != "mxfp8":
            raise ValueError("hybrid sweep requires MXFP8")
        arms = tuple(
            f"{backend}-{schedule}"
            for backend in ("baseline", "direct", "hybrid")
            for schedule in ("overlap", "serial")
        )
    if trtllm_matched:
        if precision != "mxfp8" or worker_sweep or pipeline_sweep or hybrid_sweep:
            raise ValueError("TRTLLM matched comparison requires MXFP8 without resource sweeps")
        arms = ("hybrid-serial", "trtllm-serial", "trtllm-overlap", "hybrid-overlap")
    expected = {
        (rank, round_, sample, layer)
        for rank in range(4)
        for round_ in (1, 2)
        for sample in range(16)
        for layer in (7, 32, 68)
    }
    data = {}
    for arm in arms:
        backend, schedule = arm.split("-")
        if backend.startswith("direct") or backend in ("hybrid", "trtllm"):
            backend = "direct"
        if trtllm_matched:
            metadata = json.loads((root / arm / "summary.json").read_text())
            if not (
                metadata["precision"] == "mxfp8"
                and metadata["mxfp8_nvls_hybrid"]
                and metadata["branch_native_output"]
                and metadata["trtllm_clamped_branch"] == arm.startswith("trtllm-")
                and metadata["branch_input_mode"] == "replay"
            ):
                raise ValueError(f"incorrect matched backend configuration: {arm}")
        if hybrid_sweep:
            metadata = json.loads((root / arm / "summary.json").read_text())
            if (
                metadata["precision"] != "mxfp8"
                or metadata["mxfp8_nvls_hybrid"] != arm.startswith("hybrid-")
                or metadata["branch_backend"] != backend
                or metadata["mxfp8_worker_clusters"] is not None
                or metadata["mxfp8_ab_stage_reduction"] is not None
                or metadata["cutlass_decode_profile_ids"]
                != ([39, 428] if backend == "baseline" else None)
                or not metadata["model_graphs"]
                or not metadata["shortcut_overlap"]
            ):
                raise ValueError(f"incorrect hybrid configuration: {arm}")
        if pipeline_sweep:
            metadata = json.loads((root / arm / "summary.json").read_text())
            reduction = (
                int(arm.split("-")[0][11:]) if arm.startswith("directstage") else None
            )
            if (
                metadata["precision"] != "mxfp8"
                or metadata["mxfp8_worker_clusters"]
                != (152 if reduction is not None else None)
                or metadata["mxfp8_ab_stage_reduction"] != reduction
                or metadata["cutlass_decode_profile_ids"]
                != ([39, 428] if reduction is None else None)
            ):
                raise ValueError(f"incorrect pipeline/tactic configuration: {arm}")
        if worker_sweep:
            metadata = json.loads((root / arm / "summary.json").read_text())
            cap = int(arm.split("-")[0][6:]) if arm.startswith("direct") else None
            if (
                metadata["precision"] != "mxfp8"
                or metadata["mxfp8_worker_clusters"] != cap
                or metadata["cutlass_decode_profile_ids"]
                != ([39, 428] if cap is None else None)
            ):
                raise ValueError(f"incorrect worker/tactic configuration: {arm}")
        rows = {}
        for rank in range(4):
            path = root / arm / backend / f"branch-rank{rank}.jsonl"
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = tuple(row[k] for k in ("rank", "round", "sample", "layer"))
                if (
                    key in rows
                    or row["schedule"] != schedule
                    or row["clock"] != "device"
                ):
                    raise ValueError(f"invalid/duplicate record: {path}, {key}")
                if row["input_mode"] != ("record" if arm == arms[0] and not trtllm_matched else "replay"):
                    raise ValueError(f"input replay was not enabled: {path}")
                rows[key] = row
        if set(rows) != expected:
            raise ValueError(f"incomplete timing records: {arm}")
        data[arm] = rows
    control = data[arms[0]]
    for arm, rows in data.items():
        for key, row in rows.items():
            if any(
                row[k] != control[key][k]
                for k in ("input_sha256", "active_requests", "decode_step")
            ):
                raise ValueError(f"unmatched inputs/step: {arm}, {key}")
    results = []
    for rank in range(4):
        for layer in (7, 32, 68):
            for arm in arms:
                rows = [
                    r for k, r in data[arm].items() if k[0] == rank and k[3] == layer
                ]
                median = lambda values: statistics.median(values)
                offsets = [r["phase_offsets_us"] for r in rows]
                results.append(
                    dict(
                        arm=arm,
                        rank=rank,
                        layer=layer,
                        join_us=median(r["join_us"] for r in rows),
                        expert_branch_done_us=median(
                            o["experts_combine"][1] for o in offsets
                        ),
                        tail_done_us=median(o["mixer_post_shared"][1] for o in offsets),
                        tail_after_experts_us=median(
                            max(0, o["mixer_post_shared"][1] - o["experts_combine"][1])
                            for o in offsets
                        ),
                        experts_after_tail_us=median(
                            max(0, o["experts_combine"][1] - o["mixer_post_shared"][1])
                            for o in offsets
                        ),
                        expert_internal={
                            name: {metric: median(r["expert_internal"][name][metric] for r in rows)
                                   for metric in ("count", "median_us", "max_us", "span_us", "first_start_us", "last_end_us")}
                            for name in rows[0].get("expert_internal", {})
                        },
                        expert_mma_timing_version=rows[0].get("expert_mma_timing_version"),
                        expert_mma_counters={
                            name: {metric: median(r["expert_mma_counters"][name][metric] for r in rows)
                                   for metric in ("median", "max")}
                            for name in rows[0].get("expert_mma_counters", {})
                        },
                        expert_phase_us={
                            name: median(r["expert_phase_us"][name] for r in rows)
                            for name in rows[0].get("expert_phase_us", {})
                        },
                        phase_us={
                            name: median(r["phase_us"][name] for r in rows)
                            for name in rows[0]["phase_us"]
                        },
                        tail_phase_us={
                            name: median(r["tail_phase_us"][name] for r in rows)
                            for name in rows[0]["tail_phase_us"]
                        },
                    )
                )
    report = dict(
        valid_matched_block_inputs=True,
        precision=precision,
        native_backend=(
            "FlashInfer CUTLASS + NVLS"
            if precision == "mxfp8"
            else "vLLM grouped GEMM + NVLS"
        ),
        samples_per_arm=len(expected),
        worker_sweep=worker_sweep,
        pipeline_sweep=pipeline_sweep,
        hybrid_sweep=hybrid_sweep,
        trtllm_matched=trtllm_matched,
        cutlass_decode_profile_ids=(
            [39, 428] if worker_sweep or pipeline_sweep or hybrid_sweep else None
        ),
        results=results,
        scope=f"{precision.upper()}, three selected blocks, recorded native hidden-state sequence on every measured decode step. Whole-model graphs. Copy/select occurs before fork; per-step synchronization and input I/O perturb execution. No E2E throughput claim. KV/recurrent state is not independently hashed; inspect mixer_core as a consistency check. Phase intervals include waiting and marker overhead. Other layers retain their backend and overlap schedule.",
    )
    (root / "branch-backend-summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--precision", choices=("bf16", "mxfp8"), default="bf16")
    parser.add_argument("--worker-sweep", action="store_true")
    parser.add_argument("--pipeline-sweep", action="store_true")
    parser.add_argument("--hybrid-sweep", action="store_true")
    parser.add_argument("--trtllm-matched", action="store_true")
    args = parser.parse_args()
    summarize(
        args.root,
        args.precision,
        args.worker_sweep,
        args.pipeline_sweep,
        args.hybrid_sweep,
        args.trtllm_matched,
    )
