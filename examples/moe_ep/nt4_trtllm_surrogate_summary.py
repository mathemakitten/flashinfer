"""Matched-input surrogate diagnostics. No NT4 E2E/quality equivalence claim."""

import json
from pathlib import Path
import statistics
import sys


def summarize(root):
    arms = (
        "baseline-overlap",
        "baseline-serial",
        "hybrid-serial",
        "trtllm-serial",
        "hybrid-overlap",
        "trtllm-overlap",
        "baseline-end",
    )
    keys = {
        (r, t, s, l)
        for r in range(4)
        for t in (1, 2)
        for s in range(16)
        for l in (7, 32, 68)
    }
    data = {}
    for arm in arms:
        backend = "baseline" if arm.startswith("baseline") else "direct"
        schedule = "overlap" if arm == "baseline-end" else arm.split("-")[-1]
        m = json.loads((root / arm / "summary.json").read_text())
        assert (
            m["precision"] == "mxfp8" and m["model_graphs"] and m["shortcut_overlap"]
        ), arm
        assert (
            m["branch_input_mode"] == "replay" and m["branch_backend"] == backend
        ), arm
        assert m["branch_native_output"], arm
        assert m["trtllm_relu2_surrogate"] == arm.startswith("trtllm"), arm
        assert m["mxfp8_nvls_hybrid"] == (backend == "direct"), arm
        assert m["cutlass_decode_profile_ids"] == (
            [39, 428] if backend == "baseline" else None
        ), arm
        rows = {}
        for rank in range(4):
            for line in (
                (root / arm / backend / f"branch-rank{rank}.jsonl")
                .read_text()
                .splitlines()
            ):
                row = json.loads(line)
                key = tuple(row[k] for k in ("rank", "round", "sample", "layer"))
                assert (
                    key not in rows
                    and row["schedule"] == schedule
                    and row["clock"] == "device"
                ), (arm, key)
                assert row["native_output_policy"] == "cutlass_after_join_v1", (
                    arm,
                    key,
                )
                assert row["native_output_sha256"], (arm, key)
                rows[key] = row
        assert set(rows) == keys, f"Incomplete records: {arm}"
        data[arm] = rows
    ref = data[arms[0]]
    for arm, rows in data.items():
        for key, row in rows.items():
            assert all(
                row[k] == ref[key][k]
                for k in (
                    "input_sha256",
                    "native_output_sha256",
                    "active_requests",
                    "decode_step",
                )
            ), (arm, key)
    results = []
    for arm, rows in data.items():
        for layer in (7, 32, 68):
            for rank in range(4):
                r = [v for k, v in rows.items() if k[0] == rank and k[3] == layer]
                results.append(
                    dict(
                        arm=arm,
                        layer=layer,
                        rank=rank,
                        join_us=statistics.median(v["join_us"] for v in r),
                        phase_us={
                            k: statistics.median(v["phase_us"][k] for v in r)
                            for k in r[0]["phase_us"]
                        },
                        tail_phase_us={
                            k: statistics.median(v["tail_phase_us"][k] for v in r)
                            for k in r[0]["tail_phase_us"]
                        },
                        tail_after_experts_us=statistics.median(
                            max(
                                0,
                                v["phase_offsets_us"]["mixer_post_shared"][1]
                                - v["phase_offsets_us"]["experts_combine"][1],
                            )
                            for v in r
                        ),
                    )
                )
    return dict(
        status="PASS",
        matched_records_per_arm=len(keys),
        results=results,
        note="TRTLLM-gen uses unclamped Relu2 at layers 7/32/68. This is a performance surrogate, not NT4 E2E throughput or a speed-of-light bound. Hidden inputs and native continuation outputs are hash-matched. Every arm recomputes clamped CUTLASS output after the timed join; unmeasured layers use native CUTLASS. Continuation work is excluded from branch timing but changes cache history. KV/recurrent state is not independently matched. Phase intervals include waits. TRTLLM finalizes local contributions to BF16 before FP32 NVLS reduction; MegaMoE hybrid assembles FP32 partials.",
    )


if __name__ == "__main__":
    root = Path(sys.argv[1])
    report = summarize(root)
    (root / "surrogate-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                stage="surrogate_comparison_complete",
                summary=str(root / "surrogate-summary.json"),
            )
        )
    )
