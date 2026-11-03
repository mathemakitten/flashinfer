"""Pair branch timings only when the live block inputs match byte for byte."""

import json
import statistics
import sys
from pathlib import Path


def summarize(root):
    data = {}
    for clusters in (76, 56):
        for schedule in ("serial", "overlap"):
            folder = root / f"clusters{clusters}-{schedule}" / "direct"
            rows = {}
            for rank in range(4):
                path = folder / f"branch-rank{rank}.jsonl"
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    key = (row["rank"], row["round"], row["sample"], row["layer"])
                    if key in rows:
                        raise ValueError(f"duplicate sample {key}")
                    rows[key] = row
            if not rows:
                raise ValueError(f"no samples in {folder}")
            data[clusters, schedule] = rows
    expected = {
        (rank, round_, sample, layer)
        for rank in range(4)
        for round_ in (1, 2)
        for sample in range(16)
        for layer in (7, 32, 68)
    }
    if any(set(rows) != expected for rows in data.values()):
        raise ValueError(
            "expected 16 sampled steps in each of two rounds on all ranks/layers"
        )
    control = data[76, "serial"]
    if any(set(rows) != set(control) for rows in data.values()):
        raise ValueError("incomplete or unmatched timing samples")
    mismatches = {
        str(k): sum(
            row["input_sha256"] != control[key]["input_sha256"]
            for key, row in rows.items()
        )
        for k, rows in data.items()
    }
    overlap_evidence = {}
    for arm, rows in data.items():
        durations = []
        for row in rows.values():
            offsets = row["phase_offsets_us"]
            durations.append(
                sum(
                    max(
                        0,
                        min(offsets[a][1], offsets[b][1])
                        - max(offsets[a][0], offsets[b][0]),
                    )
                    for a in ("route_dispatch", "prepare", "experts_combine")
                    for b in ("projection", "mixer_core", "mixer_post_shared")
                )
            )
        overlap_evidence[str(arm)] = dict(
            records=len(durations),
            records_with_overlap=sum(v > 0 for v in durations),
            median_overlap_us=statistics.median(durations),
            max_overlap_us=max(durations),
            clocks=sorted({r.get("clock", "events") for r in rows.values()}),
        )
    observed_overlap = all(
        overlap_evidence[str((clusters, "overlap"))]["records_with_overlap"] > 0
        for clusters in (76, 56)
    )
    results = []
    for clusters in (76, 56):
        for layer in (7, 32, 68):
            # Keep rank-level observations; do not treat them as independent trials.
            for rank in range(4):
                keys = [k for k in control if k[0] == rank and k[3] == layer]
                phases = {}
                for metric in ("moe_phase_sum_us", "mixer_phase_sum_us", "join_us"):
                    isolated = statistics.median(
                        data[clusters, "serial"][k][metric] for k in keys
                    )
                    overlap = statistics.median(
                        data[clusters, "overlap"][k][metric] for k in keys
                    )
                    phases[metric] = dict(
                        serial=isolated, overlap=overlap, ratio=overlap / isolated
                    )
                tail = {}
                for schedule in ("serial", "overlap"):
                    sampled = [data[clusters, schedule][k] for k in keys]
                    if any("tail_phase_us" in r for r in sampled):
                        if not all("tail_phase_us" in r for r in sampled):
                            raise ValueError("incomplete tail timing data")
                        names = sampled[0]["tail_phase_us"]
                        tail[schedule] = {
                            name: statistics.median(
                                r["tail_phase_us"][name] for r in sampled
                            )
                            for name in names
                        }
                        tail[schedule]["shared_fc1_to_fc2_gap"] = statistics.median(
                            r["shared_fc1_to_fc2_gap_us"] for r in sampled
                        )
                results.append(
                    dict(
                        clusters=clusters,
                        layer=layer,
                        rank=rank,
                        timings=phases,
                        tail_median_us=tail,
                    )
                )
    report = dict(
        overlap_evidence=overlap_evidence,
        overlap_observed_in_both_caps=observed_overlap,
        interpretation=(
            "Overlap observed; inspect per-layer intervals before interpreting contention."
            if observed_overlap
            else "No overlap demonstrated for one or both caps. Ratios cannot establish absence of contention; validate instrumentation and graph dependencies."
        ),
        input_mismatches=mismatches,
        valid_matched_inputs=not any(mismatches.values()),
        results=results,
        scope="Diagnostic phase intervals; serialized phases are the uncontended control, not standalone full-branch kernels. Timestamp/snapshot overhead is present; phase intervals are not individual kernel residency measurements. E2E throughput is not usable.",
    )
    (root / "branch-summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["valid_matched_inputs"]:
        raise SystemExit(
            "Inputs differ across schedules/caps; do not infer contention from these ratios"
        )


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
