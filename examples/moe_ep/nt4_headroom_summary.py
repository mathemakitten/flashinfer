"""Summarize the headroom experiment without turning one sweep into a default."""

import json
import statistics
import sys
from pathlib import Path


def summarize(data):
    names = (
        "vllm-start",
        "clusters76-start",
        "clusters74",
        "clusters72",
        "clusters68",
        "clusters76-end",
        "vllm-end",
    )
    rows = {}
    occupancy = None
    for name in names:
        rounds = [r for r in data[name] if not r["warmup"]]
        if len(rounds) < 2:
            raise ValueError(f"{name}: need two measured rounds")
        for r in rounds:
            if (
                r.get("model_graph_vs_eager_outputs") != "EXACT"
                or not r.get("requested_batch_reached")
                or r.get("model_graph_replays") != r.get("model_decode_steps")
                or not r.get("model_graph_replays")
                or r["gc_policy"]["policy"]
                != "collect_before_round_disable_during_round"
            ):
                raise ValueError(f"{name}: correctness/occupancy/graph/GC gate failed")
            if occupancy is None:
                occupancy = r["decode_active_requests_histogram"]
            if r["decode_active_requests_histogram"] != occupancy:
                raise ValueError("decode occupancy differs; inspect before comparing")
        rows[name] = dict(
            mean_tps=statistics.mean(r["output_tokens_per_second"] for r in rounds),
            round_tps=[r["output_tokens_per_second"] for r in rounds],
        )
    drift = {
        label: rows[end]["mean_tps"] / rows[start]["mean_tps"] - 1
        for label, start, end in [
            ("vllm", "vllm-start", "vllm-end"),
            ("megamoe", "clusters76-start", "clusters76-end"),
        ]
    }
    # Compare with the faster control, rather than claiming improvement from drift.
    direct = max(rows[n]["mean_tps"] for n in ("clusters76-start", "clusters76-end"))
    vllm = max(rows[n]["mean_tps"] for n in ("vllm-start", "vllm-end"))
    for name in ("clusters74", "clusters72", "clusters68"):
        rows[name]["vs_faster_full_worker"] = rows[name]["mean_tps"] / direct - 1
        rows[name]["vs_faster_vllm"] = rows[name]["mean_tps"] / vllm - 1
    return dict(
        results=rows,
        control_drift_fraction=drift,
        note="One sweep; positive candidates require an alternating confirmation. Cross-backend output comparisons remain in the E2E summary.",
    )


if __name__ == "__main__":
    source = Path(sys.argv[1])
    result = summarize(json.loads(source.read_text()))
    source.with_name("headroom-summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
