"""Summarize matched maximum-rank forward times, preserving control drift."""
import json
from pathlib import Path
from statistics import median
import sys


def summarize(root):
    arms = {}
    identity = None
    for arm in ("exp-start", "hardware", "exp-end"):
        reports = [json.loads((root/arm/"direct"/f"matched-forward-rank{rank}.json").read_text())
                   for rank in range(4)]
        rows = {(r["rank"], r["sample"]): r for report in reports for r in report["rows"]}
        keys = {k: (v["input_route_sha256"], v["logits_sha256"], v["active_requests"])
                for k, v in rows.items()}
        if identity is None:
            identity = keys
        elif keys != identity:
            raise AssertionError("Arm inputs/routes/logits/sample coverage differ")
        samples = sorted({sample for _, sample in rows})
        values = [max(rows[rank, sample]["forward_us"] for rank in range(4)) for sample in samples]
        arms[arm] = dict(max_rank_us=values, median_us=median(values))
    control = min(arms[a]["median_us"] for a in ("exp-start", "exp-end"))
    drift = abs(arms["exp-end"]["median_us"]/arms["exp-start"]["median_us"]-1)
    return dict(arms=arms, control_drift=drift,
                hardware_speedup_vs_faster_control=control/arms["hardware"]["median_us"],
                qualification="DIAGNOSTIC_ONLY", common_inputs_routes_logits=True,
                note="Fixed block continuations and common copy/select overhead; not generation throughput")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    report = summarize(root)
    (root/"matched-forward-summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
