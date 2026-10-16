"""CPU helpers and saved-artifact token census for NT4 checkpoint sweeps."""

import argparse
import json
from pathlib import Path

import torch

from nt4_checkpoint_artifact import load_payload


def token_census(sample):
    ids, probs = sample["ids"], sample["probs"]
    valid = ids >= 0
    active = valid.any(-1)
    return dict(
        buffer_rows=int(ids.shape[0]),
        active_routed_rows=int(active.sum()),
        positive_weight_rows=int((valid & (probs > 0)).any(-1).sum()),
        padding_rows=int((~active).sum()),
        valid_expert_slots=int(valid.sum()),
        unique_experts=int(ids[valid].unique().numel()),
        expert_assignments=torch.bincount(ids[valid].long(), minlength=512).tolist(),
        assignments_per_owner=torch.bincount(
            (ids[valid] // 128).long(), minlength=4
        ).tolist(),
    )


def sweep_points(counts, buffer_rows, buckets=None):
    """Map active counts to the smallest declared buffer bucket that fits."""
    sizes = [buffer_rows] if buckets is None else buckets
    if not sizes or any(n < 1 for n in sizes) or sizes != sorted(set(sizes)):
        raise ValueError("sweep buckets must be unique, positive, and increasing")
    if (
        not counts
        or len(set(counts)) != len(counts)
        or any(n < 1 or n > sizes[-1] for n in counts)
    ):
        raise ValueError(
            "sweep counts must be unique positive counts within the largest buffer bucket"
        )
    return [(n, next(b for b in sizes if b >= n)) for n in counts]


def make_sweep_sample(pool, active_rows, buffer_rows, rank):
    """Repeat valid captured rows; do not fabricate routes for padding rows."""
    if not 0 <= active_rows <= buffer_rows:
        raise ValueError("active rows must fit buffer rows")
    keep = (pool["ids"] >= 0).any(-1)
    available = int(keep.sum())
    if active_rows and not available:
        raise ValueError("no active captured rows available for sweep")
    result = {
        "x": torch.zeros(buffer_rows, 768, dtype=pool["x"].dtype),
        "ids": torch.full((buffer_rows, 10), -1, dtype=torch.int64),
        "probs": torch.zeros(buffer_rows, 10, dtype=torch.float32),
    }
    if active_rows:
        indices = (torch.arange(active_rows) + rank * active_rows) % available
        for key in result:
            result[key][:active_rows] = pool[key][keep][indices]
    result["probs"].masked_fill_(result["ids"] < 0, 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.artifact / "manifest.json").read_text())
    for rank, part in enumerate(manifest["ranks"]):
        for index, hashes in enumerate(part["samples"]):
            sample = load_payload(
                args.artifact / f"sample{index}-rank{rank}.pt", hashes
            )
            print(json.dumps(dict(rank=rank, sample=index, **token_census(sample))))


if __name__ == "__main__":
    main()
