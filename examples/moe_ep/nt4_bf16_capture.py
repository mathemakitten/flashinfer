"""Untimed, rank-local real BF16 samples for isolated tuning."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


class Capture:
    def __init__(self):
        root = os.environ.get("NT4_BF16_CAPTURE_ROOT")
        self.root = Path(root) if root else None
        if self.root is not None:
            print(
                json.dumps(
                    dict(
                        stage="bf16_capture_enabled",
                        root=str(self.root),
                        module=__file__,
                        rank=dist.get_rank(),
                    )
                ),
                flush=True,
            )
        self.counts = {}
        self.layers = [
            int(x)
            for x in os.environ.get("NT4_BF16_CAPTURE_LAYERS", "7,32,68").split(",")
        ]
        self.samples = int(os.environ.get("NT4_BF16_CAPTURE_SAMPLES", "2"))
        self.requested = int(os.environ.get("NT4_E2E_REQUEST_COUNT", "128"))

    def save(self, layer, x, ids, probs, output, capacity):
        from nt4_bf16_replay import export_bf16_weights
        from nt4_capture_state import snapshot

        if self.root is None or layer.layer_number not in self.layers:
            return
        n = self.counts.get(layer.layer_number, 0)
        state = snapshot()
        if (
            n >= self.samples
            or state is None
            or state["active_requests"] != self.requested
        ):
            return
        directory = self.root / f"layer{layer.layer_number}" / f"rank{dist.get_rank()}"
        directory.mkdir(parents=True, exist_ok=True)
        if n == 0:
            weights = {
                k: v.cpu() for k, v in export_bf16_weights(layer.experts).items()
            }
            torch.save(weights, directory / "weights.pt")
        torch.save(
            dict(x=x.cpu(), ids=ids.cpu(), probs=probs.cpu(), baseline=output.cpu()),
            directory / f"sample{n}.pt",
        )
        (directory / "metadata.json").write_text(
            json.dumps(
                dict(
                    schema=1,
                    precision="bf16",
                    layer=layer.layer_number,
                    rank=dist.get_rank(),
                    world_size=dist.get_world_size(),
                    capacity=capacity,
                    buffer_rows=x.shape[0],
                    active_routed_rows=int((ids >= 0).any(dim=1).sum().item()),
                    active_requests=state["active_requests"],
                    samples=n + 1,
                ),
                indent=2,
            )
        )
        self.counts[layer.layer_number] = n + 1

    def finish(self):
        if self.root is None:
            return
        error = (
            None
            if all(self.counts.get(layer, 0) == self.samples for layer in self.layers)
            else str(self.counts)
        )
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(f"incomplete BF16 tuning capture: {errors}")
        dist.barrier()
        if dist.get_rank() == 0:
            (self.root / "manifest.json").write_text(
                json.dumps(
                    dict(
                        complete=True,
                        schema=1,
                        world_size=4,
                        precision="bf16",
                        layers=self.layers,
                        samples=self.samples,
                        requests=self.requested,
                    ),
                    indent=2,
                )
            )
