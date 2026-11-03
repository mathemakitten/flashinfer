"""Checkpoint-free EP4 hybrid graph lifetime diagnostic using a saved artifact.

Capture every bucket before replaying any of them. Include a side-stream fork
and join, reset validation graphs, and run allocator churn between replays.
This isolates graph lifetimes; it is not an E2E correctness or speed benchmark.
"""

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from nt4_checkpoint_artifact import SCHEMA, load_payload, validate_sample
from nt4_nvls_hybrid import NvlsHybrid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--mcore-root", type=Path, required=True)
    p.add_argument("--shared-pool", action="store_true")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--replays", type=int, default=8)
    a = p.parse_args()
    if a.layers < 1 or a.replays < 1:
        p.error("layers and replays must be positive")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    if dist.get_world_size() != 4:
        raise ValueError("requires four ranks")
    rank = dist.get_rank()

    def emit(stage, **fields):
        print(json.dumps(dict(stage=stage, rank=rank, **fields)), flush=True)

    manifest = json.loads((a.artifact / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or not manifest.get("capture_complete"):
        raise ValueError("requires a completed NT4 capture")
    part = manifest["ranks"][rank]
    weights = load_payload(a.artifact / f"weights-rank{rank}.pt", part["weights"])
    weights = {k: v.cuda() for k, v in weights.items()}
    sample = load_payload(a.artifact / f"sample0-rank{rank}.pt", part["samples"][0])
    validate_sample(sample)
    active = (sample["ids"] >= 0).any(dim=1)
    if not active.any():
        raise ValueError("sample has no routed rows")
    sample = {k: v[active].cuda() for k, v in sample.items()}
    side = torch.cuda.Stream()
    capture_stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle() if a.shared_pool else None
    buckets = []
    packed = None
    knobs = dict(
        mma_tiler_mnk=(128, 256, 128),
        cluster_shape_mnk=(1, 1, 1),
        use_2cta_instrs=False,
        group_hint=1024,
        flag_batch=1,
    )
    for rows in (32, 16, 2, 1):
        indices = torch.arange(rows, device="cuda") % sample["x"].shape[0]
        inputs = {k: sample[k][indices].clone() for k in ("x", "ids", "probs")}
        hybrids = []
        for _ in range(a.layers):
            hybrid = NvlsHybrid(
                SimpleNamespace(
                    **inputs, capacity=rows, rank=rank,
                    group=dist.group.WORLD, weights=weights, knobs=knobs,
                ),
                a.mcore_root,
                packed_weights=packed,
            )
            packed = hybrid.weights
            hybrids.append(hybrid)
        # Bind each bucket's list; later construction must not change closures.
        def run(hybrids=hybrids, inputs=inputs):
            main_stream = torch.cuda.current_stream()
            side.wait_stream(main_stream)
            with torch.cuda.stream(side):
                outputs = [hybrid.run() for hybrid in hybrids]
            # Independent main-stream work and explicit join, as in ShortcutMoE.
            mixer = inputs["x"].square()
            main_stream.wait_stream(side)
            return torch.stack(outputs).sum(dim=0) + mixer

        for _ in range(3):
            expected = run().clone()
        torch.cuda.synchronize()
        dist.barrier()
        validation = torch.cuda.CUDAGraph()
        with torch.cuda.graph(validation, stream=capture_stream):
            validated = run()
        validation.replay()
        torch.testing.assert_close(validated, expected, rtol=0, atol=0)
        validation.reset()
        del validated, validation
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream, pool=pool):
            output = run()
        buckets.append(dict(
            rows=rows, hybrids=hybrids, inputs=inputs, graph=graph,
            output=output, expected=expected, run=run,
        ))
        emit("bucket_captured", rows=rows, layers=a.layers)

    # Exercise a large bucket again after all smaller buckets have been built.
    for iteration in range(a.replays):
        gc.collect()
        churn = torch.empty(8 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        churn.fill_(165)
        del churn
        for bucket in buckets:
            bucket["graph"].replay()
            torch.testing.assert_close(
                bucket["output"], bucket["expected"], rtol=0, atol=0
            )
            # Changed live routing must affect replay; then restore it exactly.
            ids = bucket["inputs"]["ids"]
            saved = ids.clone()
            ids.fill_(-1)
            masked = bucket["run"]().clone()
            bucket["graph"].replay()
            torch.testing.assert_close(bucket["output"], masked, rtol=0, atol=0)
            ids.copy_(torch.where(saved >= 0, (saved + 127) % 512, saved))
            rerouted = bucket["run"]().clone()
            bucket["graph"].replay()
            torch.testing.assert_close(bucket["output"], rerouted, rtol=0, atol=0)
            ids.copy_(saved)
            bucket["graph"].replay()
            torch.testing.assert_close(
                bucket["output"], bucket["expected"], rtol=0, atol=0
            )
        emit("bucket_replay_pass", iteration=iteration)
    torch.cuda.synchronize()
    for bucket in buckets:
        bucket["graph"].reset()
    for bucket in buckets:
        for hybrid in bucket["hybrids"]:
            hybrid.close()
    emit("hybrid_graph_repro", status="PASS", shared_pool=a.shared_pool)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
