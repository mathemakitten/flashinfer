"""EP4 full NT4 expert geometry smoke gate before any BF16 E2E sweep."""

import json
import os
import torch
import torch.distributed as dist
from nt4_bf16_replay import Bf16Replay


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.manual_seed(1234 + rank)
    weights = dict(
        w1=torch.randn(128, 4096, 768, device="cuda", dtype=torch.bfloat16) * 0.02,
        w2=torch.randn(128, 768, 4096, device="cuda", dtype=torch.bfloat16) * 0.02,
    )
    replay = Bf16Replay(weights, 64, dist.group.WORLD)
    try:
        for epoch in range(3):
            x = torch.randn(8, 768, device="cuda", dtype=torch.bfloat16)
            ids = (
                torch.arange(80, device="cuda").reshape(8, 10) * 13
                + rank * 37
                + epoch * 91
            ) % 512
            probs = torch.rand(8, 10, device="cuda") * 0.8
            if epoch == 1 and rank == 0:
                ids.fill_(-1)
                probs.zero_()
            if epoch == 2:
                ids[:, 1::2] = -1
                probs[:, 1::2] = 0
            replay.check(dict(x=x, ids=ids, probs=probs))
        # Capture and reuse on changed inputs, without any allocations in graph.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                replay.run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = replay.run()
        for epoch in range(3):
            replay.x.normal_(0, 0.25 + epoch * 0.25)
            replay.ids[:8] = (
                torch.arange(80, device="cuda").reshape(8, 10) + rank * 71 + epoch * 101
            ) % 512
            replay.ids[8:].fill_(-1)
            if epoch == 2 and rank == 0:
                replay.ids.fill_(-1)
            expected = replay.run().clone()
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.cuda.synchronize()
        graph.reset()
        print(
            json.dumps(
                dict(
                    stage="bf16_bringup",
                    rank=rank,
                    status="PASS",
                    eager=True,
                    graph=True,
                    empty_rank=True,
                    changing_routes=True,
                )
            ),
            flush=True,
        )
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        replay.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
