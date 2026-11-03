"""Four-rank NVTX capture/export smoke test. No model/checkpoint dependencies.

Run under nsys with --trace=nvtx,osrt --capture-range=nvtx
--nvtx-capture=nt4_profile_generation and NSYS_NVTX_PROFILER_REGISTER_ONLY=0.
This checks the collection path only; it does not reproduce NVSHMEM or NT4.
"""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    x = torch.ones(1024, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            x.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        x.add_(1)
    torch.cuda.synchronize()
    dist.monitored_barrier(wait_all_ranks=True)
    if rank == 0:
        torch.cuda.nvtx.range_push("nt4_profile_generation")
    dist.monitored_barrier(wait_all_ranks=True)
    for step in range(16):
        with torch.cuda.nvtx.range(f"smoke_rank{rank}_step{step}"):
            graph.replay()
    torch.cuda.synchronize()
    dist.monitored_barrier(wait_all_ranks=True)
    if rank == 0:
        torch.cuda.nvtx.range_pop()
    dist.monitored_barrier(wait_all_ranks=True)
    print(json.dumps(dict(stage="profile_smoke_pass", rank=rank)), flush=True)
    graph.reset()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
