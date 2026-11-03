"""Compare capped FC1/FC2 against uncapped CUTLASS in fresh processes.

Run before the overlap sweep. This tests GB300 BF16 ClampedRelu2, EP-local
experts, padded routes, and graph replay. No checkpoint is loaded.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--child", type=int)
    a = p.parse_args()
    if a.child is None:
        a.output.mkdir(parents=True, exist_ok=False)
        for cap in (0, 64, 128, 304):
            env = dict(os.environ, NT4_CUTLASS_MAX_BLOCKS=str(cap))
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--output",
                    str(a.output),
                    "--child",
                    str(cap),
                ],
                env=env,
                check=True,
            )
        import torch

        control = torch.load(a.output / "cap0.pt", weights_only=True)
        for cap in (64, 128, 304):
            actual = torch.load(a.output / f"cap{cap}.pt", weights_only=True)
            for expected, result in zip(control, actual):
                torch.testing.assert_close(result, expected, rtol=0.01, atol=0.01)
        print("PASS: capped CUTLASS matches uncapped for all smoke cases", flush=True)
        return

    import torch
    from flashinfer.fused_moe import ActivationType, cutlass_fused_moe

    assert torch.cuda.get_device_capability() == (10, 3), "GB300 test only"
    torch.manual_seed(1234)
    device = "cuda"
    dtype = torch.bfloat16
    h, f, experts, topk = 768, 4096, 16, 10
    w1 = torch.randn(experts, f, h, device=device, dtype=dtype) / h**0.5
    w2 = torch.randn(experts, h, f, device=device, dtype=dtype) / f**0.5
    clamp = torch.tensor([16.0], device=device, dtype=torch.float32)
    results = []
    for rows in (128, 512):
        x = torch.randn(rows, h, device=device, dtype=dtype)
        routes = (
            torch.rand(rows, experts * 4, device=device)
            .argsort(dim=1)[:, :topk]
            .int()
            .contiguous()
        )
        routes[-7:] = -1
        weights = torch.full((rows, topk), 0.316, device=device)
        weights[-7:] = 0

        def run():
            return cutlass_fused_moe(
                x,
                routes,
                weights,
                w1,
                w2,
                dtype,
                quant_scales=None,
                ep_size=4,
                ep_rank=0,
                activation_type=ActivationType.ClampedRelu2,
                clamped_relu2_limit=clamp,
            )[0]

        for _ in range(3):
            expected = run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
        results.append(actual.cpu())
        print(f"cap={a.child} rows={rows} graph/eager PASS", flush=True)
    torch.save(results, a.output / f"cap{a.child}.pt")


if __name__ == "__main__":
    main()
