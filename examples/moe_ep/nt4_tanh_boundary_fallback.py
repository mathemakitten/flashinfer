"""Standalone block fallback prototype, not integrated into MegaMoE.

The activation interval is an empirical candidate envelope, not a proven bound.
Report containment violations and exact quantized mismatches; never qualify on
speed alone. The proxy includes three block quantizations and omits GEMM/TMEM.
"""
import argparse
import json
from pathlib import Path
import statistics

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from nt4_tanh_register_bench import Fragment, fmax, quant_sfd_row


class BoundaryFallback:
    def __init__(self, relative):
        self.relative = relative

    @cute.kernel
    def kernel(self, x: cute.Tensor, out: cute.Tensor, scales: cute.Tensor):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        tid = bx * 128 + tx
        y = cute.make_rmem_tensor(32, cutlass.Float32)
        low = cute.make_rmem_tensor(32, cutlass.Float32)
        high = cute.make_rmem_tensor(32, cutlass.Float32)
        q = cute.make_rmem_tensor(32, cutlass.Float8E4M3FN)
        qlow = cute.make_rmem_tensor(32, cutlass.Float8E4M3FN)
        qhigh = cute.make_rmem_tensor(32, cutlass.Float8E4M3FN)
        for j in cutlass.range_constexpr(32):
            z = fmax(x[tid * 32 + j], cutlass.Float32(0))
            c = cutlass.Float32(16) * cute.math.tanh(z / cutlass.Float32(16), approx=True)
            y[j] = c * c
            radius = y[j] * cutlass.Float32(self.relative) + cutlass.Float32(1e-6)
            low[j] = fmax(y[j] - radius, cutlass.Float32(0))
            high[j] = y[j] + radius
        slo = quant_sfd_row(low, qlow, cutlass.Float32(1), 32,
                            cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN)
        shi = quant_sfd_row(high, qhigh, cutlass.Float32(1), 32,
                            cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN)
        ambiguous = cutlass.Int32(0)
        if slo != shi:
            ambiguous = cutlass.Int32(1)
        for j in cutlass.range_constexpr(32):
            if qlow[j].to(cutlass.Float32) != qhigh[j].to(cutlass.Float32):
                ambiguous = cutlass.Int32(1)
        # Whole-block fallback also handles uncertainty in the shared E8M0 scale.
        if ambiguous != 0:
            for j in cutlass.range_constexpr(32):
                z = fmax(x[tid * 32 + j], cutlass.Float32(0))
                c = cutlass.Float32(16) * cute.math.tanh(z / cutlass.Float32(16), approx=False)
                y[j] = c * c
        scale = quant_sfd_row(y, q, cutlass.Float32(1), 32,
                              cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN)
        for j in cutlass.range_constexpr(32):
            out[tid * 32 + j] = q[j].to(cutlass.Float32) * scale
        scales[tid * 2] = scale
        scales[tid * 2 + 1] = ambiguous.to(cutlass.Float32)

    @cute.jit
    def __call__(self, x: cute.Tensor, out: cute.Tensor, scales: cute.Tensor, stream: cuda.CUstream):
        self.kernel(x, out, scales).launch(grid=(x.shape[0] // 4096, 1, 1), block=(128, 1, 1), stream=stream)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in a.snapshot:
        snap = torch.load(path, map_location='cpu', weights_only=True)
        for key in ('z_reference', 'z_reference_fp64'):
            sources.append((str(path) + ':' + key, snap[key].float().flatten()))
    g = torch.Generator().manual_seed(8137)
    sources.append(('random', torch.randn(262144, generator=g) * 24))
    # FP8 midpoint targets plus block-scale transitions; vary surrounding values.
    positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mids = (positive[:-1] + positive[1:]) / 2
    target = torch.cat([mids * (2.0 ** k) for k in range(-12, 1)] +
                       [448 * torch.pow(2.0, torch.arange(-20, 0).float())])
    target = target[(target > 0) & (target < 256)]
    target = (target[:, None] * (1 + torch.tensor([-1e-5, -1e-6, 0., 1e-6, 1e-5]))).flatten()
    z = 16 * torch.atanh(target.double().sqrt() / 16)
    blocks = torch.rand((z.numel(), 32), generator=g) * z.float()[:, None]
    blocks[:, 0] = z.float()
    sources.append(('midpoint_and_scale_stress', blocks.flatten()))
    records = []
    for name, src in sources:
        n = src.numel(); size = ((n + 4095) // 4096) * 4096
        x = torch.zeros(size, device='cuda'); x[:n].copy_(src)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        arms = {}
        for label, obj in [('exponential', Fragment(False, True, 1)), ('hardware', Fragment(True, True, 1)),
                           ('exp_activation', Fragment(False, False, 1)), ('hw_activation', Fragment(True, False, 1))] + [
                           (f'fallback-{rel}', BoundaryFallback(rel)) for rel in (1e-5, 1e-4, 1e-3)]:
            y = torch.empty_like(x); sf = torch.empty(size // 16, device='cuda')
            views = tuple(from_dlpack(t) for t in (x, y, sf))
            fn = cute.compile(obj, *views, stream)
            fn(*views, cuda.CUstream(torch.cuda.current_stream().cuda_stream)); torch.cuda.synchronize()
            assert bool(torch.isfinite(y).all() and torch.isfinite(sf).all())
            initial = (y.clone(), sf.clone())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn(*views, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
            original = x.clone(); x.neg_(); fn(*views, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
            expected = (y.clone(), sf.clone()); y.fill_(float('nan')); sf.fill_(float('nan'))
            graph.replay(); torch.cuda.synchronize()
            for got, want in zip((y, sf), expected):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            x.copy_(original); graph.replay(); torch.cuda.synchronize()
            torch.testing.assert_close(y, initial[0], rtol=0, atol=0)
            arms[label] = (initial, graph, (fn, views, y, sf))
        exp = arms['exponential'][0]; hw = arms['hardware'][0]
        ea, ha = arms['exp_activation'][0][0], arms['hw_activation'][0][0]
        for rel in (1e-5, 1e-4, 1e-3):
            label = f'fallback-{rel}'; got = arms[label][0]
            accepted = got[1][1::2] == 0
            radius = ha.abs() * rel + 1e-6
            contained = ((ea-ha).abs() <= radius)
            bad = (got[0] != exp[0]).reshape(-1, 32).any(-1) | (got[1][::2] != exp[1][::2])
            record = dict(source=name, relative_envelope=rel, absolute_envelope=1e-6,
                real_elements=n, blocks=size//32, fallback_blocks=int((~accepted).sum()),
                envelope_violations=int((~contained[:n]).sum()),
                quantized_mismatched_blocks=int(bad.sum()), unsafe_accepted_blocks=int((accepted & bad).sum()),
                hardware_changed_elements=int((hw[0][:n] != exp[0][:n]).sum()),
                status='EMPIRICAL_PASS' if not bool(bad.any()) and bool(contained[:n].all()) else 'FAIL',
                scope='Standalone activation/quantization proxy; empirical envelope, not a proven bound or E2E qualification')
            # Bracket proxy timing, do not claim integrated kernel speed.
            times = {}
            for tag, key in [('exp_start','exponential'), ('hardware','hardware'), ('fallback',label), ('exp_end','exponential')]:
                graph = arms[key][1]
                for _ in range(5): graph.replay()
                rounds = []
                for _ in range(3):
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(50): graph.replay()
                    end.record(); end.synchronize(); rounds.append(start.elapsed_time(end)*1000/50)
                times[tag] = statistics.median(rounds)
            record['proxy_us'] = times
            records.append(record); print(json.dumps(record), flush=True)
    (a.output/'summary.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
