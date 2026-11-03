"""FC1-shaped register-fragment proxy; excludes TMEM, GEMM, stores and overlap."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/src'))
from common.moe_utils import fmax, quant_sfd_row


class Fragment:
    def __init__(self, approx, quant, repeats):
        self.approx, self.quant, self.repeats = approx, quant, repeats

    @cute.kernel
    def kernel(self, x: cute.Tensor, out: cute.Tensor, scales: cute.Tensor):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        tid = bx * 128 + tx
        z = cute.make_rmem_tensor(32, cutlass.Float32)
        y = cute.make_rmem_tensor(32, cutlass.Float32)
        q = cute.make_rmem_tensor(32, cutlass.Float8E4M3FN)
        for j in cutlass.range_constexpr(32):
            z[j] = x[tid * 32 + j]
        checksum = cutlass.Float32(0)
        scale = cutlass.Float32(1)
        for iteration in range(self.repeats):
            for j in cutlass.range_constexpr(32):
                c = cutlass.Float32(16) * cute.math.tanh(
                    fmax(z[j], cutlass.Float32(0)) / cutlass.Float32(16),
                    approx=self.approx)
                y[j] = c * c
                # Register recurrence prevents loop-invariant hoisting. This
                # is a throughput proxy, NOT repeated real FC1 invocations.
                z[j] = y[j] * cutlass.Float32(0.0625) + cutlass.Float32(0.25)
            if cutlass.const_expr(self.quant):
                scale = quant_sfd_row(y, q, cutlass.Float32(1), 32,
                                      cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN)
                for j in cutlass.range_constexpr(32):
                    checksum += q[j].to(cutlass.Float32)
            else:
                scale = cutlass.Float32(1)
            # All output components remain observable, including quantization.
        for j in cutlass.range_constexpr(32):
            if cutlass.const_expr(self.quant):
                out[tid * 32 + j] = q[j].to(cutlass.Float32) * scale
            else:
                out[tid * 32 + j] = y[j]
        scales[tid * 2] = scale
        scales[tid * 2 + 1] = checksum

    @cute.jit
    def __call__(self, x: cute.Tensor, out: cute.Tensor, scales: cute.Tensor, stream: cuda.CUstream):
        self.kernel(x, out, scales).launch(grid=(x.shape[0] // 4096, 1, 1),
                                          block=(128, 1, 1), stream=stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    snapshot = torch.load(a.snapshot, map_location='cpu', weights_only=True)
    # Independent FC1 accumulators, not actual kernel accumulators.
    source = snapshot['z_reference'].float().flatten()
    count = 152 * 128 * 32
    x = source.repeat((count + source.numel() - 1) // source.numel())[:count].cuda().contiguous()
    for quant in (False, True):
        for repeats in (1, 32, 128):
            compiled, graphs, outputs = {}, {}, {}
            for approx in (False, True):
                y = torch.empty_like(x)
                sf = torch.empty(count // 32 * 2, device='cuda')
                views = tuple(from_dlpack(t) for t in (x, y, sf))
                dump = a.output / f'q{int(quant)}-r{repeats}-a{int(approx)}'
                dump.mkdir()
                f = cute.compile(Fragment(approx, quant, repeats), *views,
                                 cuda.CUstream(torch.cuda.current_stream().cuda_stream),
                                 options=f'--keep-ptx --keep-cubin --dump-dir={dump}')
                def launch():
                    f(*views, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
                launch()
                torch.cuda.synchronize()
                assert torch.isfinite(y).all() and torch.isfinite(sf).all()
                outputs[approx] = (y.clone(), sf.clone())
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(10):
                        launch()
                original = x.clone()
                x.neg_()
                launch()
                expected = (y.clone(), sf.clone())
                y.fill_(float('nan'))
                sf.fill_(float('nan'))
                g.replay()
                torch.cuda.synchronize()
                for got, want in zip((y, sf), expected):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
                x.copy_(original)
                compiled[approx] = (f, views, y, sf)
                graphs[approx] = g
            arms = []
            for name, approx in [('control-start', False), ('hardware', True), ('control-end', False)]:
                g = graphs[approx]
                for _ in range(5):
                    g.replay()
                times = []
                for _ in range(5):
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    for _ in range(10):
                        g.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end) * 10)  # 100 kernel launches
                arms.append(dict(arm=name, median_us=statistics.median(times), rounds_us=times))
            control, hardware = outputs[False][0], outputs[True][0]
            result = dict(stage='register_fragment_proxy', quantized=quant, repeats=repeats,
                          ctas=152, threads=128, elements_per_thread=32, arms=arms,
                          changed_elements=int((control != hardware).sum()),
                          relative_l2=float((control.double()-hardware.double()).norm()/control.double().norm()),
                          changed_scales=int((outputs[False][1][::2] != outputs[True][1][::2]).sum()),
                          graph_replay='EXACT',
                          note='Same quant_sfd_row helper and scalar activation sequence; independent FC1 reference inputs. Register recurrence and checksum are synthetic. No TMEM, routing, GEMM or overlap. Only repeats=1 represents original activations.')
            if repeats == 1 and not quant:
                reference = (16 * torch.tanh(x.double().clamp_min(0) / 16)).square()
                result['error_vs_fp64'] = {str(k): float((outputs[k][0].double()-reference).norm()/reference.norm()) for k in outputs}
            print(json.dumps(result), flush=True)
            (a.output / f'q{int(quant)}-r{repeats}.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
