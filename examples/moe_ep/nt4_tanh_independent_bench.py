"""Independent FC1 fragments staged once in shared memory; no recurrence."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
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
        smem = utils.SmemAllocator()
        staged = smem.allocate_tensor(cutlass.Float32,
                                      cute.make_layout((128, 32), stride=(1, 128)),
                                      byte_alignment=128)
        # Transposed shared layout gives conflict-free per-channel warp loads.
        # Global staging is paid once and is included in reported kernel time.
        for j in cutlass.range_constexpr(32):
            staged[tx, j] = x[tid * 32 + j]
        cute.arch.sync_threads()
        y = cute.make_rmem_tensor(32, cutlass.Float32)
        q = cute.make_rmem_tensor(32, cutlass.Float8E4M3FN)
        total = cute.make_rmem_tensor(32, cutlass.Float32)
        total.fill(cutlass.Float32(0))
        scale = cutlass.Float32(1)
        for iteration in range(self.repeats):
            row = (tx + iteration) % 128
            for j in cutlass.range_constexpr(32):
                c = cutlass.Float32(16) * cute.math.tanh(
                    fmax(staged[row, j], cutlass.Float32(0)) / cutlass.Float32(16),
                    approx=self.approx)
                y[j] = c * c
            if cutlass.const_expr(self.quant):
                scale = quant_sfd_row(y, q, cutlass.Float32(1), 32,
                                      cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN)
                for j in cutlass.range_constexpr(32):
                    y[j] = q[j].to(cutlass.Float32) * scale
            # Explicit rounded addition prevents checksum FMA contraction from
            # changing the activation's final multiplication rounding.
            for j in cutlass.range_constexpr(0, 32, 2):
                total[j], total[j + 1] = cute.arch.add_packed_f32x2(
                    (total[j], total[j + 1]), (y[j], y[j + 1]),
                    rnd="rn", ftz=False)
        for j in cutlass.range_constexpr(32):
            out[tid * 32 + j] = total[j]
        scales[tid * 2] = scale
        scales[tid * 2 + 1] = cutlass.Float32(0)

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
    single_pass = {}
    for quant in (False, True):
        for repeats in (1, 8, 32, 128):
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
                if repeats == 1:
                    single_pass[quant, approx] = y.clone().reshape(152, 128, 32)
                else:
                    # Independent oracle: sum separately computed single-pass
                    # fragments in the same order. Detects indexing/hoisting bugs.
                    expected_sum = torch.zeros_like(single_pass[quant, approx])
                    for iteration in range(repeats):
                        expected_sum.add_(single_pass[quant, approx].roll(-iteration, dims=1))
                    torch.testing.assert_close(y.reshape_as(expected_sum), expected_sum,
                                               rtol=0, atol=0)
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
            result = dict(stage='independent_fragment_proxy', quantized=quant, repeats=repeats,
                          ctas=152, threads=128, elements_per_thread=32, arms=arms,
                          changed_elements=int((control != hardware).sum()),
                          relative_l2=float((control.double()-hardware.double()).norm()/control.double().norm()),
                          changed_scales=int((outputs[False][1][::2] != outputs[True][1][::2]).sum()),
                          graph_replay='EXACT',
                          note='Independent 32-value fragments from saved FC1 reference inputs; 16KiB shared staging per CTA included. No recurrence. Outputs sum fragment results to retain all work; repeated sums checked against separately computed single-pass fragments. No TMEM/GEMM/overlap; final FP32 stores and dequantized checksum are synthetic. Scales refer to last fragment.')
            if repeats == 1 and not quant:
                reference = (16 * torch.tanh(x.double().clamp_min(0) / 16)).square()
                result['error_vs_fp64'] = {str(k): float((outputs[k][0].double()-reference).norm()/reference.norm()) for k in outputs}
            print(json.dumps(result), flush=True)
            (a.output / f'q{int(quant)}-r{repeats}.json').write_text(json.dumps(result, indent=2))

    import subprocess
    import re
    assembly = []
    for cubin in sorted(a.output.rglob('*.cubin')):
        resource = subprocess.run(['cuobjdump', '--dump-resource-usage', str(cubin)],
                                  capture_output=True, text=True, check=True).stdout
        sass = subprocess.run(['cuobjdump', '--dump-sass', str(cubin)],
                              capture_output=True, text=True, check=True).stdout
        assembly.append(dict(file=str(cubin.relative_to(a.output)), resources=resource,
                             instructions={op: len(re.findall(re.escape(op), sass))
                                           for op in ('MUFU.TANH', 'MUFU.EX2', 'MUFU.RCP', 'LDL', 'STL', 'LDS')}))
    (a.output / 'assembly.json').write_text(json.dumps(assembly, indent=2))
    print(json.dumps(dict(stage='complete', configurations=len(assembly))), flush=True)


if __name__ == '__main__':
    main()
