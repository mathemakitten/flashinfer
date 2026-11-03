"""Isolated activation timing, not a fused-FC1 or end-to-end benchmark."""
import json
import statistics
import torch
import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from nt4_tanh_qualification import Activation


def main():
    torch.manual_seed(42)
    for count in (32768, 128 * 4096, 256 * 4096, 512 * 4096):
        x = (torch.randn(count, device='cuda') * 24).contiguous()
        buffers = {}
        graphs = {}
        errors = {}
        ref = (16 * torch.tanh(x.cpu().double().clamp_min(0) / 16)).square()
        for approx in (False, True):
            y, t = torch.empty_like(x), torch.empty_like(x)
            tensors = (from_dlpack(x), from_dlpack(y), from_dlpack(t))
            f = cute.compile(Activation(approx), *tensors,
                             cuda.CUstream(torch.cuda.current_stream().cuda_stream))
            def launch():
                f(*tensors, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
            for _ in range(20):
                launch()
            torch.cuda.synchronize()
            assert torch.isfinite(y).all()
            actual = y.cpu().double()
            errors[str(approx)] = {'max_abs': (actual-ref).abs().max().item(),
                                  'relative_l2': ((actual-ref).norm()/ref.norm()).item()}
            # Batch launches inside the graph so Python replay gaps do not
            # dominate the tiny kernels. All launches use the capture stream.
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(100):
                    launch()
            original = x.clone()
            x.neg_()
            launch()
            expected = y.clone()
            y.fill_(float('nan'))
            g.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(y, expected, rtol=0, atol=0)
            x.copy_(original)
            # Retain every buffer, compiled function and DLPack view for graphs.
            buffers[approx] = (y, t, tensors, f)
            graphs[approx] = g
        arms = []
        for name, approx in [('control-start', False), ('hardware', True), ('control-end', False)]:
            g = graphs[approx]
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize()
            rounds = []
            for _ in range(5):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):
                    g.replay()
                end.record()
                end.synchronize()
                rounds.append(start.elapsed_time(end))  # ms / 1000 launches * 1000 us/ms
            arms.append({'arm': name, 'median_us': statistics.median(rounds), 'rounds_us': rounds})
        print(json.dumps({'stage': 'activation_microbench', 'elements': count,
              'gpu': torch.cuda.get_device_name(), 'arms': arms, 'errors_vs_fp64': errors,
              'changed_input_graph_replay': 'EXACT',
              'note': 'Standalone global-memory input/output, one element/thread, writes activation and tanh. Excludes GEMM, quantization and overlap; not a prediction of fused-kernel speedup.'}), flush=True)


if __name__ == '__main__':
    main()
