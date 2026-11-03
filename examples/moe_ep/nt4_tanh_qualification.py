"""Screen hardware tanh against CPU FP64; full-model gates remain mandatory."""
import argparse
import json
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


class Activation:
    def __init__(self, approx):
        self.approx = approx

    @cute.kernel
    def kernel(self, x: cute.Tensor, y: cute.Tensor, t: cute.Tensor):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        i = bx * 128 + tx
        if i < x.shape[0]:
            z = x[i]
            if z < 0:
                z = cutlass.Float32(0)
            h = cute.math.tanh(z / cutlass.Float32(16), approx=self.approx)
            c = cutlass.Float32(16) * h
            t[i] = h
            y[i] = c * c

    @cute.jit
    def __call__(self, x: cute.Tensor, y: cute.Tensor, t: cute.Tensor, stream: cuda.CUstream):
        self.kernel(x, y, t).launch(
            grid=((x.shape[0] + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream
        )


def quantize_blocks(y):
    blocks = y.reshape(-1, 32)
    # E8M0 ceiling scale for E4M3, norm_const=1. This is a diagnostic
    # emulation, not a replacement for the kernel's independent block gates.
    scale = torch.pow(2., torch.ceil(torch.log2(blocks.abs().amax(1) / 448.)))
    scale = scale.clamp(min=2.**-126)
    return scale, (blocks / scale[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explore-approximation", action="store_true",
                        help="Report finite hardware-tanh screening exceedances without treating them as qualification")
    args = parser.parse_args()
    torch.manual_seed(0)
    # Dense range, small inputs, saturation, random inputs, and immediate
    # neighbors of inverse-mapped E4M3 midpoints at unit scale.
    fp8 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    fp8 = fp8[torch.isfinite(fp8) & (fp8 >= 0) & (fp8 < 256)].unique().sort().values
    mids = ((fp8[:-1] + fp8[1:]) / 2).double()
    edge = (16 * torch.atanh(torch.sqrt(mids) / 16)).float()
    x = torch.cat([torch.linspace(-160, 160, 262144),
                   torch.linspace(0, .01, 65536), torch.randn(65536) * 24,
                   edge, torch.nextafter(edge, torch.full_like(edge, float('inf'))),
                   torch.nextafter(edge, torch.full_like(edge, -float('inf')))])
    x = torch.cat([x, torch.zeros((-x.numel()) % 32)]).cuda()
    # Every block includes the saturated activation 256, fixing its scale at 1
    # so the inverse-mapped midpoint cases really exercise FP8 boundaries.
    x.reshape(-1, 32)[:, -1] = 512
    outputs = {}
    for approx in (False, True):
        y, t = torch.empty_like(x), torch.empty_like(x)
        tensors = (from_dlpack(x), from_dlpack(y), from_dlpack(t))
        f = cute.compile(
            Activation(approx), *tensors,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )

        def launch():
            # torch.cuda.graph selects its capture stream on context entry.
            # Resolve it at each launch, never before entering capture.
            f(*tensors, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

        launch()
        torch.cuda.synchronize()
        outputs[approx] = (y.cpu(), t.cpu())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        saved = x.clone()
        x.copy_(saved.roll(32))
        launch()
        expected = y.clone()
        y.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        x.copy_(saved)
    ref_t = torch.tanh(x.cpu().double().clamp_min(0) / 16)
    ref = (16 * ref_t).square()
    metrics = {}
    # Explicit screening tolerance, not a claim about an ISA accuracy guarantee.
    tanh_budget = 2e-6
    for approx, (y, t) in outputs.items():
        err = (t.double() - ref_t).abs()
        activation_budget = 512 * ref_t.abs() * tanh_budget + 256 * tanh_budget**2 + 4 * torch.finfo(torch.float32).eps * ref.abs()
        worst = int(err.argmax())
        print(json.dumps({"stage": "activation_error", "approx": approx,
            "finite": bool(torch.isfinite(y).all() & torch.isfinite(t).all()),
            "worst_input": x[worst].item(), "tanh_actual": t[worst].item(),
            "tanh_reference": ref_t[worst].item(), "tanh_max_abs": err.max().item(),
            "activation_max_abs": (y.double()-ref).abs().max().item(),
            "tanh_gate": bool(err.max() <= tanh_budget),
            "activation_gate": bool(((y.double()-ref).abs() <= activation_budget).all())}), flush=True)
        metrics[str(approx)] = {'tanh_max_abs': err.max().item(), 'activation_max_abs': (y.double()-ref).abs().max().item()}
    screen_passed = True
    for approx, (y, t) in outputs.items():
        err = (t.double() - ref_t).abs()
        assert torch.isfinite(y).all() and torch.isfinite(t).all(), "Nonfinite activation"
        passed = bool((err.max() <= tanh_budget) & ((y.double() - ref).abs() <= activation_budget).all())
        screen_passed &= passed
        if not approx or not args.explore_approximation:
            assert passed, "Activation screening tolerance exceeded"
    s0, q0 = quantize_blocks(outputs[False][0])
    s1, q1 = quantize_blocks(outputs[True][0])
    decoded0 = q0.view(torch.float8_e4m3fn).float() * s0[:, None]
    decoded1 = q1.view(torch.float8_e4m3fn).float() * s1[:, None]
    delta = (decoded1 - decoded0).double()
    print(json.dumps({'status': 'PASS' if screen_passed else 'SCREEN_EXCEEDED_EXPLORATORY',
                      'samples': x.numel(), 'tanh_absolute_screening_budget': tanh_budget,
                      'metrics': metrics, 'fp8_changed_elements': (q0 != q1).sum().item(),
                      'fp8_changed_fraction': (q0 != q1).float().mean().item(),
                      'fp8_dequantized_max_abs': delta.abs().max().item(),
                      'fp8_dequantized_relative_l2': (delta.norm()/decoded0.double().norm()).item(),
                      'scale_changed_blocks': (s0 != s1).sum().item(), 'changed_input_graph_replay': 'EXACT',
                      'note': 'Synthetic unit-scale screening with deliberately oversampled boundaries; not real-data error rate or model qualification. Full expert/output gates remain mandatory.'}))



if __name__ == '__main__':
    main()
