"""Isolate activation and FC2 differences using saved independent FC1 inputs.

This diagnostic changes no acceptance gates and reports no performance numbers.
Actual prequantization kernel accumulators were not captured.
"""
import argparse
import json
from pathlib import Path

import torch
import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from nt4_tanh_register_bench import Fragment
from nt4_expert_diagnostics import decode_e8m0_exact


def relative(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--token', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snap = torch.load(args.snapshot, map_location='cpu', weights_only=True)
    token = torch.load(args.token, map_location='cpu', weights_only=True)
    matches = ((snap['rows'] == token['row']) & (snap['slots'] == token['slot'])).nonzero().flatten()
    assert matches.numel() == 1
    row = int(matches[0])
    slot = int(token['slot'])
    assert int(token['ids'][slot]) == int(snap['expert'])
    actual = snap['actual_fp8'].contiguous().view(torch.float8_e4m3fn).double() * decode_e8m0_exact(snap['actual_scale_bytes']).repeat_interleave(32, -1)
    ref = snap['reference_fp8'].contiguous().view(torch.float8_e4m3fn).double() * snap['reference_scales'].double().repeat_interleave(32, -1)
    down = snap['down'].double().cuda()
    outputs = {}
    for name, source in [('fp32_reference', snap['z_reference']), ('fp64_reference_rounded_to_fp32', snap['z_reference_fp64'].float())]:
        flat = source.float().flatten()
        size = ((flat.numel() + 4095) // 4096) * 4096
        x = torch.zeros(size, device='cuda')
        x[:flat.numel()].copy_(flat)
        arms = {}
        for approx in (False, True):
            arm = 'hardware' if approx else 'existing'
            values = {}
            for quant in (False, True):
                y = torch.empty_like(x)
                sf = torch.empty(size // 32 * 2, device='cuda')
                views = tuple(from_dlpack(t) for t in (x, y, sf))
                stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                fn = cute.compile(Fragment(approx, quant, 1), *views, stream)
                fn(*views, stream)
                torch.cuda.synchronize()
                assert bool(torch.isfinite(y).all()) and bool(torch.isfinite(sf).all())
                values['quantized' if quant else 'activation'] = y[:flat.numel()].reshape(source.shape).cpu()
                if quant:
                    values['scales'] = sf[::2][:flat.numel() // 32].reshape(source.shape[0], -1).cpu()
            arms[arm] = values
        changed = (arms['existing']['quantized'][row] != arms['hardware']['quantized'][row]).nonzero().flatten().tolist()
        report = {'source': name, 'focus_row': row,
                  'hardware_vs_existing_activation_l2': relative(arms['hardware']['activation'][row], arms['existing']['activation'][row]),
                  'hardware_vs_saved_kernel_quantized_l2': relative(arms['hardware']['quantized'][row], actual[row]),
                  'existing_vs_saved_reference_quantized_l2': relative(arms['existing']['quantized'][row], ref[row]),
                  'hardware_vs_existing_quantized_l2': relative(arms['hardware']['quantized'][row], arms['existing']['quantized'][row]),
                  'changed_scales_focus': int((arms['hardware']['scales'][row] != arms['existing']['scales'][row]).sum()),
                  'changed_channels_focus': [{'channel': c, 'existing_activation': float(arms['existing']['activation'][row,c]), 'hardware_activation': float(arms['hardware']['activation'][row,c]), 'existing_quantized': float(arms['existing']['quantized'][row,c]), 'hardware_quantized': float(arms['hardware']['quantized'][row,c]), 'saved_kernel': float(actual[row,c]), 'saved_reference': float(ref[row,c])} for c in changed]}
        outputs[name] = report
        print(json.dumps(report), flush=True)
    # Same FP64 FC2 computation on each saved quantized input isolates propagation.
    fc2_actual_input = (actual[row].cuda() @ down.T).cpu()
    fc2_reference_input = (ref[row].cuda() @ down.T).cpu()
    observed = token['terms'][slot].double()
    staged = token['staged_terms'][slot].double()
    original = token['reference_terms'][slot].double()
    report = {'layer': token['layer'], 'row': token['row'], 'expert': int(snap['expert']),
              'saved_activation_row_l2': relative(actual[row], ref[row]),
              'fp64_fc2_input_change_l2': relative(fc2_actual_input, fc2_reference_input),
              'kernel_vs_same_input_fp64_fc2_l2': relative(observed, fc2_actual_input),
              'kernel_vs_staged_bf16_fc2_l2': relative(observed, staged),
              'staged_vs_recomputed_bf16_fc2_l2': relative(staged, fc2_actual_input.to(torch.bfloat16)),
              'kernel_vs_original_reference_fc2_l2': relative(observed, original),
              'weighted_error_over_final_reference_norm': float(((observed-original) * token['probs'][slot].double()).norm() / token['reference'].double().norm()),
              'final_row_l2': relative(token['actual'], token['reference']),
              'note': 'One saved expert/token; no task-accuracy or E2E qualification claim. Independent FC1 inputs, not actual kernel accumulators.'}
    print(json.dumps(report), flush=True)
    (args.output / 'summary.json').write_text(json.dumps({'activation_replays': outputs, 'fc2_attribution': report}, indent=2))


if __name__ == '__main__':
    main()
