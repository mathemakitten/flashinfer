"""Compare three final routed outputs to one quantized-input FP32 reference.

CUTLASS is the saved capture output. This is a numerical diagnostic, not a
throughput benchmark or a replacement for existing qualification gates.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from nt4_checkpoint_artifact import load_payload, validate_sample, validate_weights
from nt4_nvls_hybrid import NvlsHybrid
from nt4_trtllm_surrogate import TrtllmNvlsSurrogate


def reference(direct):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        mxfp8_quantize_per_block_32,
    )

    def gather(tensor):
        pieces = [torch.empty_like(tensor) for _ in range(4)]
        dist.all_gather(pieces, tensor)
        return torch.cat(pieces)

    def quantized(x):
        q, scale = mxfp8_quantize_per_block_32(x.contiguous(), torch.float8_e4m3fn)
        return q.float() * scale.float().repeat_interleave(32, -1)

    def weight(q, scale):
        return q.view(torch.float8_e4m3fn).float() * torch.pow(
            2.0, scale.float() - 127
        ).repeat_interleave(32, -1)

    x, ids, probs = gather(direct.x), gather(direct.ids), gather(direct.probs)
    x = quantized(x.float())
    # FP64 summation isolates recombination from the FP32 GEMM reference.
    result = torch.zeros_like(x, dtype=torch.float64)
    w = direct.weights
    for expert in range(128):
        rows, slots = (ids == expert + direct.rank * 128).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        fc1 = x[rows] @ weight(w['w1'][expert], w['s1'][expert]).T
        activation = (16.0 * torch.tanh(torch.relu(fc1) / 16.0)).square()
        fc2 = quantized(activation) @ weight(w['w2'][expert], w['s2'][expert]).T
        result.index_add_(0, rows, fc2.double() * probs[rows, slots, None].double())
    dist.all_reduce(result)
    start = direct.rank * direct.capacity
    return result[start : start + direct.capacity].clone()


def metrics(actual, expected):
    actual = actual.double()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {'finite': False}
    delta = actual - expected
    row_l2 = delta.norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-8)
    worst = int(row_l2.argmax())
    zero = expected.eq(0).all(dim=-1)
    return {
        'finite': True,
        'global_rel_l2': float(delta.norm() / expected.norm().clamp_min(1e-8)),
        'max_row_rel_l2': float(row_l2[worst]),
        'worst_local_row': worst,
        'worst_row_reference_norm': float(expected[worst].norm()),
        'max_abs': float(delta.abs().max()),
        'nonzero_elements_in_zero_reference_rows': int(actual[zero].ne(0).sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--mcore-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    manifest = json.loads((a.artifact / 'manifest.json').read_text())
    assert manifest['capture_complete'] and manifest['world_size'] == 4
    part = manifest['ranks'][rank]
    w = load_payload(a.artifact / f'weights-rank{rank}.pt', part['weights'])
    validate_weights(w, rank)
    records = []
    for i, hashes in enumerate(part['samples']):
        sample = load_payload(a.artifact / f'sample{i}-rank{rank}.pt', hashes)
        validate_sample(sample)
        direct = SimpleNamespace(
            rank=rank, group=dist.group.WORLD, capacity=sample['x'].shape[0],
            x=sample['x'].cuda(), ids=sample['ids'].cuda(),
            probs=sample['probs'].cuda(), weights={k: v.cuda() for k, v in w.items()},
            knobs={},
        )
        expected = reference(direct)
        outputs = {'FlashInfer CUTLASS + NVLS (saved)': sample['baseline'].cuda()}
        for label, cls, options in (
            ('MegaMoE + NVLS', NvlsHybrid, {}),
            ('TRTLLM-gen + NVLS', TrtllmNvlsSurrogate, {'clamp_scale': 16}),
        ):
            backend = cls(direct, a.mcore_root, **options)
            try:
                if isinstance(backend, TrtllmNvlsSurrogate):
                    from flashinfer.autotuner import autotune
                    backend.gather()
                    with autotune(tune_mode=True):
                        backend.local()
                    backend.tuned = True
                outputs[label] = backend.run().clone()
                torch.cuda.synchronize()
            finally:
                backend.close()
        for label, output in outputs.items():
            record = dict(rank=rank, sample=i, backend=label,
                          rows=direct.capacity, **metrics(output, expected))
            records.append(record)
            print(json.dumps(record), flush=True)
        del direct, outputs, expected
    all_records = [None] * 4
    dist.all_gather_object(all_records, records)
    if rank == 0:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps({
            'artifact': str(a.artifact),
            'reference': 'Independent block32 MXFP8 input/activation quantization; '
                         'FP32 FC1/FC2 with TF32 disabled; tanh-clamped squared ReLU '
                         'scale16; after-FC2 probability weighting and EP sum in FP64; '
                         'no BF16 rounding of reference output or expert contributions',
            'scope': 'Final routed output before shared expert/residual; saved CUTLASS, '
                     'live eager hybrid/TRTLLM. Diagnostic only; no gate changes.',
            'results': [r for rows in all_records for r in rows],
        }, indent=2) + '\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
