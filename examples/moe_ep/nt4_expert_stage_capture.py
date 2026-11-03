"""One matched decode invocation per rank, for standalone expert-stage replay."""
import os
from pathlib import Path

_CANDIDATES = []
_SAVED = set()


def register(hybrid):
    if os.environ.get('NT4_EXPERT_STAGE_CAPTURE'):
        _CANDIDATES.append(hybrid)


def save(probe, row):
    root = os.environ.get('NT4_EXPERT_STAGE_CAPTURE')
    if not root or probe.layer != 32 or probe.layer in _SAVED:
        return
    import json
    import torch
    from nt4_trtllm_diagnostic import tensor_hash

    if row['round'] != 1 or row['sample'] != 0 or row['input_mode'] != 'replay':
        raise RuntimeError('Stage capture requires first matched sample of round 1')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Stage capture must run after graph replay')
    lo, hi = probe.timestamps[[0, 13]].cpu().tolist()
    current = []
    for h in _CANDIDATES:
        if getattr(h.direct, 'layer_number', None) != probe.layer or h.phase_clock is None:
            continue
        ticks = h.phase_clock.ticks.cpu().tolist()
        if lo <= ticks[0] <= ticks[-1] <= hi:
            current.append(h)
    if len(current) != 1:
        raise RuntimeError(f'Expected one current expert bucket, found {len(current)}')
    h = current[0]
    directory = Path(root) / f'rank{h.rank}'
    directory.mkdir(parents=True, exist_ok=False)
    payload = dict(schema='nt4_expert_stage_v1', rank=h.rank, layer=probe.layer,
                   x=h.x.detach().cpu(), ids=h.ids.detach().cpu(), probs=h.probs.detach().cpu(),
                   weights={k:v.detach().cpu() for k,v in h.direct.weights.items()},
                   clamp_limit=h.clamp_limit.detach().cpu(), local_output=h.partial.detach().cpu(),
                   export=os.environ['FLASHINFER_TRTLLM_NT4_EXPORT'], branch_record=row)
    hashes = {k:tensor_hash(payload[k]) for k in ('x','ids','probs','local_output')}
    torch.save(payload, directory / 'inputs.pt')
    (directory / 'manifest.json').write_text(json.dumps(dict(complete=True, hashes=hashes,
        schema=payload['schema'], rank=h.rank, layer=probe.layer, branch_record=row), indent=2))
    _SAVED.add(probe.layer)
    print(json.dumps(dict(stage='expert_stage_capture_complete', rank=h.rank,
                          layer=probe.layer, artifact=str(directory))), flush=True)
