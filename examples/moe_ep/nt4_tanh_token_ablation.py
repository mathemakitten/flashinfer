"""Saved-token FC1-channel ablation; no model run or acceptance-policy changes.

Hold other experts and routing weights fixed, and report recombination replay
residuals explicitly. Counterfactual FC2 uses FP64 followed by BF16 rounding.
"""
import argparse
import json
from pathlib import Path

import torch
from nt4_expert_diagnostics import decode_e8m0_exact
from nt4_megamoe_bringup import weighted_bf16_recombine


def rel(a, b):
    return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--token', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    s = torch.load(a.snapshot, map_location='cpu', weights_only=True)
    t = torch.load(a.token, map_location='cpu', weights_only=True)
    matches = ((s['rows']==t['row']) & (s['slots']==t['slot'])).nonzero().flatten()
    assert len(matches)==1
    row, slot = int(matches[0]), int(t['slot'])
    assert int(s['expert'])==int(t['ids'][slot])
    actual = s['actual_fp8'].contiguous().view(torch.float8_e4m3fn).double()*decode_e8m0_exact(s['actual_scale_bytes']).repeat_interleave(32,-1)
    reference = s['reference_fp8'].contiguous().view(torch.float8_e4m3fn).double()*s['reference_scales'].double().repeat_interleave(32,-1)
    actual, reference = actual[row].cuda(), reference[row].cuda()
    down = s['down'].double().cuda()
    changed = (actual!=reference).nonzero().flatten().tolist()
    assert changed, 'No activation change found'
    assert len(changed)<=32, 'Unexpectedly broad discrepancy; inspect before expanding ablations'
    def fc2(x):
        return (x@down.T).to(torch.bfloat16).cpu().float()
    def combine(terms):
        return weighted_bf16_recombine(terms[None], t['probs'][None])[0]
    original_terms = t['terms'].float()
    ref_terms = t['reference_terms'].float()
    raw = combine(original_terms)
    assert torch.equal(combine(ref_terms),t['reference']), 'Independent reference recombination not reproduced'
    checks = dict(reference_recombination_exact=True,
                  saved_recombination_exact=torch.equal(raw,t['actual']),
                  saved_recombination_relative_l2=rel(raw,t['actual']),
                  fc2_saved_input_exact=torch.equal(fc2(actual),original_terms[slot]))
    assert checks['fc2_saved_input_exact'], 'Must reproduce saved expert before attribution'
    trials=[]
    def record(name, terms, channels):
        out=combine(terms)
        trials.append(dict(name=name, restored_channels=channels,
                           expert_relative_l2=rel(terms[slot],ref_terms[slot]),
                           recombined_relative_l2=rel(out,t['reference']),
                           recombined_max_abs=float((out.float()-t['reference'].float()).abs().max()),
                           output_difference_from_replayed_actual_l2=rel(out,raw)))
    record('replayed_actual',original_terms,[])
    for c in changed:
        x=actual.clone();x[c]=reference[c]
        terms=original_terms.clone();terms[slot]=fc2(x)
        record(f'restore_channel_{c}',terms,[c])
    terms=original_terms.clone();terms[slot]=fc2(reference)
    record('restore_all_changed_channels',terms,changed)
    # Separate the saved FP32-reference FC2 arithmetic from our FP64 counterfactual.
    terms=original_terms.clone();terms[slot]=ref_terms[slot]
    record('replace_expert_with_saved_reference',terms,changed)
    terms=ref_terms.clone();terms[slot]=original_terms[slot]
    record('introduce_only_target_expert_error_into_reference',terms,[])
    record('all_reference',ref_terms,changed)
    report=dict(layer=t['layer'],expert=int(s['expert']),row=t['row'],slot=slot,
                changed_channels=changed,routing_weight=float(t['probs'][slot]),
                controls=checks, trials=trials,
                scope='One saved routed-output vector; unchanged other experts and routes in restore trials. No logits, token-choice, accuracy, or performance claim.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
