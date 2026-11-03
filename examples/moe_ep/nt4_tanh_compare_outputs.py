"""Summarize completed tanh E2E arms and retain common-prefix token evidence.

No model execution; no task-accuracy claim. Request order must match the E2E
harness, which rebuilds the same request list and resets its seed each round.
"""
import argparse
import json
import statistics
from pathlib import Path

ARMS = ('baseline-start', 'hybrid-control-start', 'hybrid-hardware-tanh',
        'hybrid-control-end', 'baseline-end')


def compare(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError('Request count changed')
    changes = []
    for request, (left, right) in enumerate(zip(reference, candidate)):
        if left == right:
            continue
        pos = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
        changes.append(dict(request_index=request, first_difference=pos,
                            common_generated_prefix_ids=left[:pos],
                            reference_next_id=left[pos] if pos < len(left) else None,
                            candidate_next_id=right[pos] if pos < len(right) else None,
                            reference_continuation_ids=left[pos:pos+32],
                            candidate_continuation_ids=right[pos:pos+32],
                            reference_length=len(left), candidate_length=len(right)))
    return dict(total_requests=len(reference), exact_requests=len(reference)-len(changes), differences=changes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    a = p.parse_args()
    required = [a.root/arm/file for arm in ARMS
                for file in ['results.json']+[f'outputs-round{i}.json' for i in range(1,a.rounds+1)]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps(dict(status='INCOMPLETE', missing=missing), indent=2))
        raise SystemExit(2)
    outputs = {arm: [json.loads((a.root/arm/f'outputs-round{i}.json').read_text())
                     for i in range(1,a.rounds+1)] for arm in ARMS}
    timing = {}
    for arm in ARMS:
        records = json.loads((a.root/arm/'results.json').read_text())
        measured = [r for r in records if not r.get('warmup', False) and r['round'] > 0]
        if len(measured) != a.rounds:
            raise ValueError(f'{arm}: expected {a.rounds} measured rounds')
        if any(r.get('profiled') or r.get('model_graph_vs_eager_outputs') != 'EXACT' for r in measured):
            raise ValueError(f'{arm}: require unprofiled, graph-qualified results')
        counts = {r['total_output_tokens'] for r in measured}
        if len(counts) != 1:
            raise ValueError(f'{arm}: output token count changed')
        timing[arm] = dict(median_wall_seconds=statistics.median(r['wall_seconds'] for r in measured),
                           median_tokens_per_second=statistics.median(r['output_tokens_per_second'] for r in measured),
                           total_output_tokens=next(iter(counts)))
    if len({v['total_output_tokens'] for v in timing.values()}) != 1:
        raise ValueError('Output token count differs between arms')
    reference = outputs['hybrid-control-start'][0]
    controls_exact = all(rows == reference for arm in ('hybrid-control-start','hybrid-control-end') for rows in outputs[arm])
    candidate_stable = all(rows == outputs['hybrid-hardware-tanh'][0] for rows in outputs['hybrid-hardware-tanh'])
    comparisons = [dict(round=i+1, **compare(reference, rows)) for i, rows in enumerate(outputs['hybrid-hardware-tanh'])]
    faster = max(timing[k]['median_tokens_per_second'] for k in ('hybrid-control-start','hybrid-control-end'))
    report = dict(root=str(a.root), timing=timing, control_tokens_exact_across_brackets_and_rounds=controls_exact,
                  candidate_tokens_exact_across_rounds=candidate_stable,
                  candidate_over_faster_hybrid_control=timing['hybrid-hardware-tanh']['median_tokens_per_second']/faster,
                  comparisons=comparisons,
                  interpretation='Token differences only; logits and answer accuracy are not measured. Common prefixes exclude original prompt IDs. Control repeatability is required for tanh attribution.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'comparisons'},indent=2))
    print(json.dumps({'changed_requests_round1':len(comparisons[0]['differences']), 'report':str(a.output)}))


if __name__ == '__main__':
    main()
