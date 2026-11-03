"""Summarize contention and an optimistic scheduling model, not an E2E prediction."""
import argparse
import json
from pathlib import Path

from nt4_branch_backend_summary import summarize


def analyze(root):
    """Validate matched inputs, then compare each rank and layer separately."""
    for arm in ('hybrid-serial', 'hybrid-overlap'):
        metadata = json.loads((root / arm / 'summary.json').read_text())
        if not metadata.get('nt4_detached_tail'):
            raise ValueError(f'{arm} did not enable detached tail')
    report = summarize(root, precision='mxfp8', trtllm_matched=True)
    by_key = {(r['arm'], r['rank'], r['layer']): r for r in report['results']}
    rows = []
    for rank in range(4):
        for layer in (7, 32, 68):
            serial = by_key['hybrid-serial', rank, layer]
            overlap = by_key['hybrid-overlap', rank, layer]
            trt = by_key['trtllm-overlap', rank, layer]
            phases = ('projection', 'mixer_core', 'mixer_post_shared')
            serial_mixer = sum(serial['phase_us'][p] for p in phases)
            overlap_mixer = sum(overlap['phase_us'][p] for p in phases)
            # Idealized model: mixer chain starts at fork, runs at serial speed,
            # and does not change the measured expert completion time. Ignores
            # launch/dependency gaps and join overhead: not an achievable schedule.
            ideal_join = max(overlap['expert_branch_done_us'], serial_mixer)
            rows.append(dict(
                rank=rank, layer=layer,
                detached_join_us=overlap['join_us'], trtllm_join_us=trt['join_us'],
                observed_gap_us=overlap['join_us'] - trt['join_us'],
                serial_mixer_phase_sum_us=serial_mixer,
                overlap_mixer_phase_sum_us=overlap_mixer,
                mixer_slowdown=overlap_mixer / serial_mixer,
                expert_branch_done_us=overlap['expert_branch_done_us'],
                tail_after_experts_us=overlap['tail_after_experts_us'],
                idealized_join_us=ideal_join,
                optimistic_recoverable_us=max(0, overlap['join_us'] - ideal_join),
                idealized_gap_to_trtllm_us=ideal_join - trt['join_us'],
                expert_phase_us=overlap['expert_phase_us'],
            ))
    result = dict(
        scope=report['scope'],
        assumptions='Idealized diagnostic calculated from phase medians. Mixer starts at fork at serial speed; expert completion unchanged. Ignores dependency gaps and join overhead. Not a rigorous bound, measured speedup, or E2E prediction. Mixer KV/recurrent states are not independently hashed.',
        results=rows,
    )
    (root / 'headroom.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
