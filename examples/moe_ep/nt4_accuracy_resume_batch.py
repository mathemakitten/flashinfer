"""Resume one accuracy batch in a new directory, preserving completed baseline."""
import json
from pathlib import Path
import sys

def main():
    original=Path(__file__).resolve().with_name('nt4_e2e.py')
    source=original.read_text()
    old='        trial_launch = launch[:5] + trial_model'
    new=old+"""
        if name == 'baseline-start':
            saved=json.loads((a.output / 'baseline-start-command.json').read_text())
            if saved != trial_launch:
                raise RuntimeError('Saved baseline command differs; refusing reuse')
            records=json.loads((a.output / name / 'results.json').read_text())
            measured=[r for r in records if not r.get('warmup')]
            assert len(measured)==a.rounds and all(r.get('requested_batch_reached') for r in measured)
            outputs=json.loads((a.output / name / 'outputs-round1.json').read_text())
            assert len(outputs)==a.request_count
            print(json.dumps(dict(stage='accuracy_baseline_reused', performance_use='cross_node_accuracy_only')),flush=True)
            continue
"""
    if source.count(old)!=1:raise RuntimeError('Driver changed; review bootstrap')
    source=source.replace(old,new)
    old='a.output.mkdir(parents=True, exist_ok=a.resume)'
    if source.count(old)!=1:raise RuntimeError('Driver changed; review mkdir')
    source=source.replace(old,'a.output.mkdir(parents=True, exist_ok=True)')
    scope=dict(__name__='__main__',__file__=str(original))
    exec(compile(source,str(original),'exec'),scope)

if __name__=='__main__':main()
