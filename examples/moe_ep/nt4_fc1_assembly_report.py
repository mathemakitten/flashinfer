"""Disassemble a bounded, node-local NT4 kernel dump; does not execute kernels."""
import collections
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

root = Path(sys.argv[1])
import cutlass.cute as cute
(root / 'installed-tanh.py.txt').write_text(inspect.getsource(cute.math.tanh))
cuobjdump = shutil.which('cuobjdump')
if not cuobjdump:
    raise RuntimeError('cuobjdump is required to inspect actual register/resource usage')
records = []
for cubin in sorted(root.glob('*/*/*.cubin')):
    for flag, suffix in [('--dump-resource-usage', '.resources.txt'), ('--dump-sass', '.sass.txt')]:
        with cubin.with_suffix(suffix).open('w') as f:
            subprocess.run([cuobjdump, flag, str(cubin)], stdout=f, stderr=subprocess.STDOUT, check=True)
    records.append({'cubin': str(cubin.relative_to(root)), 'resources': cubin.with_suffix('.resources.txt').read_text()})
for ptx in sorted(root.glob('*/*/*.ptx')):
    text = ptx.read_text()
    counts = collections.Counter(re.findall(r'^\s*(?:@!?%\w+\s+)?([a-z][a-z0-9_.]+)\s', text, re.M))
    records.append({'ptx': str(ptx.relative_to(root)), 'static_instruction_counts': dict(counts)})
if not any('cubin' in r for r in records):
    raise RuntimeError('No cubins were exported; no assembly conclusion is possible')
(root / 'report.json').write_text(json.dumps(records, indent=2))
print(json.dumps({'stage': 'assembly_report', 'cubins': sum('cubin' in r for r in records), 'root': str(root)}))
