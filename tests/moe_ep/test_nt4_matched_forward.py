"""Reject changed routing/continuations and aggregate the distributed critical path."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2] / 'examples/moe_ep'
tree = ast.parse((ROOT/'nt4_matched_forward.py').read_text())
ns = dict(torch=torch)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in ('check_block', 'select_blocks')], type_ignores=[]), '<check>', 'exec'), ns)
spec = importlib.util.spec_from_file_location('report', ROOT/'nt4_matched_forward_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class MatchedForwardTest(unittest.TestCase):
    def test_discovery_excludes_mtp_and_bf16(self):
        from types import SimpleNamespace as NS
        class Block:
            def __init__(self, layer, fp8=True):
                self.moe_layer = NS(layer_number=layer,
                    mlp=NS(experts=NS(_expert_weights_use_mxfp8=lambda: fp8)))
        main = [Block(7), Block(9), Block(71, False)]
        mtp = [Block(7), Block(9)]
        model = NS(decoder=NS(modules=lambda: iter(main)),
                   modules=lambda: iter(main + mtp))
        self.assertEqual(ns['select_blocks'](model, Block), main[:2])

    def test_changed_route_rejected(self):
        base = dict(input=torch.ones(2), route=[torch.ones(2)], output=[torch.ones(2)])
        ns['check_block'](base, base)
        with self.assertRaisesRegex(AssertionError, 'route'):
            ns['check_block']({**base, 'route': [torch.zeros(2)]}, base)

    def test_changed_continuation_rejected(self):
        base = dict(input=torch.ones(2), route=[torch.ones(2)], output=[torch.ones(2)])
        with self.assertRaisesRegex(AssertionError, 'output'):
            ns['check_block']({**base, 'output': [torch.zeros(2)]}, base)

    def test_report_uses_max_rank_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for arm, factor in [('exp-start', 1), ('hardware', .8), ('exp-end', 1.1)]:
                directory = root/arm/'direct'
                directory.mkdir(parents=True)
                for rank in range(4):
                    rows = [dict(rank=rank, sample=i, forward_us=(rank+1)*100*factor,
                                 input_route_sha256=str(i), logits_sha256=str(i), active_requests=128)
                            for i in range(2)]
                    (directory/f'matched-forward-rank{rank}.json').write_text(json.dumps(dict(rows=rows)))
            result = report.summarize(root)
            self.assertEqual(result['arms']['exp-start']['median_us'], 400)
            self.assertAlmostEqual(result['hardware_speedup_vs_faster_control'], 1.25)
            path = root/'hardware/direct/matched-forward-rank0.json'
            value = json.loads(path.read_text()); value['rows'][0]['logits_sha256'] = 'changed'
            path.write_text(json.dumps(value))
            with self.assertRaises(AssertionError):
                report.summarize(root)


if __name__ == '__main__':
    unittest.main()
