"""Exercise output-exception policy without importing the GPU replay runtime."""
import ast
from pathlib import Path
import unittest
import torch

source = Path(__file__).resolve().parents[2] / 'examples/moe_ep/nt4_checkpoint_replay.py'
tree = ast.parse(source.read_text())
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
            ('compare', 'report_backend_agreement', 'qualification_comparisons')]
records = []
ns = dict(torch=torch, emit=lambda **kw: records.append(kw))
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), ns)
check = ns['qualification_comparisons']


class PolicyTest(unittest.TestCase):
    def test_default_rejects_l2(self):
        expected = torch.ones(2, 4)
        errors, _ = check(expected*1.1, expected, expected, 2)
        self.assertTrue(errors)

    def test_explicit_mode_records_l2(self):
        records.clear(); expected = torch.ones(2, 4)
        errors, _ = check(expected*1.1, expected, expected, 2, allow_output_l2=True)
        self.assertFalse(errors)
        self.assertTrue(any(r.get('stage') == 'tanh_output_numerical_exception' for r in records))

    def test_nonfinite_stays_fatal(self):
        expected = torch.ones(2, 4); actual=expected.clone(); actual[0,0]=float('nan')
        self.assertTrue(check(actual, expected, expected, 2, allow_output_l2=True)[0])

    def test_zero_row_stays_fatal_even_with_l2_exception(self):
        expected = torch.ones(2, 4); expected[0]=0; actual=expected.clone(); actual[0]=1
        self.assertTrue(check(actual, expected, expected, 2, allow_output_l2=True)[0])

    def test_nonfinite_backend_stays_fatal(self):
        expected=torch.ones(2,4); captured=expected.clone(); captured[0,0]=float('inf')
        self.assertTrue(check(expected*1.1, expected, captured, 2, allow_output_l2=True)[0])


if __name__ == '__main__':
    unittest.main()
