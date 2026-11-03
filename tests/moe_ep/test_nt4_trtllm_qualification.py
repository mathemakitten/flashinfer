"""Acceptance checks must not turn explained FC1 differences into output passes."""

import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[2] / "examples/moe_ep/nt4_trtllm_qualification.py"
spec = importlib.util.spec_from_file_location("qualification", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report():
    metric = dict(finite=True, global_rel_l2=0.001, max_row_rel_l2=0.002)
    return dict(total_rows=1, expected_local_terms=1, snapshot_exact=True,
                repeat_exact=True, graph_exact=True, tactic_pinned=True,
                exact_zero_rows_preserved=True,
                input_quantization=dict(metric, byte_mismatches=0, scale_mismatches=0),
                saved_reference=dict(metric),
                experts=[dict(token=0, slot=0, fc1_unexplained_blocks=0,
                              fc1_quantized=dict(metric),
                              fc2_using_actual_intermediate_bf16=dict(metric))],
                tokens=[dict(token=0, finalization_fp32_then_bf16=dict(metric))])


class QualificationTests(unittest.TestCase):
    def test_complete_valid_report(self):
        self.assertEqual(module.qualify(report())["status"], "PASS")

    def test_explained_fc1_cannot_override_overall_output(self):
        r = report()
        r["saved_reference"]["max_row_rel_l2"] = 0.0157684
        result = module.qualify(r)
        self.assertEqual(result["stages_status"], "PASS")
        self.assertEqual(result["status"], "FAIL")

    def test_stage_and_coverage_failures(self):
        mutations = [
            lambda r: r.update(total_rows=2),
            lambda r: r.update(expected_local_terms=2),
            lambda r: r.update(graph_exact=False),
            lambda r: r.update(exact_zero_rows_preserved=False),
            lambda r: r["experts"][0].update(fc1_unexplained_blocks=1),
            lambda r: r["experts"][0]["fc2_using_actual_intermediate_bf16"].update(max_row_rel_l2=0.02),
            lambda r: r["tokens"][0]["finalization_fp32_then_bf16"].update(global_rel_l2=float("nan")),
        ]
        for mutate in mutations:
            r = report()
            mutate(r)
            with self.subTest(report=r):
                self.assertEqual(module.qualify(r)["stages_status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
