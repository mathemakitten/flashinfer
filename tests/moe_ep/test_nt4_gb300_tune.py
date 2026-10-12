"""CPU tests for the isolated tuning harness; no GPU imports needed."""

import importlib.util
import json
from pathlib import Path
import unittest

p = Path(__file__).resolve().parents[2] / "examples/moe_ep/nt4_gb300_tune.py"
spec = importlib.util.spec_from_file_location("nt4_tune", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TuningTest(unittest.TestCase):
    def test_return_control_changes_only_return_mode(self):
        names = m.RETURN_CONTROL_SEQUENCE
        self.assertEqual(len(names), 3)
        self.assertEqual(names[0], names[-1])
        direct = m.validate_knobs(m.CANDIDATES[names[0]])
        staged = m.validate_knobs(m.CANDIDATES[names[1]])
        self.assertEqual(direct.pop("token_back_mode"), "epi_warps")
        self.assertEqual(staged.pop("token_back_mode"), "reuse_dispatch_warps")
        self.assertEqual(direct, staged)
        self.assertEqual(direct, {"group_hint": 512})

    def test_group256_pass(self):
        sequence = m.trial_sequence(m.GROUP256_CANDIDATES, control="group256")
        self.assertEqual(sequence[:2], ["default", "group256"])
        self.assertEqual(sequence[-2:], ["group256", "default"])
        self.assertEqual(len(sequence), 10)
        for name in m.GROUP256_CANDIDATES:
            m.validate_knobs(m.CANDIDATES[name])
        self.assertEqual(
            m.CANDIDATES["group256_flag2"], {"group_hint": 256, "flag_batch": 2}
        )

    def test_refinement_controls_and_candidates(self):
        sequence = m.trial_sequence(m.REFINE_CANDIDATES, refine=True)
        self.assertEqual(sequence[:2], ["default", "group128"])
        self.assertEqual(sequence[-2:], ["group128", "default"])
        for name in m.REFINE_CANDIDATES:
            m.validate_knobs(m.CANDIDATES[name])
        self.assertEqual(
            m.CANDIDATES["group128_epi12"],
            {"group_hint": 128, "epi_flag_batch": (1, 2)},
        )

    def test_candidate_contract(self):
        for knobs in m.CANDIDATES.values():
            m.validate_knobs(knobs)
        self.assertEqual(
            m.validate_knobs({"epi_flag_batch": [2, 2]})["epi_flag_batch"], (2, 2)
        )

    def test_reject_silent_noops_and_numerics_changes(self):
        for knobs in (
            {"unknown": 1},
            {"force_static_sched": False},
            {"mma_tiler_mnk": [128, 256, 128]},
            {"routing_weight_placement": "before_quantization"},
            {"group_hint": 0},
            {"flag_batch": True},
            {"epi_flag_batch": [0, 2]},
        ):
            with self.subTest(knobs=knobs), self.assertRaises(ValueError):
                m.validate_knobs(knobs)

    def log(self, ranks=range(4)):
        return "".join(
            json.dumps(dict(rank=r, stage="correctness", status="PASS")) for r in ranks
        ) + json.dumps(dict(rank=0, stage="benchmark", megamoe={"median_us": 100}))

    def test_all_ranks_must_pass(self):
        self.assertEqual(m.trial_summary(self.log(), 0)["status"], "PASS")
        self.assertEqual(m.trial_summary(self.log(range(3)), 0)["status"], "FAIL")
        self.assertEqual(m.trial_summary(self.log(), 124)["status"], "FAIL")
        self.assertEqual(m.trial_summary("", 0)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
