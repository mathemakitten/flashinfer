"""CPU checks for conservative candidate selection; no GPU imports."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/moe_ep"))
from nt4_bf16_tune import recommend
from nt4_bf16_tuning_config import CANDIDATES, validate_knobs


class TuningTest(unittest.TestCase):
    def trials(self, slow_sample=False, drift=False):
        trials = []
        for candidate, duration in [
            ("default", 100),
            ("group128", 95),
            ("default", 110 if drift else 101),
        ]:
            for artifact in ["128", "256"]:
                trials.append(
                    dict(
                        candidate=candidate,
                        artifact=artifact,
                        status="PASS",
                        results=[
                            dict(layer=7, sample=0, median_us=duration),
                            dict(
                                layer=32,
                                sample=1,
                                median_us=(
                                    105
                                    if slow_sample
                                    and candidate == "group128"
                                    and artifact == "256"
                                    else duration
                                ),
                            ),
                        ],
                    )
                )
        return trials

    def test_consistent_win(self):
        self.assertEqual(
            recommend(self.trials())["eligible"][0]["candidate"], "group128"
        )

    def test_regressing_sample_rejected(self):
        self.assertEqual(recommend(self.trials(slow_sample=True))["eligible"], [])

    def test_control_drift_rejected(self):
        self.assertEqual(recommend(self.trials(drift=True))["eligible"], [])

    def test_missing_control_rejected(self):
        self.assertEqual(recommend(self.trials()[:-1])["eligible"], [])

    def test_failed_candidate_rejected(self):
        trials = self.trials()
        trials[2]["status"] = "FAIL"
        self.assertEqual(recommend(trials)["eligible"], [])

    def test_supported_knobs_only(self):
        for knobs in CANDIDATES.values():
            validate_knobs(knobs)
        for knobs in [
            dict(mma_tiler_mnk=[128, 256, 64]),
            dict(apply_topk_in_fc1=True),
            dict(group_hint=0),
            dict(epi_flag_batch=[1]),
        ]:
            with self.assertRaises(ValueError):
                validate_knobs(knobs)


if __name__ == "__main__":
    unittest.main()
