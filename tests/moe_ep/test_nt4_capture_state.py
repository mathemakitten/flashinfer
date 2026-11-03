"""CPU tests for decode capture eligibility, including padded/mixed steps."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[2] / "examples/moe_ep/nt4_capture_state.py"
spec = importlib.util.spec_from_file_location("nt4_capture_state", path)
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


class CaptureStateTest(unittest.TestCase):
    def test_filter_summary_distinguishes_active_from_padded_rows(self):
        stats = state.FilterStats()
        defaults = dict(
            skip_calls=1,
            max_local_tokens=128,
            last_capture_call=None,
            capture_interval=8,
            decode_only=True,
            min_active_tokens=128,
        )
        for call, active, padded, prefill in [
            (1, 1024, 1024, 18),
            (9, 76, 128, 0),
            (10, 75, 128, 0),
        ]:
            steps = [
                dict(
                    decode_only=prefill == 0,
                    prefill_requests=prefill,
                    active_tokens=active,
                )
            ] * 4
            reasons = state.filter_reasons([padded] * 4, steps, call=call, **defaults)
            stats.observe(call, [padded] * 4, steps, reasons)
        report = stats.report()
        self.assertEqual(report["decode_calls"], 2)
        self.assertEqual(report["best_decode"]["min_active_tokens"], 76)
        self.assertEqual(report["rejected_by_reason"]["active_tokens_below_minimum"], 2)
        self.assertEqual(report["rejected_by_reason"]["padded_rows_exceed_limit"], 1)
        steps = [dict(decode_only=True, prefill_requests=0, active_tokens=128)] * 4
        self.assertEqual(
            state.filter_reasons([128] * 4, steps, call=11, **defaults), []
        )

    def tearDown(self):
        state.set_context(None)

    def test_real_count_not_padding(self):
        state.set_context(
            SimpleNamespace(
                is_decode_only=lambda: True,
                get_active_request_count=lambda: 123,
                active_token_count=123,
                padded_active_token_count=128,
                num_prefill_requests=0,
            )
        )
        step = state.snapshot()
        self.assertFalse(state.eligible([step] * 4, min_active_tokens=128))
        self.assertTrue(state.eligible([step] * 4, min_active_tokens=120))

    def test_rejects_mixed_or_missing_peer(self):
        step = dict(decode_only=True, prefill_requests=0, active_tokens=128)
        self.assertFalse(state.eligible([step, None], min_active_tokens=128))
        self.assertFalse(
            state.eligible(
                [step, dict(step, prefill_requests=1)], min_active_tokens=128
            )
        )
        self.assertFalse(
            state.eligible([dict(step, decode_only=False)], min_active_tokens=128)
        )


if __name__ == "__main__":
    unittest.main()
