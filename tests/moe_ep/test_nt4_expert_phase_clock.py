"""CPU checks for interpreting diagnostic GPU clocks without hiding stale replays."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("nt4_clock", Path(__file__).parents[2] / "examples/moe_ep/nt4_expert_phase_clock.py")
clock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clock)

class InternalClockTests(unittest.TestCase):
    def test_sparse_endpoints_and_tile_identity(self):
        cta = [[0] * 40 for _ in range(16)]
        cta[5][:6] = [1000, 2000, 9000, 3, 1, 2]
        cta[4][:6] = [1500, 10000, 0, 3, 1, 2]
        row = clock.summarize_sparse([cta], 0, [0, 20])["expert_fc2_sparse"][0]
        self.assertEqual(row["producer_ready_to_last_issue_us"], 7)
        self.assertEqual(row["mma_acquire_to_commit_issue_us"], 8.5)
        cta[4][5] = 4
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            clock.summarize_sparse([cta], 0, [0, 20])
        cta[4][5] = 2
        cta[5][2] = 22000
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            clock.summarize_sparse([cta], 0, [0, 20])

    def test_trt_separate_stages(self):
        result = clock.summarize_internal("trtllm", [11000, 21000, 22000, 32000], 1000, [0, 40])
        self.assertEqual(result["expert_internal"]["fc1_permute_activation"]["median_us"], 10)
        self.assertEqual(result["expert_internal"]["fc2"]["first_start_us"], 21)

    def test_mega_overlapping_warps_not_summed(self):
        # Both warps span 10 us, but their union spans only 15 us.
        ticks = [[[1000,11000,2000,12000,3000,13000], [6000,16000,7000,17000,8000,18000], [0]*6]]
        r = clock.summarize_internal("megamoe", ticks, 0, [0,20])["expert_internal"]["dispatch_pack"]
        self.assertEqual(r["count"], 2)
        self.assertEqual(r["max_us"], 10)
        self.assertEqual(r["span_us"], 15)

    def test_mma_counters_and_lifetime(self):
        cta = [[0] * 18 for _ in range(16)]
        cta[0][:6] = [1000, 2000, 1000, 10000, 9000, 11000]
        cta[4][6:] = [1000, 11000, 1000, 1000, 500, 500, 1000, 2, 500, 500, 1000, 3]
        result = clock.summarize_internal("megamoe", [cta], 0, [0, 20])
        self.assertEqual(result["expert_mma_counters"]["fc2_tiles"]["median"], 3)
        self.assertEqual(result["expert_mma_counters"]["fc1_operand_wait_us"]["median"], .5)
        cta[4][8] = 20000
        with self.assertRaisesRegex(RuntimeError, "exceed"):
            clock.summarize_internal("megamoe", [cta], 0, [0, 20])

    def test_fc2_handoff_and_operand_partition(self):
        cta = [[0] * 26 for _ in range(16)]
        cta[0][:6] = [1000, 2000, 1000, 10000, 9000, 11000]
        cta[4][6:18] = [1000, 11000, 1000, 1000, 500, 500, 1000, 2, 500, 500, 1000, 3]
        cta[4][24:26] = [400, 100]
        cta[5][18:24] = [1000, 11000, 2000, 1000, 1000, 3]
        result = clock.summarize_internal("megamoe", [cta], 0, [0, 20])
        self.assertEqual(result["expert_mma_counters"]["fc2_ready_and_acquire_fence_us"]["median"], 2)
        cta[4][25] = 200
        with self.assertRaisesRegex(RuntimeError, "partition"):
            clock.summarize_internal("megamoe", [cta], 0, [0, 20])

    def test_fc1_turnover_lifetime(self):
        cta = [[0] * 40 for _ in range(16)]
        cta[0][:6] = [1000, 2000, 1000, 10000, 9000, 11000]
        cta[4][6:18] = [1000, 11000, 1000, 1000, 500, 500, 1000, 2, 500, 500, 1000, 3]
        cta[4][24:26] = [400, 100]
        cta[5][18:24] = [1000, 11000, 2000, 1000, 1000, 3]
        for warp in cta[:4]:
            warp[26:32] = [1000, 11000, 2, 2000, 1000, 3000]
        result = clock.summarize_internal("megamoe", [cta], 0, [0, 20])
        self.assertEqual(result["expert_mma_counters"]["fc1_epi_after_release_us"]["median"], 3)
        cta[0][31] = 20000
        with self.assertRaisesRegex(RuntimeError, "turnover counters exceed"):
            clock.summarize_internal("megamoe", [cta], 0, [0, 20])

    def test_split_handoff_waits_and_first_tile_identity(self):
        cta = [[0] * 40 for _ in range(16)]
        cta[0][:6] = [1000, 2000, 1000, 10000, 9000, 11000]
        cta[4][6:18] = [1000, 11000, 1000, 1000, 500, 500, 1000, 2, 500, 500, 1000, 3]
        cta[4][24:26] = [400, 100]
        cta[5][18:24] = [1000, 11000, 2000, 1000, 1000, 3]
        cta[5][26:40] = [100, 300, 700, 100, 200, 1800, 200,
                          2000, 4000, 5000, 8000, 13, 8, 2]
        for warp in cta[:4]:
            warp[26:32] = [1000, 11000, 2, 2000, 1000, 3000]
            warp[32:37] = [300, 400, 500, 100, 2]
        result = clock.summarize_internal("megamoe", [cta], 0, [0, 20])
        self.assertEqual(result["expert_mma_timing_version"], 6)
        self.assertEqual(result["expert_fc2_first_handoff"][0]["expert"], 7)
        self.assertEqual(result["expert_mma_counters"]["fc2_later_slot_wait_us"]["median"], .7)
        cta[5][28] += 1
        with self.assertRaisesRegex(RuntimeError, "partition"):
            clock.summarize_internal("megamoe", [cta], 0, [0, 20])
        cta[5][28] -= 1
        cta[5][35] = 3000
        with self.assertRaisesRegex(RuntimeError, "timestamps"):
            clock.summarize_internal("megamoe", [cta], 0, [0, 20])

    def test_stale_replay_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "stale"):
            clock.summarize_internal("trtllm", [1000,2000,3000,4000], 10000, [0,10])

    def test_incomplete_stage_rejected(self):
        with self.assertRaises(RuntimeError):
            clock.summarize_internal("trtllm", [1000,2000,3000,0], 0, [0,10])

    def test_missing_role_rejected(self):
        with self.assertRaises(RuntimeError):
            clock.summarize_internal("megamoe", [[[1000,2000,0,0,3000,4000]]], 0, [0,10])


class ReadinessClockTests(unittest.TestCase):
    def test_detached_gap_and_current_epoch(self):
        ticks = [[[0] * 40 for _ in range(16)]]
        ticks[0][0][5] = 5000
        ticks[0][15][32:37] = [6000, 7000, 8000, 10000, 11000]
        result = clock.summarize_detached_readiness(ticks, 1000, [0, 20])
        self.assertEqual(result['last_compute_tail_to_finalizer_entry_us'], 1)
        self.assertEqual(result['detached_offsets_us']['second_peer_barrier_entry'], 7)
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            clock.summarize_detached_readiness(ticks, 20000, [0, 20])
        ticks[0][15][34] = 0
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            clock.summarize_detached_readiness(ticks, 1000, [0, 20])

    def test_nvls_per_cta_not_cross_cta_difference(self):
        rows = [[1000, 9000, 10000], [5000, 6000, 12000], [1, 2, 3]]
        result = clock.summarize_nvls_readiness(rows, 0, [0, 20], 2)
        self.assertEqual([r['barrier_inclusive_us'] for r in result], [8, 1])
        self.assertEqual([r['reduction_issue_us'] for r in result], [1, 6])
        self.assertEqual(len(result), 2)  # Skipped CTA's old epoch is ignored.
        for bad in ([[1000, 900, 2000]], [[1000, 2000, 21000]], [[0, 2000, 3000]]):
            with self.assertRaises(RuntimeError):
                clock.summarize_nvls_readiness(bad, 0, [0, 20], 1)

if __name__ == "__main__":
    unittest.main()
