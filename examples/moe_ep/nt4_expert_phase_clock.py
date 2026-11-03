"""Opt-in graph-replayed expert-path intervals; marker overhead is included."""
import os
from contextlib import contextmanager, nullcontext

_TIMERS = {}


class ExpertPhaseClock:
    def __init__(self, layer, phases):
        import torch
        from nt4_branch_clock import stamp

        self.internal_ticks = None
        self.internal_kind = None
        self.detached_tail = False
        self.local_world_size = None
        self.nvls_ticks = None
        self.trt_buffers = {}
        self.phases = tuple(phases)
        self.stamp = stamp
        self.ticks = torch.zeros(2 * len(phases), dtype=torch.int64, device="cuda")
        for slot in range(2 * len(phases)):
            stamp(self.ticks, slot)
        _TIMERS.setdefault(layer, []).append(self)

    def set_trt_ticks(self, ticks):
        # Eager validation after capture allocates another buffer. Keep captured
        # buffers alive and select the one updated by this replay, not the last call.
        self.trt_buffers[ticks.data_ptr()] = ticks
        self.internal_kind = "trtllm"

    def set_mega_ticks(self, workspace):
        if not any(os.environ.get(k) == "1" for k in ("NT4_EXPERT_INTERNAL_TIMING", "NT4_EXPERT_SPARSE_TIMING")):
            return
        mega = workspace._frontend._mega
        region = mega.kernel._local_region_by_name["internal_ticks"]
        import torch
        self.internal_ticks = mega.local_workspace.narrow(
            0, mega.kernel._local_offsets["internal_ticks"], region.nbytes
        ).view(torch.int64).view(region.shape)
        self.internal_kind = "megamoe"
        self.detached_tail = mega.kernel.detached_tail
        self.local_world_size = mega.kernel.world_size

    def enable_nvls_readiness(self, capacity):
        import torch
        from megatron.core.inference.communication.torch_symm_triton.kernel_config import SYMM_MEM_KERNEL_CONFIG
        self.nvls_active_ctas = min(capacity, SYMM_MEM_KERNEL_CONFIG.max_num_blocks)
        self.nvls_ticks = torch.zeros((self.nvls_active_ctas, 3), dtype=torch.int64, device="cuda")

    @contextmanager
    def scope(self, name):
        slot = 2 * self.phases.index(name)
        self.stamp(self.ticks, slot)
        yield
        self.stamp(self.ticks, slot + 1)


def make_clock(direct, phases):
    layer = getattr(direct, "layer_number", None)
    if os.environ.get("NT4_EXPERT_PHASE_TIMING") == "1" and layer in (7, 32, 68):
        return ExpertPhaseClock(layer, phases)
    return None


def phase_scope(clock, name):
    return clock.scope(name) if clock is not None else nullcontext()


def read_phases(layer, fork_tick, parent):
    if os.environ.get("NT4_EXPERT_PHASE_TIMING") != "1":
        return {}
    candidates = []
    for clock in _TIMERS.get(layer, []):
        ticks = clock.ticks.cpu().tolist()
        offsets = {
            name: [(ticks[2 * i + j] - fork_tick) / 1000 for j in (0, 1)]
            for i, name in enumerate(clock.phases)
        }
        if all(parent[0] <= a <= b <= parent[1] for a, b in offsets.values()):
            candidates.append((clock, offsets))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one expert timer inside parent interval, got {len(candidates)} for layer {layer}")
    clock, offsets = candidates[0]
    internal = {}
    if os.environ.get("NT4_EXPERT_INTERNAL_TIMING") == "1":
        parent_name = "local_expert_api" if clock.internal_kind == "trtllm" else "local_fused_experts"
        if clock.internal_kind == "trtllm":
            current = []
            for buffer in clock.trt_buffers.values():
                values = buffer.cpu().tolist()
                lo, hi = offsets[parent_name]
                if lo <= (values[0] - fork_tick) / 1000 <= (values[-1] - fork_tick) / 1000 <= hi:
                    current.append(values)
            if len(current) != 1:
                raise RuntimeError(f"expected one current TRT internal timer, got {len(current)}")
            values = current[0]
        else:
            if clock.internal_ticks is None:
                raise RuntimeError(f"missing internal timestamps for layer {layer}")
            values = clock.internal_ticks.cpu().tolist()
        internal = summarize_internal(clock.internal_kind, values, fork_tick, offsets[parent_name])
    if os.environ.get("NT4_EXPERT_SPARSE_TIMING") == "1":
        if clock.internal_kind != "megamoe" or clock.internal_ticks is None:
            raise RuntimeError("Sparse FC2 timing requires MegaMoE timestamps")
        internal = summarize_sparse(clock.internal_ticks.cpu().tolist(), fork_tick,
                                    offsets["local_fused_experts"])
    readiness = {}
    if internal and clock.detached_tail and clock.internal_kind == "megamoe" and os.environ.get("NT4_EXPERT_SPARSE_TIMING") != "1":
        readiness.update(summarize_detached_readiness(values, fork_tick, offsets[parent_name]))
        readiness["local_world_size"] = clock.local_world_size
    if clock.nvls_ticks is not None:
        readiness["nvls_ctas"] = summarize_nvls_readiness(
            clock.nvls_ticks.cpu().tolist(), fork_tick, offsets["nvls_combine"], clock.nvls_active_ctas
        )
    return dict(**internal, expert_readiness=readiness, expert_phase_offsets_us=offsets,
                expert_phase_us={name: b - a for name, (a, b) in offsets.items()},
                expert_phase_note="GPU marker intervals include waits and instrumentation overhead; not pure GEMM time")


def summarize_internal(kind, ticks, fork_tick, parent):
    """Intervals overlap across warps/CTAs; never add MegaMoE roles together."""
    import statistics
    if kind == "trtllm":
        groups = {"fc1_permute_activation": [ticks[:2]], "fc2": [ticks[2:4]]}
    else:
        groups = {
            name: [warp[slot:slot + 2] for cta in ticks for warp in cta if warp[slot] != 0]
            for name, slot in (("dispatch_pack", 0), ("epilogue_fc1_fc2", 2), ("tail_cleanup", 4))
        }
    result = {}
    for name, pairs in groups.items():
        offsets = [[(a - fork_tick) / 1000, (b - fork_tick) / 1000] for a, b in pairs]
        if not offsets or not all(parent[0] <= a <= b <= parent[1] for a, b in offsets):
            raise RuntimeError(f"missing/stale/out-of-parent {kind} {name} timestamps")
        durations = [b - a for a, b in offsets]
        result[name] = dict(count=len(pairs), median_us=statistics.median(durations),
                            max_us=max(durations), span_us=max(b for a,b in offsets)-min(a for a,b in offsets),
                            first_start_us=min(a for a,b in offsets), last_end_us=max(b for a,b in offsets))
    mma = {}
    if kind == "megamoe" and len(ticks[0][0]) in (18, 26, 40):
        records = [cta[4] for cta in ticks if cta[4][6] != 0]
        if not records:
            raise RuntimeError("missing MMA timing records")
        for row in records:
            lo, hi = [(t - fork_tick) / 1000 for t in row[6:8]]
            if not parent[0] <= lo <= hi <= parent[1] or any(v < 0 for v in row[8:18]):
                raise RuntimeError("stale/invalid MMA timing record")
            # These counters partition one warp's elapsed time; never sum CTAs.
            duration_ns = row[7] - row[6]
            accounted_ns = sum(row[i] for i in (8,9,10,11,12,14,15,16))
            if accounted_ns > duration_ns:
                raise RuntimeError("MMA counters exceed warp lifetime")
        names = ("scheduler_us", "drain_us", "fc1_accumulator_wait_us",
                 "fc1_operand_wait_us", "fc1_issue_and_bookkeeping_us", "fc1_tiles",
                 "fc2_accumulator_wait_us", "fc2_operand_wait_us",
                 "fc2_issue_and_bookkeeping_us", "fc2_tiles")
        for slot, name in enumerate(names, 8):
            vals = [row[slot] / (1 if name.endswith("tiles") else 1000) for row in records]
            mma[name] = dict(median=statistics.median(vals), max=max(vals))
        mma["active_ctas"] = dict(median=len(records), max=len(records))
        if len(ticks[0][0]) in (26, 40):
            producers = [cta[5] for cta in ticks if cta[5][18] != 0]
            if len(producers) != len(records):
                raise RuntimeError("missing TMA-A timing records")
            for row in producers:
                lo, hi = [(t - fork_tick) / 1000 for t in row[18:20]]
                if not parent[0] <= lo <= hi <= parent[1] or any(v < 0 for v in row[20:24]):
                    raise RuntimeError("stale/invalid TMA-A timing record")
                if sum(row[20:23]) > row[19] - row[18]:
                    raise RuntimeError("TMA-A counters exceed warp lifetime")
            for name, slot, rows in (
                ("fc2_ready_and_acquire_fence_us", 20, producers),
                ("fc2_staging_slot_wait_us", 21, producers),
                ("fc2_staging_issue_and_bookkeeping_us", 22, producers),
                ("fc2_staged_tiles", 23, producers),
                ("fc2_mma_activation_wait_us", 24, records),
                ("fc2_mma_weight_wait_us", 25, records),
            ):
                vals = [row[slot] / (1 if name.endswith("tiles") else 1000) for row in rows]
                mma[name] = dict(median=statistics.median(vals), max=max(vals))
            for row in records:
                if row[24] < 0 or row[25] < 0 or row[24] + row[25] != row[15]:
                    raise RuntimeError("FC2 A/B waits do not partition operand wait")

    if kind == "megamoe" and len(ticks[0][0]) == 40:
        epi = [warp for cta in ticks for warp in cta[:4] if warp[26] != 0]
        if not epi:
            raise RuntimeError("missing FC1 turnover records")
        for row in epi:
            lo, hi = [(v - fork_tick) / 1000 for v in row[26:28]]
            if not parent[0] <= lo <= hi <= parent[1] or any(v < 0 for v in row[28:32]):
                raise RuntimeError("stale/invalid FC1 turnover record")
            if sum(row[29:32]) > row[27] - row[26]:
                raise RuntimeError("FC1 turnover counters exceed epilogue lifetime")
        for slot, name in enumerate(("fc1_epi_tiles", "fc1_epi_accumulator_ready_wait_us",
                                     "fc1_epi_ready_to_release_us", "fc1_epi_after_release_us"), 28):
            vals = [row[slot] / (1 if name.endswith("tiles") else 1000) for row in epi]
            mma[name] = dict(median=statistics.median(vals), max=max(vals))
    handoff = []
    if kind == "megamoe" and len(ticks[0][0]) == 40:
        producers = [(i, cta[5]) for i, cta in enumerate(ticks) if cta[5][18] != 0]
        detailed = [row for _, row in producers if row[33] != 0]
        if detailed:
            for cta, row in producers:
                if row[23] == 0:
                    continue
                if row[33] == 0 or row[37] <= 0 or row[38] <= 0 or row[39] <= 0:
                    raise RuntimeError("missing first FC2 handoff identity")
                offsets = [(v - fork_tick) / 1000 for v in row[33:37]]
                if not parent[0] <= offsets[0] <= offsets[1] <= offsets[2] <= offsets[3] <= parent[1]:
                    raise RuntimeError("invalid first FC2 handoff timestamps")
                if row[27] + row[28] != row[21] or row[31] + row[32] != row[20]:
                    raise RuntimeError("handoff split counters do not partition total wait")
                handoff.append(dict(cta=cta, counter_slot=row[37]-1,
                                    expert=row[38]-1, token_tile=row[39]-1,
                                    wait_begin_us=offsets[0], ready_us=offsets[1],
                                    first_tma_issue_us=offsets[2], last_tma_issue_us=offsets[3]))
            for slot, name in enumerate(("fc2_descriptor_setup_us", "fc2_first_slot_wait_us",
                    "fc2_later_slot_wait_us", "fc2_first_tma_issue_us", "fc2_later_tma_issue_us",
                    "fc2_counter_spin_us", "fc2_acquire_fences_us"), 26):
                vals = [row[slot]/1000 for _, row in producers]
                mma[name] = dict(median=statistics.median(vals), max=max(vals))
            for slot, name in enumerate(("fc1_handoff_scheduler_us", "fc1_store_drain_fence_us",
                    "fc1_handoff_barrier_us", "fc1_flag_bookkeeping_us", "fc1_handoff_tiles"), 32):
                vals = [row[slot]/(1 if name.endswith("tiles") else 1000) for row in epi]
                if any(v < 0 for v in vals):
                    raise RuntimeError("negative FC1 handoff duration")
                mma[name] = dict(median=statistics.median(vals), max=max(vals))
    return dict(expert_internal=result, expert_mma_counters=mma,
                expert_fc2_first_handoff=handoff, expert_mma_timing_version=6 if handoff else 5,
                expert_internal_note="Diagnostic timestamps: MegaMoE warp roles overlap and include waits, epilogue spans both FC1/FC2; TRT markers may serialize PDL. Do not sum roles or use for speed claims.")


def summarize_detached_readiness(ticks, fork_tick, parent):
    """New markers must all belong to this invocation; missing markers fail closed."""
    values = ticks[0][15][32:37]
    names = ("finalizer_entry", "first_peer_barrier_done", "second_peer_barrier_entry",
             "second_peer_barrier_done", "thread0_local_reset_done")
    offsets = [(t - fork_tick) / 1000 for t in values]
    tail_end = max(w[5] for cta in ticks for w in cta)
    if (len(values) != 5 or any(v == 0 for v in values)
            or not parent[0] <= (tail_end - fork_tick) / 1000 <= offsets[0]
            or offsets != sorted(offsets) or offsets[-1] > parent[1]):
        raise RuntimeError("missing/stale/out-of-order detached readiness markers")
    return dict(detached_offsets_us=dict(zip(names, offsets)),
                last_compute_tail_to_finalizer_entry_us=(values[0] - tail_end) / 1000,
                note="Issue markers, not exact CTA exits; local reset completion is thread 0 only. "
                     "Barrier intervals include existing CTA rendezvous; no new synchronization.")


def summarize_nvls_readiness(ticks, fork_tick, parent, active_ctas):
    """Retain each CTA instead of conflating latest arrival with longest wait."""
    if not 0 <= active_ctas <= len(ticks):
        raise RuntimeError("invalid NVLS active CTA count")
    result = []
    for cta, row in enumerate(ticks[:active_ctas]):
        if len(row) != 3 or any(t == 0 for t in row):
            raise RuntimeError("missing NVLS readiness markers")
        a, b, c = [(t - fork_tick) / 1000 for t in row]
        if not parent[0] <= a <= b <= c <= parent[1]:
            raise RuntimeError("stale/out-of-order NVLS readiness markers")
        result.append(dict(cta=cta, entry_us=a, barrier_done_us=b,
                           reduction_issue_end_us=c, barrier_inclusive_us=b-a,
                           reduction_issue_us=c-b))
    return result


def summarize_sparse(ticks, fork_tick, parent):
    """First FC2 tile endpoints; issue/commit timestamps are not GPU completion."""
    rows = []
    for cta, warps in enumerate(ticks):
        a, mma = warps[5], warps[4]
        if a[0] == 0 and mma[0] == 0:
            continue
        if not a[0] or not mma[0] or a[3:6] != mma[3:6]:
            raise RuntimeError(f"Sparse FC2 tile mismatch at CTA {cta}")
        producer = [(v - fork_tick) / 1000 for v in a[:3]]
        consumer = [(v - fork_tick) / 1000 for v in mma[:2]]
        if not (parent[0] <= producer[0] <= producer[1] <= producer[2] <= parent[1]
                and parent[0] <= consumer[0] <= consumer[1] <= parent[1]):
            raise RuntimeError(f"Stale/unordered sparse FC2 timestamps at CTA {cta}")
        rows.append(dict(cta=cta, expert=a[3]-1, tile_m=a[4]-1, tile_n=a[5]-1,
                         producer_offsets_us=producer, mma_offsets_us=consumer,
                         producer_ready_to_last_issue_us=producer[2]-producer[1],
                         mma_acquire_to_commit_issue_us=consumer[1]-consumer[0]))
    if not rows:
        raise RuntimeError("No sparse FC2 records")
    return dict(expert_fc2_sparse=rows, expert_fc2_sparse_note=
                "First FC2 tile per CTA; register-held endpoints, no per-K timers; "
                "commit issue does not mean MMA completed; diagnostic only")
