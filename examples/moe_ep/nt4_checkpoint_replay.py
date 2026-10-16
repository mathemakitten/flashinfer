"""Checkpoint-layer MegaMoE reference replay; optionally time a live baseline.

Offline: torchrun --nproc-per-node=4 this_file.py --artifact DIR
Live baseline capture/comparison: nt4_checkpoint_capture.py --help
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import threading
import sys
import traceback

import torch
import torch.distributed as dist

from nt4_checkpoint_artifact import (
    SCHEMA,
    load_payload,
    validate_sample,
    validate_weights,
)
from nt4_megamoe_bringup import validate_contributions, weighted_bf16_recombine


def emit(**record):
    print(json.dumps(dict(rank=dist.get_rank(), **record)), flush=True)


def gather(value, group=None):
    results = [None] * dist.get_world_size(group)
    dist.all_gather_object(results, value, group=group)
    return results


def compare(actual, expected, label):
    if not bool(torch.isfinite(actual).all()) or not bool(
        torch.isfinite(expected).all()
    ):
        raise AssertionError(f"{label}: nonfinite output")
    delta = actual.float() - expected.float()
    global_l2 = (delta.norm() / expected.float().norm().clamp_min(1e-8)).item()
    row_l2 = delta.norm(dim=-1) / expected.float().norm(dim=-1).clamp_min(1e-8)
    metrics = dict(
        global_rel_l2=global_l2,
        max_row_rel_l2=row_l2.max().item() if row_l2.numel() else 0,
        max_abs=delta.abs().max().item() if delta.numel() else 0,
        strict_mismatches=int(
            (~torch.isclose(actual, expected, rtol=0.08, atol=0.002)).sum().item()
        ),
    )
    emit(stage="comparison", label=label, **metrics)
    if global_l2 > 0.015 or metrics["max_row_rel_l2"] > 0.015:
        raise AssertionError(
            f"{label}: provisional global/per-token L2 gate failed: {metrics}"
        )
    empty_rows = (expected == 0).all(dim=-1)
    if bool((actual[empty_rows] != 0).any()):
        raise AssertionError(f"{label}: nonzero output for an exact-zero reference row")
    return metrics


class Replay:
    def __init__(
        self,
        weights,
        capacity,
        group=None,
        routing_weight_placement="after_fc2",
        knobs=None,
    ):
        self.knobs = {} if knobs is None else dict(knobs)
        self.routing_weight_placement = routing_weight_placement
        from flashinfer.moe_ep import (
            BootstrapConfig,
            FleetParams,
            MegaConfig,
            MoEEpLayer,
            MoEWeightPack,
            Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
            bootstrap_moe_ep_runtime,
        )
        from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

        self.group = group
        self.rank = dist.get_rank(group)
        if dist.get_world_size(group) != 4:
            raise ValueError("checkpoint replay requires EP4")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise ValueError(
                "checkpoint replay currently targets data-center Blackwell"
            )
        validate_weights(weights, self.rank)
        self.weights = {k: v.cuda() for k, v in weights.items()}
        self.capacity = max(1, capacity)
        config = Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=4096,
            top_k=10,
            activation="clamped_relu2",
            clamped_relu2_limit=16.0,
            routing_weight_placement=routing_weight_placement,
            enable_in_kernel_fc2_reduce=False,
            knobs=self.knobs,
        )
        bootstrap = BootstrapConfig(
            world_size=4,
            rank=self.rank,
            process_group=group,
            device=torch.cuda.current_device(),
            auto_bootstrap=False,
        )
        self.runtime = bootstrap_moe_ep_runtime(
            bootstrap, create_mega_kernel(config).runtime_requirements(bootstrap)
        )
        w = self.weights
        start = time.perf_counter()
        self.layer = MoEEpLayer(
            bootstrap=bootstrap,
            fleet_params=FleetParams(
                num_experts=512,
                max_tokens_per_rank=self.capacity,
                token_hidden_size=768,
            ),
            weights=MoEWeightPack(
                w13=w["w1"].view(torch.float8_e4m3fn),
                w2=w["w2"].view(torch.float8_e4m3fn),
                w13_scale=w["s1"],
                w2_scale=w["s2"],
            ),
            backend=MegaConfig(
                megakernel=config, quantize_input=True, preprocess_weights=True
            ),
        )
        self.x = torch.zeros(self.capacity, 768, dtype=torch.bfloat16, device="cuda")
        self.ids = torch.full((self.capacity, 10), -1, dtype=torch.int64, device="cuda")
        self.probs = torch.zeros(self.capacity, 10, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        emit(
            stage="setup",
            capacity=self.capacity,
            allocation_and_pack_seconds=time.perf_counter() - start,
        )

    def stage(self, sample):
        validate_sample(sample)
        self.n = sample["x"].shape[0]
        if self.n > self.capacity:
            raise ValueError("sample exceeds configured capacity")
        self.x.zero_()
        self.ids.fill_(-1)
        self.probs.zero_()
        self.x[: self.n].copy_(sample["x"])
        self.ids[: self.n].copy_(sample["ids"])
        self.probs[: self.n].copy_(sample["probs"])

    def run(self):
        from flashinfer.moe_ep import MoEEpTensors

        return self.layer.forward(
            MoEEpTensors(
                hidden_states=self.x, topk_ids=self.ids, topk_weights=self.probs
            )
        )

    def reference(
        self, *, diagnose_input=False, diagnose_routing=False, diagnose_term=None
    ):
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
            mxfp8_quantize_per_block_32,
        )

        def roundtrip(x):
            q, scales = mxfp8_quantize_per_block_32(x.contiguous(), torch.float8_e4m3fn)
            return q.float() * scales.float().repeat_interleave(32, -1)

        def dequant(q, s):
            return q.view(torch.float8_e4m3fn).float() * torch.pow(
                2.0, s.float() - 127
            ).repeat_interleave(32, -1)

        selected = sorted(
            set(
                e
                for values in gather(self.ids.cpu().unique().tolist(), self.group)
                for e in values
                if e >= 0
            )
        )
        terms = torch.zeros(self.capacity, 10, 768, device="cuda")
        products = torch.zeros_like(terms)
        staged_terms = torch.zeros_like(terms)
        staged_products = torch.zeros_like(terms)
        from nt4_expert_diagnostics import (
            snapshot_fc1_expert,
            validate_intermediate,
            fc1_activation_interval,
            boundary_blocks,
            decode_e8m0_exact,
        )

        all_ids = gather(self.ids.cpu(), self.group)
        counts = (
            torch.stack([torch.bincount(v[v >= 0], minlength=512) for v in all_ids])
            .sum(0)
            .tolist()
        )
        intermediate_errors = []
        stage_metrics = dict(
            changed_elements=0, changed_blocks=0, boundary_blocks=0, max_row_rel_l2=0.0
        )
        late_terms = torch.zeros_like(terms) if diagnose_routing else None
        late_bf16_terms = torch.zeros_like(terms) if diagnose_routing else None
        x = roundtrip(self.x.float())
        if diagnose_input:
            from flashinfer import mxfp8_quantize

            # Match the baseline input quantizer, requesting linear scales only
            # to make the representation comparison independent of swizzling.
            cq, cs = mxfp8_quantize(self.x.contiguous(), False, 32)
            rq, rs = mxfp8_quantize_per_block_32(self.x.float(), torch.float8_e4m3fn)
            cs = cs.view(torch.uint8).reshape(self.capacity, -1)
            rs = rs.view(torch.uint8).reshape(self.capacity, -1)
            cx = dequant(cq, cs)
            emit(
                stage="input_quantization_comparison",
                fp8_byte_mismatches=int(
                    (cq.view(torch.uint8) != rq.view(torch.uint8)).sum().item()
                ),
                scale_byte_mismatches=int((cs != rs).sum().item()),
                dequantized_rel_l2=((cx - x).norm() / x.norm().clamp_min(1e-8)).item(),
                dequantized_max_abs=(cx - x).abs().max().item(),
            )
        # Broadcast only selected experts, in identical order on every rank.
        buffers = {
            k: torch.empty_like(v[0])
            for k, v in self.weights.items()
            if k != "expert_ids"
        }
        for e in selected:
            owner, local = divmod(e, 128)
            for key in ("w1", "s1", "w2", "s2"):
                if self.rank == owner:
                    buffers[key].copy_(self.weights[key][local])
                dist.broadcast(
                    buffers[key],
                    src=(
                        dist.get_global_rank(self.group, owner)
                        if self.group is not None
                        else owner
                    ),
                    group=self.group,
                )
            # All ranks participate even when this expert has no local source rows.
            count = counts[e]
            snapshot_error = None
            if self.rank == owner:
                try:
                    metadata, q, sf = snapshot_fc1_expert(
                        self.layer._ensure_workspace(),
                        counts[owner * 128 : (owner + 1) * 128],
                        local,
                    )
                    if os.environ.get("NT4_DEBUG_PREQUANT") == "1":
                        from nt4_expert_diagnostics import snapshot_prequant_expert

                        debug_values = snapshot_prequant_expert(
                            self.layer._ensure_workspace(),
                            counts[owner * 128 : (owner + 1) * 128],
                            local,
                        )
                except Exception as error:
                    snapshot_error = str(error)
            errors = gather(snapshot_error, self.group)
            if any(error is not None for error in errors):
                raise RuntimeError(f"expert {e} intermediate snapshot failed: {errors}")
            if self.rank != owner:
                metadata = torch.empty(count, dtype=torch.int64, device="cuda")
                q = torch.empty(count, 4096, dtype=torch.uint8, device="cuda")
                sf = torch.empty(count, 128, dtype=torch.uint8, device="cuda")
                if os.environ.get("NT4_DEBUG_PREQUANT") == "1":
                    debug_values = torch.empty(
                        count, 4, dtype=torch.float32, device="cuda"
                    )
            transferred = (metadata, q, sf)
            if os.environ.get("NT4_DEBUG_PREQUANT") == "1":
                transferred += (debug_values,)
            for value in transferred:
                dist.broadcast(
                    value,
                    src=(
                        dist.get_global_rank(self.group, owner)
                        if self.group is not None
                        else owner
                    ),
                    group=self.group,
                )
            wanted = []
            for source, source_ids in enumerate(all_ids):
                for row, slot in (source_ids == e).nonzero().tolist():
                    wanted.append((source << 48) | (slot << 32) | row)
            ordered, order = metadata.sort()
            if ordered.cpu().tolist() != sorted(wanted):
                raise AssertionError(
                    f"expert {e} intermediate source metadata mismatch"
                )
            positions = (self.ids == e).nonzero()
            if not positions.numel():
                continue
            rows, slots = positions[:, 0], positions[:, 1]
            up, down = dequant(buffers["w1"], buffers["s1"]), dequant(
                buffers["w2"], buffers["s2"]
            )
            z = x[rows] @ up.T
            activated = (16 * torch.tanh(torch.relu(z) / 16)).square()
            if self.routing_weight_placement == "before_quantization":
                activated = activated * self.probs[rows, slots, None]
            a = roundtrip(activated)
            terms[rows, slots] = (a @ down.T).to(torch.bfloat16).float()
            products[rows, slots] = (a.double().abs() @ down.double().abs().T).float()
            keys = (self.rank << 48) | (slots << 32) | rows
            pool_rows = order[torch.searchsorted(ordered, keys)]
            kernel_a = dequant(q[pool_rows], sf[pool_rows])
            z64 = x[rows].double() @ up.double().T

            def activate(value):
                result = (16 * torch.tanh(torch.relu(value) / 16)).square()
                if self.routing_weight_placement == "before_quantization":
                    result = result * self.probs[rows, slots, None]
                return result

            variants = [
                a,
                roundtrip(activate(z64.float())),
                roundtrip(activate(z64).float()),
            ]
            if os.environ.get("NT4_DEBUG_PREQUANT") == "1":
                from nt4_expert_diagnostics import prequant_record

                for probe, channel in enumerate((970, 1463)):
                    for index in (
                        (kernel_a[:, channel] != a[:, channel])
                        .nonzero()
                        .flatten()
                        .tolist()
                    ):
                        pool_index = pool_rows[index]
                        raw = debug_values[
                            pool_index, probe * 2 : probe * 2 + 2
                        ].tolist()
                        scale = 2.0 ** (int(sf[pool_index, channel // 32]) - 127)
                        try:
                            detail = prequant_record(
                                raw[0],
                                raw[1],
                                z64[index, channel].item(),
                                activate(z64)[index, channel].item(),
                                kernel_a[index, channel].item(),
                                a[index, channel].item(),
                                scale,
                            )
                            emit(
                                stage="prequant_diagnostic",
                                expert=e,
                                row=int(rows[index]),
                                slot=int(slots[index]),
                                channel=channel,
                                **detail,
                            )
                        except AssertionError as error:
                            intermediate_errors.append(
                                f"expert {e} prequant snapshot: {error}"
                            )

            def boundary_check():
                lower, upper = fc1_activation_interval(x[rows], up, z64)

                def quantize_interval(value):
                    qv, sv = mxfp8_quantize_per_block_32(value, torch.float8_e4m3fn)
                    return qv, sv.float()

                diagnostics = []
                exact = torch.zeros(
                    kernel_a.numel() // 32, dtype=torch.bool, device=kernel_a.device
                )
                for variant in variants:
                    exact |= (kernel_a.reshape(-1, 32) == variant.reshape(-1, 32)).all(
                        -1
                    )
                result = boundary_blocks(
                    kernel_a,
                    variants[-1],
                    lower,
                    upper,
                    decode_e8m0_exact(sf[pool_rows]),
                    quantize_interval,
                    diagnostics=diagnostics,
                    diagnostic_mask=~exact,
                )
                for record in diagnostics:
                    r = record["expert_row"]
                    start = record["channel_start"]
                    record["stored_scale_byte"] = int(sf[pool_rows[r], start // 32])
                    record["stored_fp8_bytes"] = q[
                        pool_rows[r], start : start + 32
                    ].tolist()
                    for detail in record["details"]:
                        c = detail["channel"]
                        detail["reference_fc1_fp64"] = z64[r, c].item()
                        detail["reference_activation_fp64"] = activate(z64[r : r + 1])[
                            0, c
                        ].item()
                    emit(
                        stage="boundary_rejection_diagnostic",
                        expert=e,
                        row=int(rows[r]),
                        slot=int(slots[r]),
                        **record,
                    )
                return result

            try:
                m = validate_intermediate(
                    kernel_a,
                    variants,
                    boundary=(
                        boundary_check
                        if self.routing_weight_placement == "after_fc2"
                        else None
                    ),
                )
                stage_metrics["boundary_blocks"] += m["boundary_blocks"]
                stage_metrics["changed_elements"] += m["changed_elements"]
                stage_metrics["changed_blocks"] += m["changed_blocks"]
                stage_metrics["max_row_rel_l2"] = max(
                    stage_metrics["max_row_rel_l2"], m["max_row_rel_l2"]
                )
            except AssertionError as error:
                intermediate_errors.append(f"expert {e}: {error}")
            staged_terms[rows, slots] = (
                (kernel_a.double() @ down.double().T).to(torch.bfloat16).float()
            )
            staged_products[rows, slots] = (
                kernel_a.double().abs() @ down.double().abs().T
            ).float()
            if diagnose_term is not None:
                row, slot, channel = diagnose_term
                matches = ((rows == row) & (slots == slot)).nonzero().flatten()
                if matches.numel():
                    from nt4_expert_diagnostics import diagnose_expert

                    detail = diagnose_expert(
                        x[rows],
                        up,
                        down,
                        z,
                        a,
                        roundtrip,
                        int(matches[0]),
                        channel,
                        weights=(
                            self.probs[rows, slots]
                            if self.routing_weight_placement == "before_quantization"
                            else None
                        ),
                    )
                    snapshot = self.diagnostic_fc1_snapshot
                    kernel_a = torch.tensor(snapshot["intermediate"], device=a.device)
                    changed = (kernel_a != a[int(matches[0])]).nonzero().flatten()
                    kernel_dot = kernel_a.double() @ down[channel].double()
                    detail["kernel_intermediate"] = dict(
                        pool_row=snapshot["pool_row"],
                        sf_row=snapshot["sf_row"],
                        changed_elements_count=changed.numel(),
                        fc2_fp64_dot=kernel_dot.item(),
                        fc2_bf16=kernel_dot.to(torch.bfloat16).item(),
                        changed_elements=[
                            dict(
                                channel=j,
                                actual=kernel_a[j].item(),
                                reference=a[int(matches[0]), j].item(),
                                fp8_byte=snapshot["fp8_bytes"][j],
                                scale_byte=snapshot["scale_bytes"][j // 32],
                            )
                            for j in changed[:16].tolist()
                        ],
                    )
                    emit(
                        stage="expert_reference_diagnostic",
                        row=row,
                        slot=slot,
                        channel=channel,
                        expert=e,
                        owner=owner,
                        routing_weight=self.probs[row, slot].item(),
                        reference_term=terms[row, slot, channel].item(),
                        **detail,
                    )
            if diagnose_routing:
                unweighted = roundtrip((16 * torch.tanh(torch.relu(z) / 16)).square())
                fc2 = unweighted @ down.T
                p = self.probs[rows, slots, None]
                late_terms[rows, slots] = fc2 * p
                late_bf16_terms[rows, slots] = fc2.to(torch.bfloat16).float() * p
        self.routing_diagnostics = {}
        # Fail collectively after the broadcast loop; one failing rank must not
        # leave peers blocked in the next expert's collective.
        errors = gather(intermediate_errors, self.group)
        if any(errors):
            emit(stage="intermediate_validation", status="FAIL", errors=errors)
            raise AssertionError(
                "FC1 intermediate failed independent block-variant/boundary validation"
            )
        self.staged_terms = staged_terms
        self.staged_products = staged_products
        emit(
            stage="intermediate_validation",
            status="PASS",
            validation="nt4_stagewise_v4_provisional",
            **stage_metrics,
        )
        if diagnose_routing:
            self.routing_diagnostics = {
                "routing_after_fc2_fp32": late_terms.sum(1).to(torch.bfloat16),
                "routing_after_fc2_bf16": late_bf16_terms.sum(1).to(torch.bfloat16),
            }
        if self.routing_weight_placement == "after_fc2":
            expected = weighted_bf16_recombine(terms, self.probs)
        else:
            expected = terms.sum(1).to(torch.bfloat16)
        return expected, terms, products

    def check(
        self,
        sample,
        baseline=None,
        graph=False,
        diagnose_input=False,
        diagnose_routing=False,
        diagnose_term=None,
    ):
        self.stage(sample)
        captured = sample["baseline"].cuda()
        # First qualify the live adapter against the untouched model invocation.
        if baseline is not None:
            baseline_output = baseline().clone().reshape(-1, 768)
            compare(baseline_output, captured, "live_baseline_vs_capture")
            torch.testing.assert_close(
                baseline_output, baseline().reshape(-1, 768), rtol=0, atol=0
            )
            if graph:
                torch.cuda.synchronize()
                dist.barrier(group=self.group)
                baseline_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(baseline_graph):
                    baseline_graph_output = baseline()
                baseline_graph.replay()
                graph_copy = baseline_graph_output.clone().reshape(-1, 768)
                compare(graph_copy, captured, "graph_baseline_vs_capture")
                baseline_graph.replay()
                torch.testing.assert_close(
                    graph_copy, baseline_graph_output.reshape(-1, 768), rtol=0, atol=0
                )
        start = time.perf_counter()
        eager_output = self.run()
        torch.cuda.synchronize()
        emit(stage="first_launch", seconds=time.perf_counter() - start)
        eager_copy = eager_output.clone()
        run = self.run
        if graph:
            dist.barrier(group=self.group)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                output = self.run()

            def run():
                g.replay()
                return output

        actual = run().clone()
        torch.testing.assert_close(actual, eager_copy, rtol=0, atol=0)
        torch.testing.assert_close(actual, run(), rtol=0, atol=0)
        torch.cuda.synchronize()
        terms = self.layer._ensure_workspace().debug_combine_snapshot().float()
        if diagnose_term is not None:
            row, slot, channel = diagnose_term
            valid = row < self.n and int(self.ids[row, slot]) >= 0
            if not all(gather(valid, self.group)):
                raise ValueError(
                    "diagnostic coordinate must select an active expert slot"
                )
            from nt4_expert_diagnostics import snapshot_fc1

            requests = gather(int(self.ids[row, slot]), self.group)
            all_counts = gather(
                torch.bincount(self.ids[self.ids >= 0], minlength=512).cpu().tolist(),
                self.group,
            )
            counts = [
                sum(c[e] for c in all_counts)
                for e in range(self.rank * 128, (self.rank + 1) * 128)
            ]
            snapshots = {}
            snapshot_error = None
            try:
                for source, expert in enumerate(requests):
                    owner, local = divmod(expert, 128)
                    if owner == self.rank:
                        snapshots[source] = snapshot_fc1(
                            self.layer._ensure_workspace(),
                            counts,
                            local,
                            source,
                            row,
                            slot,
                        )
            except Exception as error:
                snapshot_error = str(error)
            reports = gather(
                dict(snapshots=snapshots, error=snapshot_error), self.group
            )
            if any(r["error"] is not None for r in reports):
                raise RuntimeError(
                    f"kernel intermediate snapshot failed: {[r['error'] for r in reports]}"
                )
            self.diagnostic_fc1_snapshot = reports[requests[self.rank] // 128][
                "snapshots"
            ][self.rank]
            emit(
                stage="expert_kernel_term",
                row=row,
                slot=slot,
                channel=channel,
                expert=int(self.ids[row, slot]),
                actual=terms[row, slot, channel].item(),
            )
        expected, reference_terms, abs_products = self.reference(
            diagnose_input=diagnose_input,
            diagnose_routing=diagnose_routing,
            diagnose_term=diagnose_term,
        )
        for label, candidate in self.routing_diagnostics.items():
            # Diagnostic hypotheses only: never substitute for the correctness gate.
            try:
                compare(captured, candidate[: self.n], "diagnostic_" + label)
            except AssertionError as error:
                emit(stage="routing_diagnostic_failure", label=label, error=str(error))
        if diagnose_term is not None:
            for lhs, rhs, label in (
                (captured, expected[: self.n], "diagnostic_baseline_vs_reference"),
                (actual[: self.n], captured, "diagnostic_megamoe_vs_baseline"),
            ):
                try:
                    compare(lhs, rhs, label)
                except AssertionError as error:
                    emit(stage="diagnostic_comparison_failure", error=str(error))
        staged_expected = (
            weighted_bf16_recombine(self.staged_terms, self.probs)
            if self.routing_weight_placement == "after_fc2"
            else self.staged_terms.sum(1).to(torch.bfloat16)
        )
        comparison_errors = []
        metrics = {}
        try:
            metrics = validate_contributions(
                actual,
                staged_expected,
                terms,
                self.staged_terms,
                self.ids,
                self.staged_products,
                4096,
                routing_weights=(
                    self.probs if self.routing_weight_placement == "after_fc2" else None
                ),
            )
            # Preserve the independent original-reference aggregate gates too.
            expert_rel = (terms - reference_terms).norm(dim=-1) / reference_terms.norm(
                dim=-1
            ).clamp_min(1e-8)
            if bool((expert_rel > 0.015).any()):
                raise AssertionError(
                    "original-reference per-expert vector relative L2 exceeds 0.015"
                )
            metrics["original_reference_expert_max_rel_l2"] = expert_rel.max().item()
        except AssertionError as error:
            comparison_errors.append(str(error))
        emit(
            stage="contribution_validation",
            routing_weight_placement=self.routing_weight_placement,
            validation="nt4_stagewise_v4_provisional",
            status="FAIL" if comparison_errors else "PASS",
            errors=comparison_errors.copy(),
            **metrics,
        )
        # Report both sides of the comparison before failing: a baseline/oracle
        # disagreement must not hide the direct MegaMoE/baseline measurement.
        for lhs, rhs, label in (
            (actual, expected, "megamoe_vs_independent_reference"),
            (captured, expected[: self.n], "captured_baseline_vs_reference"),
            (actual[: self.n], captured, "megamoe_vs_baseline"),
        ):
            try:
                compare(lhs, rhs, label)
            except AssertionError as error:
                comparison_errors.append(str(error))
        peer_errors = gather(comparison_errors, self.group)
        if any(peer_errors):
            raise AssertionError(f"stagewise validation failed: {peer_errors}")
        if bool((actual[self.n :] != 0).any()):
            raise AssertionError("nonzero padded output")
        emit(
            stage="correctness",
            status="PASS",
            graph=graph,
            validation="nt4_stagewise_v4_provisional",
            **metrics,
        )

    def close(self):
        from flashinfer.moe_ep import finalize_moe_ep_runtime

        self.layer.destroy()
        finalize_moe_ep_runtime(self.runtime)


def benchmark(
    replay, baseline, *, graph=False, warmup=20, iterations=100, rounds=5, hybrid=None
):
    """Call only after correctness. No snapshots, references or host copies in calls."""
    if os.environ.get("NT4_DEBUG_PREQUANT") == "1":
        raise RuntimeError("instrumented prequant kernel must not be benchmarked")
    functions = {"megamoe": replay.run}
    if baseline is not None:
        functions = {"baseline": baseline, **functions}
    if hybrid is not None:
        if baseline is None:
            raise ValueError("three-way timing requires a live CUTLASS baseline")
        functions = {
            "nvls_cutlass": baseline,
            "nvls_megamoe": hybrid,
            "direct_megamoe": replay.run,
        }
    graphs = []
    for name, fn in list(functions.items()):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        if graph:
            dist.barrier(group=replay.group)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            graphs.append(g)
            functions[name] = g.replay
    samples = {name: [] for name in functions}
    walls = {name: [] for name in functions}
    torch.cuda.reset_peak_memory_stats()
    for round_id in range(rounds):
        names = list(functions)
        if hybrid is not None:
            # Rotate first position across all three backends, then reverse it.
            shift = round_id % len(names)
            names = names[shift:] + names[:shift]
            if (round_id // len(names)) % 2:
                names.reverse()
        else:
            names = names[:: 1 if round_id % 2 == 0 else -1]
        for name in names:
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(iterations)
            ]
            dist.barrier(group=replay.group)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for begin, end in events:
                begin.record()
                functions[name]()
                end.record()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - start) * 1e6 / iterations
            local = [begin.elapsed_time(end) * 1000 for begin, end in events]
            ranks = gather(local, replay.group)
            samples[name].extend(max(values) for values in zip(*ranks))
            walls[name].append(max(gather(wall, replay.group)))
    report = {}
    for name, values in samples.items():
        values = torch.tensor(values, dtype=torch.float64)
        report[name] = dict(
            median_us=values.median().item(),
            p95_us=torch.quantile(values, 0.95).item(),
            batch_wall_us=walls[name],
        )
    if hybrid is not None:
        report["speedups_vs_nvls_cutlass"] = {
            name: report["nvls_cutlass"]["median_us"] / report[name]["median_us"]
            for name in ("nvls_megamoe", "direct_megamoe")
        }
    elif baseline is not None:
        report["baseline_over_megamoe"] = (
            report["baseline"]["median_us"] / report["megamoe"]["median_us"]
        )
    report["tokens_per_rank"] = gather(replay.n, replay.group)
    report["capacity"] = replay.capacity
    report["graph"] = graph
    report["peak_allocated_bytes"] = max(
        gather(torch.cuda.max_memory_allocated(), replay.group)
    )
    report["peak_reserved_bytes"] = max(
        gather(torch.cuda.max_memory_reserved(), replay.group)
    )
    report["memory_scope"] = (
        "whole process including resident weights and all instantiated backends"
    )
    emit(stage="benchmark", **report)
    return report


def profile_layers(
    replay, destination, *, baseline=None, hybrid=None, graph=True, iterations=10
):
    """Export per-rank Chrome traces after correctness and compilation warmup.

    Trace timings are diagnostic, not a replacement for unprofiled benchmarks.
    """
    from torch.profiler import ProfilerActivity, profile, record_function

    if os.environ.get("NT4_DEBUG_PREQUANT") == "1":
        raise RuntimeError("instrumented prequant kernel must not be profiled")
    if hybrid is not None and baseline is None:
        raise ValueError("three-way profiling requires the live production baseline")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    functions = {"megamoe": replay.run}
    if baseline is not None:
        functions = {"baseline": baseline, **functions}
    if hybrid is not None:
        functions = {
            "nvls_cutlass": baseline,
            "nvls_megamoe": hybrid,
            "direct_megamoe": replay.run,
        }
    for name, fn in functions.items():
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        dist.barrier(group=replay.group)
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            run = g.replay
            run()
            torch.cuda.synchronize()
        else:
            run = fn
        dist.barrier(group=replay.group)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            # Profiler startup is not simultaneous across ranks. Keep its peer
            # waits out of the measured replay ranges, and explicitly label the
            # startup work so trace analysis can exclude it.
            with record_function(f"nt4/{name}/profiler_warmup"):
                dist.barrier(group=replay.group)
                for _ in range(2):
                    run()
                torch.cuda.synchronize()
                dist.barrier(group=replay.group)
            for iteration in range(iterations):
                with record_function(f"nt4/{name}/iteration_{iteration}"):
                    run()
            torch.cuda.synchronize()
        filename = destination / f"{name}-rank{replay.rank}.json"
        prof.export_chrome_trace(str(filename))
        emit(
            stage="profile_saved",
            backend=name,
            graph=graph,
            iterations=iterations,
            path=str(filename),
        )
        dist.barrier(group=replay.group)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument(
        "--sample-index", type=int, help="replay only this saved sample (zero-based)"
    )
    parser.add_argument(
        "--knobs-json",
        default="{}",
        help="explicit NT4 scheduling knobs; bypasses tuning cache",
    )
    parser.add_argument(
        "--benchmark-megamoe",
        action="store_true",
        help="time MegaMoE only after correctness; saved baseline is not executable",
    )
    parser.add_argument("--require-gb300", action="store_true")
    parser.add_argument(
        "--profile-dir", type=Path, help="export warmed per-rank CPU/CUDA Chrome traces"
    )
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument(
        "--routing-weight-placement",
        choices=("before_quantization", "after_fc2"),
        default="after_fc2",
    )
    parser.add_argument("--diagnose-input-quantization", action="store_true")
    parser.add_argument(
        "--diagnose-prequant",
        action="store_true",
        help="compile instrumented NT4 kernel for channels 970/1463; no timing",
    )
    parser.add_argument(
        "--diagnose-expert-term",
        type=int,
        nargs=3,
        metavar=("ROW", "SLOT", "CHANNEL"),
        help="log FP64 reference sensitivity for one contribution; retains all gates",
    )
    parser.add_argument("--diagnose-routing-weight-placement", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="replay saved samples from a failed/partial capture",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.sample_index is not None and args.sample_index < 0:
        parser.error("sample-index must be nonnegative")
    if args.diagnose_prequant:
        os.environ["NT4_DEBUG_PREQUANT"] = "1"
    if os.environ.get("NT4_DEBUG_PREQUANT") == "1":
        if args.benchmark_megamoe or args.profile_dir:
            parser.error("prequant instrumentation cannot be benchmarked or profiled")
        if args.routing_weight_placement != "after_fc2":
            parser.error("prequant diagnostics currently require after_fc2 routing")
        emit_record = dict(
            stage="debug_prequant_enabled", channels=[970, 1463], timing_allowed=False
        )
        print(json.dumps(emit_record), flush=True)
    if args.diagnose_expert_term is not None:
        row, slot, channel = args.diagnose_expert_term
        if row < 0 or not 0 <= slot < 10 or not 0 <= channel < 768:
            parser.error("diagnostic coordinate requires row>=0, slot<10, channel<768")
    from nt4_gb300_tune import validate_knobs

    try:
        knobs = validate_knobs(json.loads(args.knobs_json))
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    if args.timeout_seconds < 1:
        parser.error("timeout-seconds must be positive")
    if args.profile_iterations < 1:
        parser.error("profile-iterations must be positive")
    watchdog = threading.Timer(
        args.timeout_seconds,
        lambda: (print("FAIL: replay watchdog expired", flush=True), os._exit(124)),
    )
    watchdog.daemon = True
    watchdog.start()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if args.require_gb300 and (
        torch.cuda.get_device_capability() != (10, 3)
        or "GB300" not in torch.cuda.get_device_name()
    ):
        raise RuntimeError("This tuning experiment requires GB300 (SM103)")
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    rank = dist.get_rank()
    manifest = json.loads((args.artifact / "manifest.json").read_text())
    if not manifest.get("capture_complete", False) and not args.allow_incomplete:
        raise ValueError(
            "capture incomplete; use --allow-incomplete to inspect saved samples"
        )
    if (
        manifest["schema"] != SCHEMA
        or manifest["world_size"] != 4
        or dist.get_world_size() != 4
    ):
        raise ValueError("artifact requires schema v1 and world size 4")
    part = manifest["ranks"][rank]
    weights = load_payload(args.artifact / f"weights-rank{rank}.pt", part["weights"])
    samples = [
        load_payload(args.artifact / f"sample{i}-rank{rank}.pt", hashes)
        for i, hashes in enumerate(part["samples"])
    ]
    counts = gather(len(samples))
    if min(counts) == 0 or min(counts) != max(counts):
        raise ValueError("incomplete artifact sample set")
    if args.sample_index is not None and args.sample_index >= len(samples):
        raise ValueError(
            f"sample-index {args.sample_index} is absent; artifact has {len(samples)} samples"
        )
    capacity = max(gather(max(s["x"].shape[0] for s in samples)))
    replay = Replay(
        weights,
        capacity,
        routing_weight_placement=args.routing_weight_placement,
        knobs=knobs,
    )
    emit(
        stage="tuning_config",
        knobs=knobs,
        gpu=torch.cuda.get_device_name(),
        cc=torch.cuda.get_device_capability(),
        sm_count=torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )
    for i, sample in enumerate(samples):
        if args.sample_index is not None and i != args.sample_index:
            continue
        emit(
            stage="sample",
            sample=i,
            comparison="saved baseline; no live baseline timing",
        )
        replay.check(
            sample,
            graph=args.graph,
            diagnose_input=args.diagnose_input_quantization,
            diagnose_routing=args.diagnose_routing_weight_placement,
            diagnose_term=args.diagnose_expert_term,
        )
        if args.profile_dir:
            profile_layers(
                replay,
                args.profile_dir / f"sample{i}",
                graph=args.graph,
                iterations=args.profile_iterations,
            )
        if args.benchmark_megamoe:
            benchmark(replay, None, graph=args.graph)
    replay.close()
    dist.destroy_process_group()
    watchdog.cancel()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
