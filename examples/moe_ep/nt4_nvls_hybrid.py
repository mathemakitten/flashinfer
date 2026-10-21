"""Experimental NVLS -> local fused NT4 experts -> NVLS composition.

Uses saved artifacts; no checkpoint load. The default fuses routing and FP32
partial assembly and skips the unused BF16 top-k reduction. --legacy-hybrid
retains the original composition for A/B checks. Local MegaMoE packing/cleanup
remains timed; this is not yet a lean compute-only kernel or production path.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import threading

import torch
import torch.distributed as dist

from nt4_checkpoint_artifact import SCHEMA, load_payload, validate_sample
from nt4_checkpoint_replay import (
    Replay,
    benchmark,
    compare,
    emit,
    gather,
    profile_layers,
)


def local_routes(ids, probs, rank):
    """Preserve top-k slot order, masking experts not owned by this EP rank."""
    if not 0 <= rank < 4:
        raise ValueError("requires EP4 rank")
    owned = (ids >= rank * 128) & (ids < (rank + 1) * 128)
    return torch.where(owned, ids - rank * 128, -1), torch.where(owned, probs, 0.0)


class NvlsHybrid:
    def __init__(self, direct, mcore_root, *, fused=True, packed_weights=None):
        sys.path.insert(0, str(mcore_root))
        from megatron.core.inference.communication.torch_symm_triton import (
            multimem_all_gatherv_3tensor,
            multimem_reduce_scatter_v,
        )
        from megatron.core.inference.symmetric_memory import SymmetricMemoryBuffer
        from flashinfer.moe_ep import MoEWeightPack
        from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.weights import (
            preprocess_mega_weights,
        )
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
            get_symm_buffer_for_mxfp8_mega_moe,
        )

        self.direct = direct
        self.fused = fused
        self.group, self.rank, self.capacity = (
            direct.group,
            direct.rank,
            direct.capacity,
        )
        self.n = self.capacity
        self.ag, self.rs = multimem_all_gatherv_3tensor, multimem_reduce_scatter_v
        self.buffers = []
        if dist.get_world_size(direct.group) != 4 or dist.get_world_size() != 4:
            raise ValueError("hybrid currently requires a single EP4 WORLD group")
        group = direct.group

        def allocate(shape, dtype):
            n = 1
            for d in shape:
                n *= d
            size = n * torch.empty((), dtype=dtype).element_size()
            buffer = SymmetricMemoryBuffer(max(1, (size + 1048575) // 1048576), group)
            if buffer.symm_mem_hdl is None:
                raise RuntimeError(
                    f"NVLS symmetric allocation failed: {buffer.init_failure_reason}"
                )
            self.buffers.append(buffer)
            return (
                buffer.symm_buffer[:size].view(dtype).view(shape),
                buffer.symm_mem_hdl,
            )

        total = self.capacity * 4
        self.x, self.hx = allocate((total, 768), torch.bfloat16)
        self.ids, self.hids = allocate((total, 10), torch.int64)
        self.probs, self.hp = allocate((total, 10), torch.float32)
        self.partial, self.hr = allocate((total, 768), torch.float32)
        self.output = torch.empty(
            (self.capacity, 768), dtype=torch.float32, device="cuda"
        )
        self.local_ids = torch.empty_like(self.ids)
        self.local_probs = torch.empty_like(self.probs)
        self.offset = torch.tensor(
            self.rank * self.capacity, dtype=torch.int32, device="cuda"
        )
        self.maximum = torch.tensor(self.capacity, dtype=torch.int32, device="cuda")
        self.workspace = get_symm_buffer_for_mxfp8_mega_moe(
            num_total_experts=128,
            num_max_tokens=total,
            num_topk=10,
            hidden=768,
            intermediate=4096,
            rank=0,
            world_size=1,
            activation="clamped_relu2",
            clamped_relu2_limit=16.0,
            routing_weight_placement="after_fc2",
            knobs=dict(direct.knobs),
            local_only=True,
            return_contributions_only=fused,
        )
        w = direct.weights
        self.weights = packed_weights
        if self.weights is None:
            self.weights = preprocess_mega_weights(
                MoEWeightPack(
                    w13=w["w1"].view(torch.float8_e4m3fn),
                    w2=w["w2"].view(torch.float8_e4m3fn),
                    w13_scale=w["s1"],
                    w2_scale=w["s2"],
                ),
                intermediate_size=4096,
                hidden_size=768,
                activation="clamped_relu2",
            )

    def close(self):
        self.workspace.destroy()
        self.buffers.clear()

    def run(self, *, annotate=False):
        from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.staging import (
            stage_mega_moe_inputs,
        )
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import mxfp8_mega_moe

        # Labels are enabled only for a separate diagnostic eager trace.
        from torch.profiler import record_function

        def scope(name):
            return record_function("nt4/hybrid/" + name) if annotate else nullcontext()

        with scope("nvls_dispatch"):
            # Fixed padded capacity, exactly as in the existing saved-artifact runs.
            self.ag(
                self.x,
                self.ids,
                self.probs,
                self.direct.x,
                self.direct.ids,
                self.direct.probs,
                self.hx,
                self.hids,
                self.hp,
                rank_token_offset=self.offset,
                ep_max_tokens=self.maximum,
                per_rank_max_tokens=self.capacity,
            )
        with scope("route_mask"):
            if self.fused:
                from nt4_nvls_kernels import route_mask

                route_mask(
                    self.ids, self.probs, self.local_ids, self.local_probs, self.rank
                )
                ids, probs = self.local_ids, self.local_probs
            else:
                ids, probs = local_routes(self.ids, self.probs, self.rank)
        with scope("quantize_stage"):
            ws = self.workspace
            stage_mega_moe_inputs(
                self.x, probs, ids, ws.x, ws.x_sf, ws.topk_idx, ws.topk_weights
            )
        with scope("local_fused_experts"):
            mxfp8_mega_moe(None, *self.weights, ws, num_tokens=self.capacity * 4)
        with scope("local_fp32_combine"):
            # Preserve FP32 local accumulation for the NVLS rank reduction. Avoid
            # using the already rounded BF16 local output as the reduction input.
            if self.fused:
                from nt4_nvls_kernels import weighted_partial

                weighted_partial(ws.local_contribution_view(), ids, probs, self.partial)
            else:
                terms = ws.local_contribution_view().float()
                self.partial.copy_((terms * probs[:, :, None]).sum(dim=1))
        with scope("nvls_combine"):
            self.rs(
                self.output,
                self.partial,
                self.hr,
                rank_token_offset=self.offset,
                ep_max_tokens=self.maximum,
                per_rank_max_tokens=self.capacity,
            )
        with scope("output_cast"):
            return self.output.to(torch.bfloat16)

    def check_fused_helpers(self):
        """Untimed GPU checks on actual captured routes/contributions."""
        if not self.fused:
            return
        ids, probs = local_routes(self.ids, self.probs, self.rank)
        torch.testing.assert_close(self.local_ids, ids, rtol=0, atol=0)
        torch.testing.assert_close(self.local_probs, probs, rtol=0, atol=0)
        terms = self.workspace.local_contribution_view().float()
        expected = torch.zeros_like(self.partial)
        for slot in range(10):
            value = torch.where(ids[:, slot, None] >= 0, terms[:, slot], 0.0)
            expected = expected + value * probs[:, slot, None]
        torch.testing.assert_close(self.partial, expected, rtol=0, atol=0)
        emit(stage="fused_helper_validation", status="PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--mcore-root", type=Path, required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--legacy-hybrid",
        action="store_true",
        help="use the original 19-kernel hybrid for A/B comparison",
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.profile_iterations < 3:
        parser.error("profile-iterations must be at least 3")
    if args.timeout_seconds < 1:
        parser.error("timeout must be positive")
    watchdog = threading.Timer(args.timeout_seconds, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("This first hybrid experiment requires GB300/SM103")
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    if dist.get_world_size() != 4:
        raise ValueError("requires EP4")
    manifest = json.loads((args.artifact / "manifest.json").read_text())
    if (
        manifest["schema"] != SCHEMA
        or manifest["world_size"] != 4
        or not manifest.get("capture_complete")
    ):
        raise ValueError("requires a completed EP4 artifact")
    rank = dist.get_rank()
    part = manifest["ranks"][rank]
    weights = load_payload(args.artifact / f"weights-rank{rank}.pt", part["weights"])
    samples = [
        load_payload(args.artifact / f"sample{i}-rank{rank}.pt", h)
        for i, h in enumerate(part["samples"])
    ]
    counts = gather(len(samples))
    if min(counts) < 1 or min(counts) != max(counts):
        raise ValueError("incomplete sample set")
    capacity = max(gather(max(s["x"].shape[0] for s in samples)))
    direct = Replay(weights, capacity, knobs={"group_hint": 512})
    hybrid = NvlsHybrid(direct, args.mcore_root, fused=not args.legacy_hybrid)
    for i, sample in enumerate(samples):
        validate_sample(sample)
        direct.check(sample, graph=args.graph)
        expected = direct.run().clone()
        eager = hybrid.run().clone()
        torch.cuda.synchronize()
        hybrid.check_fused_helpers()
        compare(
            eager[: direct.n], sample["baseline"].cuda(), "nvls_hybrid_vs_saved_cutlass"
        )
        compare(eager, expected, "nvls_hybrid_vs_direct")
        torch.testing.assert_close(eager, hybrid.run(), rtol=0, atol=0)
        if args.graph:
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = hybrid.run()
            for _ in range(3):
                graph.replay()
                torch.testing.assert_close(eager, output, rtol=0, atol=0)
        # Poison every slot, including valid ones: valid contributions must be
        # rewritten, while invalid slots must never be read by the reduction.
        # Reuse the captured graph across route changes as well as eager calls.
        if hybrid.fused:
            hybrid.workspace.local_contribution_view().fill_(float("nan"))
        direct.ids.fill_(-1)
        direct.probs.zero_()
        if args.graph:
            graph.replay()
            torch.testing.assert_close(output, torch.zeros_like(eager), rtol=0, atol=0)
        torch.testing.assert_close(
            hybrid.run(), torch.zeros_like(eager), rtol=0, atol=0
        )
        hybrid.check_fused_helpers()
        if hybrid.fused:
            if not torch.isnan(hybrid.workspace.local_contribution_view()).all().item():
                raise AssertionError(
                    "all-padding launch unexpectedly wrote contribution slots"
                )
        direct.stage(sample)
        if args.graph:
            graph.replay()
            torch.testing.assert_close(eager, output, rtol=0, atol=0)
        torch.testing.assert_close(eager, hybrid.run(), rtol=0, atol=0)
        hybrid.check_fused_helpers()
        if hybrid.fused:
            emit(
                stage="poisoned_contribution_validation",
                sample=i,
                status="PASS",
                graph=args.graph,
            )
        emit(stage="hybrid_correctness", sample=i, status="PASS", graph=args.graph)
        if args.benchmark:
            emit(
                stage="benchmark_labels",
                baseline=(
                    "nvls_hybrid_legacy"
                    if args.legacy_hybrid
                    else "nvls_hybrid_fused_glue"
                ),
                megamoe="group512_direct",
                note="Includes local packing/cleanup; fused glue and skipped BF16 top-k unless legacy-hybrid. No live CUTLASS timing",
            )
            benchmark(direct, hybrid.run, graph=args.graph)
        if args.profile_dir:
            destination = args.profile_dir / f"sample{i}"
            emit(
                stage="profile_labels",
                baseline=(
                    "nvls_hybrid_legacy"
                    if args.legacy_hybrid
                    else "nvls_hybrid_fused_glue"
                ),
                megamoe="group512_direct",
                note="Graph traces locate GPU costs; annotated eager traces map kernels to phases. Profiler timings are diagnostic.",
            )
            profile_layers(
                direct,
                destination / "graph",
                baseline=hybrid.run,
                graph=True,
                iterations=args.profile_iterations,
            )
            profile_layers(
                direct,
                destination / "eager_annotated",
                baseline=lambda: hybrid.run(annotate=True),
                graph=False,
                iterations=args.profile_iterations,
            )
    dist.barrier()
    hybrid.workspace.destroy()
    direct.close()
    watchdog.cancel()


if __name__ == "__main__":
    main()
