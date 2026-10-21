"""NVLS + TRTLLM-gen MXFP8 expert computation + NVLS.

Ordinary Relu2 is a performance surrogate, not the NT4 tanh-clamped activation.
Use --clamp-scale 16 with a local export for NT4 activation validation.
Weights are shuffled offline without requantization. Unclamped runs are
diagnostic only; clamped runs support the NT4 E2E comparison.
"""

import sys

import torch
import torch.distributed as dist

from nt4_checkpoint_replay import compare, emit


class TrtllmNvlsSurrogate:
    surrogate = True

    def __init__(
        self, direct, mcore_root, *, fused=True, packed_weights=None, clamp_scale=None
    ):
        sys.path.insert(0, str(mcore_root))
        from megatron.core.inference.communication.torch_symm_triton import (
            multimem_all_gatherv_3tensor,
            multimem_reduce_scatter_v,
        )
        from megatron.core.inference.symmetric_memory import SymmetricMemoryBuffer
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        self.direct = direct
        self.rank, self.capacity = direct.rank, direct.capacity
        self.buffers = []
        self.tuned = False
        self.clamp_limit = None
        self.surrogate = clamp_scale is None
        self.activation_name = "Relu2_unclamped" if self.surrogate else "ClampedRelu2"
        if clamp_scale is not None:
            import math

            if not math.isfinite(clamp_scale) or clamp_scale <= 0:
                raise ValueError("clamp_scale must be positive and finite")
            self.clamp_limit = torch.full(
                (128,), clamp_scale, device=direct.x.device, dtype=torch.float32
            )
        if dist.get_world_size(direct.group) != 4 or dist.get_world_size() != 4:
            raise ValueError("surrogate requires EP4 WORLD")
        self.ag, self.rs = multimem_all_gatherv_3tensor, multimem_reduce_scatter_v

        def allocate(shape, dtype):
            size = torch.empty((), dtype=dtype).element_size()
            for n in shape:
                size *= n
            buffer = SymmetricMemoryBuffer(
                max(1, (size + 1048575) // 1048576), direct.group
            )
            if buffer.symm_mem_hdl is None:
                raise RuntimeError(
                    f"NVLS allocation failed: {buffer.init_failure_reason}"
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
            (self.capacity, 768), device="cuda", dtype=torch.float32
        )
        self.offset = torch.tensor(
            self.rank * self.capacity, device="cuda", dtype=torch.int32
        )
        self.maximum = torch.tensor(self.capacity, device="cuda", dtype=torch.int32)
        self.weights = packed_weights
        if self.weights is None:
            w = direct.weights
            self.weights = {}
            for wk, sk in (("w1", "s1"), ("w2", "s2")):
                self.weights[wk] = torch.stack(
                    [shuffle_matrix_a(t.view(torch.uint8), 128) for t in w[wk]]
                ).view(torch.float8_e4m3fn)
                self.weights[sk] = torch.stack(
                    [shuffle_matrix_sf_a(t.view(torch.uint8), 128) for t in w[sk]]
                ).reshape(w[sk].shape)
        emit(
            stage="surrogate_setup",
            activation=self.activation_name,
            capacity=self.capacity,
            weight_layout="MajorK_shuffled",
            dimensions=[768, 4096, 512, 10],
            scope=(
                "matched_input_branch_diagnostic_only"
                if self.surrogate
                else "nt4_clamped_validation"
            ),
        )

    def close(self):
        self.buffers.clear()

    def poison(self):
        self.partial.fill_(float("nan"))
        self.output.fill_(float("nan"))

    def gather(self):
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

    def local(self):
        from flashinfer import mxfp8_quantize
        from flashinfer.fused_moe import (
            ActivationType,
            Fp8QuantizationType,
            WeightLayout,
            trtllm_fp8_block_scale_routed_moe,
        )

        q, sf = mxfp8_quantize(self.x, False, 32)
        # Invalid/padded slots carry zero probability. Use an in-range index for
        # the TRT ABI instead of depending on its handling of -1 expert IDs.
        ids = self.ids.clamp_min(0).to(torch.int32)
        probs = torch.where(self.ids >= 0, self.probs, 0.0)
        w = self.weights
        out = trtllm_fp8_block_scale_routed_moe(
            topk_ids=(ids, probs),
            routing_bias=None,
            hidden_states=q,
            hidden_states_scale=sf.view(torch.uint8).reshape(q.shape[0], -1),
            gemm1_weights=w["w1"],
            gemm1_weights_scale=w["s1"],
            gemm2_weights=w["w2"],
            gemm2_weights_scale=w["s2"],
            num_experts=512,
            top_k=10,
            n_group=None,
            topk_group=None,
            intermediate_size=4096,
            local_expert_offset=self.rank * 128,
            local_num_experts=128,
            routed_scaling_factor=None,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.MajorK.value,
            do_finalize=True,
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            activation_type=(
                ActivationType.Relu2.value
                if self.surrogate
                else ActivationType.ClampedRelu2.value
            ),
            gemm1_clamp_limit=self.clamp_limit,
        )
        return out[0] if isinstance(out, (tuple, list)) else out

    def run(self):
        self.gather()
        self.partial.copy_(self.local())
        self.rs(
            self.output,
            self.partial,
            self.hr,
            rank_token_offset=self.offset,
            ep_max_tokens=self.maximum,
            per_rank_max_tokens=self.capacity,
        )
        return self.output.to(torch.bfloat16)

    def check_fused_helpers(self):
        """Untimed full local-output comparison against independent PyTorch math.

        Quantization uses the existing independent block-32 reference. FC1/FC2
        accumulate FP32. This is the provisional 1.5% global/per-row gate used
        by the replay harness, not an NT4 activation-equivalence assertion.
        """
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
            mxfp8_quantize_per_block_32,
        )

        def roundtrip(x):
            q, sf = mxfp8_quantize_per_block_32(x.contiguous(), torch.float8_e4m3fn)
            return q.float() * sf.float().repeat_interleave(32, -1)

        def dequant(q, sf):
            return q.view(torch.float8_e4m3fn).float() * torch.pow(
                2.0, sf.float() - 127
            ).repeat_interleave(32, -1)

        if not self.tuned:
            from flashinfer.autotuner import autotune

            # Only local computation: tuning ranks must not enter NVLS collectives
            # a different number of times while exploring tactics.
            with autotune(tune_mode=True):
                self.local()
            self.tuned = True
            emit(stage="surrogate_autotune", status="complete", capacity=self.capacity)
        x = roundtrip(self.x.float())
        result = torch.zeros_like(self.x, dtype=torch.float32)
        w = self.direct.weights
        for e in range(128):
            rows, slots = (self.ids == e + self.rank * 128).nonzero(as_tuple=True)
            if not rows.numel():
                continue
            fc1 = x[rows] @ dequant(w["w1"][e], w["s1"][e]).T
            relu = torch.relu(fc1)
            if self.clamp_limit is not None:
                scale = self.clamp_limit[e]
                relu = scale * torch.tanh(relu / scale)
            act = roundtrip(relu.square())
            fc2 = act @ dequant(w["w2"][e], w["s2"][e]).T
            result.index_add_(0, rows, fc2 * self.probs[rows, slots, None])
        compare(
            self.local().float(),
            result,
            f"trtllm_{self.activation_name}_vs_independent_reference",
        )
        emit(
            stage="surrogate_reference",
            status="PASS",
            activation=self.activation_name,
            note="provisional FP32/quantized reference; full-model validation still required",
        )
        return result


def preflight():
    """EP4 saved-artifact preflight; no model load."""
    import argparse
    import json
    import os
    from pathlib import Path
    from types import SimpleNamespace
    from nt4_checkpoint_artifact import load_payload, validate_weights, validate_sample

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--mcore-root", type=Path, required=True)
    p.add_argument(
        "--clamp-scale",
        type=float,
        default=None,
        help="Opt into NT4 clamped activation; requires local TRTLLM export",
    )
    a = p.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    h = None
    try:
        rank = dist.get_rank()
        manifest = json.loads((a.artifact / "manifest.json").read_text())
        assert manifest["capture_complete"] and manifest["world_size"] == 4
        record = manifest["ranks"][rank]
        weights = load_payload(a.artifact / f"weights-rank{rank}.pt", record["weights"])
        validate_weights(weights, rank)
        sample = load_payload(
            a.artifact / f"sample0-rank{rank}.pt", record["samples"][0]
        )
        validate_sample(sample)
        direct = SimpleNamespace(
            rank=rank,
            group=dist.group.WORLD,
            capacity=sample["x"].shape[0],
            x=sample["x"].cuda(),
            ids=sample["ids"].cuda(),
            probs=sample["probs"].cuda(),
            weights={k: v.cuda() for k, v in weights.items()},
        )
        h = TrtllmNvlsSurrogate(direct, a.mcore_root, clamp_scale=a.clamp_scale)
        eager = h.run().clone()
        reference = h.check_fused_helpers()
        if not h.surrogate:
            dist.all_reduce(reference)
            start = rank * direct.capacity
            compare(
                h.run().float(),
                reference[start : start + direct.capacity],
                "nt4_trtllm_nvls_vs_independent_reference",
            )
            if a.clamp_scale == 16:
                compare(
                    h.run(), sample["baseline"].cuda(), "nt4_trtllm_vs_saved_cutlass"
                )
        # Validate graph replay against the same post-tuning tactic.
        eager = h.run().clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                h.run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = h.run()
        graph.replay()
        compare(output, eager, "surrogate_graph_vs_eager")
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
        if h.clamp_limit is not None:
            # Check that graph replay reads live per-expert scales, including
            # different scales within one persistent schedule.
            h.clamp_limit.copy_(torch.linspace(4, 20, 128, device="cuda"))
            changed_scale = h.run().clone()
            h.check_fused_helpers()
            graph.replay()
            torch.testing.assert_close(output, changed_scale, rtol=0, atol=0)
            h.clamp_limit.fill_(a.clamp_scale)
            graph.replay()
            torch.testing.assert_close(output, eager, rtol=0, atol=0)
        ids, probs = direct.ids.clone(), direct.probs.clone()
        direct.ids[:, 1::2] = -1
        direct.probs[:, 1::2] = 0
        changed = h.run().clone()
        graph.replay()
        compare(output, changed, "surrogate_changed_routes_graph_vs_eager")
        direct.ids.fill_(-1)
        direct.probs.zero_()
        h.poison()
        graph.replay()
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
        direct.ids.copy_(ids)
        direct.probs.copy_(probs)
        graph.replay()
        compare(output, eager, "surrogate_restored_routes_graph_vs_eager")
        torch.cuda.synchronize()
        graph.reset()
        emit(
            stage="surrogate_preflight" if h.surrogate else "nt4_trtllm_preflight",
            status="PASS",
            activation=h.activation_name,
        )
    finally:
        if h is not None:
            h.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    preflight()
