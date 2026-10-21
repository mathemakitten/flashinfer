"""NT4 BF16 MegaMoE checkpoint adapter and independent untimed oracle.

Contract: FP32 FC1 accumulation -> clamped squared ReLU -> BF16 intermediate
-> FP32 FC2 accumulation -> BF16 expert terms -> FP32 late weighted sum -> BF16.
Numerical thresholds are provisional; no model-quality equivalence is implied.
"""

import torch
import torch.distributed as dist

from nt4_checkpoint_replay import compare, emit


def export_bf16_weights(experts):
    if experts._expert_weights_use_mxfp8():
        raise ValueError("expected unquantized BF16 checkpoint experts")
    w1 = torch.stack(
        [
            getattr(experts.linear_fc1, f"weight{i}").detach()
            for i in range(experts.num_local_experts)
        ]
    )
    w2 = torch.stack(
        [
            getattr(experts.linear_fc2, f"weight{i}").detach()
            for i in range(experts.num_local_experts)
        ]
    )
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("NT4 BF16 weights must be BF16")
    if tuple(w1.shape) != (128, 4096, 768) or tuple(w2.shape) != (128, 768, 4096):
        raise ValueError(f"wrong NT4 expert shapes: {w1.shape}, {w2.shape}")
    return dict(w1=w1, w2=w2)


def recombine(terms, ids, probs):
    # Separate multiply/add to match the reducer's enable_fp_fusion=False.
    total = torch.zeros_like(terms[:, 0], dtype=torch.float32)
    for slot in range(terms.shape[1]):
        valid = ids[:, slot, None] >= 0
        value = torch.where(valid, terms[:, slot].float(), 0.0)
        total = total + value * torch.where(valid, probs[:, slot, None], 0.0)
    return total.bfloat16()


class Bf16Replay:
    def __init__(self, weights, capacity, group=None, knobs=None):
        from flashinfer.moe_ep import (
            BootstrapConfig,
            FleetParams,
            MegaConfig,
            MoEEpLayer,
            MoEWeightPack,
            Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
            bootstrap_moe_ep_runtime,
        )
        from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

        from nt4_bf16_tuning_config import validate_knobs

        knobs = validate_knobs(knobs or {})
        self.knobs = knobs
        self.group = group
        self.rank = dist.get_rank(group)
        if dist.get_world_size(group) != 4:
            raise ValueError("NT4 BF16 bringup requires EP4")
        self.capacity = capacity
        self.weights = weights
        cfg = Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=4096,
            top_k=10,
            activation="clamped_relu2",
            clamped_relu2_limit=16.0,
            knobs=knobs or {},
        )
        boot = BootstrapConfig(
            world_size=4,
            rank=self.rank,
            process_group=group,
            device=torch.cuda.current_device(),
            auto_bootstrap=False,
        )
        self.runtime = bootstrap_moe_ep_runtime(
            boot, create_mega_kernel(cfg).runtime_requirements(boot)
        )
        self.layer = MoEEpLayer(
            bootstrap=boot,
            fleet_params=FleetParams(
                num_experts=512, max_tokens_per_rank=capacity, token_hidden_size=768
            ),
            weights=MoEWeightPack(w13=weights["w1"], w2=weights["w2"]),
            backend=MegaConfig(
                megakernel=cfg, quantize_input=False, preprocess_weights=True
            ),
        )
        self.x = torch.zeros((capacity, 768), device="cuda", dtype=torch.bfloat16)
        self.ids = torch.full((capacity, 10), -1, device="cuda", dtype=torch.int64)
        self.probs = torch.zeros((capacity, 10), device="cuda", dtype=torch.float32)

    def run(self):
        from flashinfer.moe_ep import MoEEpTensors

        return self.layer.forward(
            MoEEpTensors(
                hidden_states=self.x, topk_ids=self.ids, topk_weights=self.probs
            )
        )

    def audit_config(self):
        """Inspect the actual frontend after warmup, outside graph capture."""
        import dataclasses
        import inspect

        frontend = self.layer._ensure_workspace()._frontend
        config = dataclasses.asdict(frontend.config)
        for key, expected in self.knobs.items():
            if config[key] != expected:
                raise RuntimeError(
                    f"BF16 knob not applied: {key}: {config[key]} != {expected}"
                )
        if frontend._mega is None:
            raise RuntimeError("BF16 audit requires a compiled kernel")
        kernel = frontend._mega.kernel
        if self.knobs.get("group_hint", kernel.group_hint) != kernel.group_hint:
            raise RuntimeError("compiled kernel object has the wrong group_hint")
        return dict(
            requested_knobs=self.knobs,
            effective_group_hint=kernel.group_hint,
            launch=frontend.launch_audit,
            frontend_config=config,
            compile_key=frontend._compile_key(),
            kernel_source=inspect.getfile(type(kernel)),
            replay_source=__file__,
        )

    def reference(self):
        xs = [torch.empty_like(self.x) for _ in range(4)]
        ids = [torch.empty_like(self.ids) for _ in range(4)]
        dist.all_gather(xs, self.x, group=self.group)
        dist.all_gather(ids, self.ids, group=self.group)
        x = torch.cat(xs)
        routes = torch.cat(ids)
        terms = torch.zeros(
            (4 * self.capacity, 10, 768), dtype=torch.float32, device="cuda"
        )
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for e in range(128):
                row, slot = torch.where(routes == self.rank * 128 + e)
                if row.numel() == 0:
                    continue
                z = x[row].float() @ self.weights["w1"][e].float().T
                act = (16 * torch.tanh(z.clamp_min(0) / 16)).square().bfloat16()
                value = (act.float() @ self.weights["w2"][e].float().T).bfloat16()
                terms[row, slot] = value.float()
            dist.all_reduce(terms, group=self.group)
            return terms[
                self.rank * self.capacity : (self.rank + 1) * self.capacity
            ].clone()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def check(self, sample, graph=False):
        if graph:
            raise ValueError("graph validation is owned by RoutedGraphAdapter")
        n = sample["x"].shape[0]
        self.x.zero_()
        self.ids.fill_(-1)
        self.probs.zero_()
        self.x[:n].copy_(sample["x"])
        self.ids[:n].copy_(sample["ids"])
        self.probs[:n].copy_(sample["probs"])
        expected_terms = self.reference()
        actual = self.run().clone()
        partials = self.layer._ensure_workspace().combine_output
        actual_terms = torch.where(self.ids[:, :, None] >= 0, partials.float(), 0.0)
        errors = (actual_terms - expected_terms).norm(dim=-1)
        norms = expected_terms.norm(dim=-1)
        failure = None
        try:
            if not bool(torch.isfinite(actual_terms).all()):
                raise AssertionError("BF16 expert produced nonfinite contributions")
            errors = (actual_terms - expected_terms).norm(dim=-1)
            norms = expected_terms.norm(dim=-1)
            if bool((errors > 0.01 * norms + 1e-5).any()):
                idx = torch.nonzero(errors > 0.01 * norms + 1e-5)[0].tolist()
                raise AssertionError(
                    f"BF16 expert contribution L2 gate failed at {idx}; no tolerance relaxation"
                )
            torch.testing.assert_close(
                actual, recombine(actual_terms, self.ids, self.probs), rtol=0, atol=0
            )
            compare(
                actual,
                recombine(expected_terms, self.ids, self.probs),
                "bf16_megamoe_vs_independent_reference",
            )
            if sample.get("baseline") is not None:
                compare(
                    actual[:n],
                    sample["baseline"].to(actual.device),
                    "bf16_megamoe_vs_live_baseline",
                )
        except (AssertionError, RuntimeError) as exc:
            failure = str(exc)
        failures = [None] * 4
        dist.all_gather_object(failures, failure, group=self.group)
        if any(failures):
            raise AssertionError(
                f"BF16 correctness gate failed across EP ranks: {failures}"
            )
        # All slots must be refreshed or masked; poison is excluded from timing.
        partials.fill_(float("nan"))
        replay = self.run()
        torch.testing.assert_close(replay, actual, rtol=0, atol=0)
        emit(
            stage="bf16_correctness",
            status="PASS",
            validation="bf16_independent_terms_v1_provisional",
            max_expert_relative_l2=(errors / norms.clamp_min(1e-8)).max().item(),
            poisoned_replay_exact=True,
        )

    def close(self):
        from flashinfer.moe_ep import finalize_moe_ep_runtime

        self.layer.destroy()
        finalize_moe_ep_runtime(self.runtime)
