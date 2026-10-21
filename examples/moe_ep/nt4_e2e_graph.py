"""Per-layer routed-section graphs. Surrounding model and request loop stay eager.

Both backends copy live inputs into stable graph buffers outside capture. The
entire dispatch/expert/combine section, including direct's padded staging, runs
inside the graph. Graphs are prepared only in the untimed generation.
"""

import json
import os
import time

import torch
import torch.distributed as dist

from nt4_e2e_adapter import DecodeAdapter, KNOBS


class GraphResources:
    def __init__(self, replay=None):
        self.replay = replay
        self.buckets = {}
        self.hybrid_weights = None

    def close(self):
        # CUDA graph nodes reference symmetric memory: destroy graphs first.
        for bucket in self.buckets.values():
            if bucket["graph"] is not None:
                bucket["graph"].reset()
        for bucket in self.buckets.values():
            if bucket.get("hybrid") is not None:
                bucket["hybrid"].close()
        self.buckets.clear()
        self.hybrid_weights = None
        if self.replay is not None:
            self.replay.close()


class RoutedGraphAdapter(DecodeAdapter):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.model_graphs = os.environ.get("NT4_E2E_MODEL_GRAPHS", "0") == "1"
        self.model_capture_layers = {}
        self.shortcut_capture_layers = set()
        self.shortcut_overlap = os.environ.get("NT4_SHORTCUT_OVERLAP", "0") == "1"
        self.graph_calls = 0
        self.graph_count = 0
        from nt4_bf16_capture import Capture

        self.capture = Capture()
        self.host_timeline_enabled = os.environ.get("NT4_HOST_TIMELINE", "0") == "1"
        self.host_timeline = []
        self.diagnostic_layers = {
            int(x) for x in os.environ.get("NT4_E2E_TIMING_LAYERS", "").split(",") if x
        }
        self.timing_events = {}
        self.timing_counts = {}

    def reset_diagnostic_timing(self):
        self.timing_counts = {}

    def diagnostic_timing(self):
        """Read events only after generation has synchronized; retain all ranks."""
        import statistics

        rows = []
        for layer, count in sorted(self.timing_counts.items()):
            events = self.timing_events[layer][:count]
            copies = [a.elapsed_time(b) * 1000 for a, b, c in events]
            graphs = [b.elapsed_time(c) * 1000 for a, b, c in events]
            rows.append(
                dict(
                    layer=layer,
                    calls=count,
                    copies_sum_us=sum(copies),
                    graph_sum_us=sum(graphs),
                    graph_median_us=statistics.median(graphs),
                    graph_per_call_us=graphs,
                )
            )
        if not self.diagnostic_layers:
            return None
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, dict(rank=dist.get_rank(), layers=rows))
        return dict(
            scope="GPU events around live copies and graph replay; instrumented diagnostic, not promotion throughput",
            ranks=ranks,
        )

    def begin_diagnostic_timing(self, layer):
        if layer not in self.diagnostic_layers:
            return None
        index = self.timing_counts.get(layer, 0)
        pool = self.timing_events.setdefault(layer, [])
        if index == len(pool):
            if not self.allow_setup:
                raise RuntimeError(
                    "diagnostic event capacity exceeded warmup call count"
                )
            pool.append(tuple(torch.cuda.Event(enable_timing=True) for _ in range(3)))
        self.timing_counts[layer] = index + 1
        events = pool[index]
        events[0].record()
        return events

    def bind(self, layer):
        from nt4_checkpoint_capture import validate_layer, export_weights
        from nt4_checkpoint_replay import Replay, compare

        if layer.experts._expert_weights_use_mxfp8() != (self.precision == "mxfp8"):
            raise ValueError(
                "loaded expert precision differs from requested experiment"
            )
        d = layer.token_dispatcher
        if self.shortcut_overlap and not getattr(
            layer.config, "inference_shortcut_moe_overlap", False
        ):
            raise RuntimeError(
                "shortcut overlap requested but inactive in loaded model"
            )
        if self.mode != "vllm":
            validate_layer(
                layer,
                expected_backend=(
                    "vllm"
                    if self.shortcut_overlap and self.precision == "bf16"
                    else "flashinfer"
                ),
            )
        elif type(d).__name__ != "NVLSAllGatherVDispatcher":
            raise ValueError("routed graph comparison requires production NVLS")
        if d.shared_experts is not None and not d._external_shared_expert_launch:
            raise ValueError("shared expert execution must stay outside routed graph")
        if os.environ.get("NT4_NATIVE_CUTLASS_SERIAL_AUDIT") == "1":
            if not (
                self.mode == "baseline"
                and self.precision == "mxfp8"
                and not self.shortcut_overlap
                and not getattr(layer.config, "inference_shortcut_moe_overlap", False)
                and type(d).__name__ == "NVLSAllGatherVDispatcher"
            ):
                raise RuntimeError(
                    "native serial audit requires MXFP8 CUTLASS + NVLS with overlap disabled"
                )
            # Keep MCore's serial dispatch/expert/combine untouched. The normal
            # serial adapter substitutes a standalone replay, which would change
            # more than the scheduling variable in this correctness experiment.
            from nt4_cutlass_decode_override import install_decode_override

            self.layers[id(layer)] = (layer.layer_number, GraphResources())
            profiles = os.environ.get("NT4_CUTLASS_DECODE_PROFILE_IDS")
            if profiles:
                self.restorers.append(
                    install_decode_override(
                        layer.experts,
                        json.loads(profiles),
                        enabled=lambda: getattr(layer, "_nt4_e2e_active", False),
                    )
                )

            def observe_serial_experts(module, inputs):
                if torch.cuda.is_current_stream_capturing():
                    n = d.__class__._per_rank_worst_case_token_count
                    self.model_capture_layers.setdefault(n, set()).add(
                        layer.layer_number
                    )

            handle = layer.experts.register_forward_pre_hook(observe_serial_experts)
            self.restorers.append(handle.remove)
            print(
                json.dumps(
                    dict(
                        stage="native_cutlass_serial_audit",
                        layer=layer.layer_number,
                        profile_ids=json.loads(profiles) if profiles else None,
                    )
                ),
                flush=True,
            )
            return
        cutlass_overlap = os.environ.get("NT4_CUTLASS_OVERLAP", "0") == "1"
        if cutlass_overlap and not (
            self.shortcut_overlap
            and self.mode == "baseline"
            and self.precision == "bf16"
        ):
            raise RuntimeError(
                "CUTLASS overlap requires baseline mode, BF16 and shortcut overlap"
            )
        isolate = os.environ.get("NT4_BRANCH_NATIVE_OUTPUT") == "1"
        if isolate:
            if not (self.precision == "mxfp8" and self.shortcut_overlap):
                raise RuntimeError(
                    "native output isolation requires MXFP8 shortcut overlap"
                )
            from nt4_cutlass_decode_override import install_decode_override

            # Common continuation backend in every arm, including unmeasured
            # layers. The tested MegaMoE/TRTLLM kernels never call this method.
            self.restorers.append(
                install_decode_override(
                    layer.experts,
                    [39, 428],
                    enabled=lambda: getattr(layer, "_nt4_e2e_active", False),
                )
            )
        native_mxfp8_overlap = self.precision == "mxfp8" and (
            self.mode == "baseline"
            or (isolate and layer.layer_number not in (7, 32, 68))
        )
        if self.shortcut_overlap and (
            self.mode == "vllm" or cutlass_overlap or native_mxfp8_overlap
        ):
            # Preserve the PR's NVLS dispatch and all three expert stages.
            # Only observe capture coverage; the routed adapter would otherwise
            # bypass NVLS dispatch even though this staged path never calls it.
            self.layers[id(layer)] = (layer.layer_number, GraphResources())
            decode_profiles = os.environ.get("NT4_CUTLASS_DECODE_PROFILE_IDS")
            if decode_profiles and not isolate:
                if not native_mxfp8_overlap:
                    raise RuntimeError(
                        "decode tactic override requires native MXFP8 overlap"
                    )
                from nt4_cutlass_decode_override import install_decode_override

                self.restorers.append(
                    install_decode_override(
                        layer.experts,
                        json.loads(decode_profiles),
                        enabled=lambda: getattr(layer, "_nt4_e2e_active", False),
                    )
                )
                print(
                    json.dumps(
                        dict(
                            stage="cutlass_decode_override",
                            layer=layer.layer_number,
                            profile_ids=json.loads(decode_profiles),
                            scope="decode_only",
                        )
                    ),
                    flush=True,
                )
            if cutlass_overlap:
                from nt4_shortcut_bridge import install_cutlass_shortcut_bridge

                self.restorers.append(
                    install_cutlass_shortcut_bridge(
                        layer.experts, enabled=lambda: layer._nt4_e2e_active
                    )
                )
            original_prepare = layer.experts.prepare_shortcut_overlap

            def observe_prepare(x, probs, routing_map, *, max_blocks=0):
                if torch.cuda.is_current_stream_capturing():
                    # Native prepare receives gathered capacity-sized tensors;
                    # direct compute receives local padded decode rows. Use the
                    # dispatcher's pre-gather row count for mixed-arm coverage.
                    n = (
                        layer.token_dispatcher._local_tokens
                        if isolate
                        else layer.token_dispatcher.__class__._per_rank_worst_case_token_count
                    )
                    self.model_capture_layers.setdefault(n, set()).add(
                        layer.layer_number
                    )
                return original_prepare(x, probs, routing_map, max_blocks=max_blocks)

            layer.experts.prepare_shortcut_overlap = observe_prepare
            self.restorers.append(
                lambda: setattr(
                    layer.experts, "prepare_shortcut_overlap", original_prepare
                )
            )
            return
        replay = None
        if self.mode == "direct":
            if self.precision == "bf16":
                from nt4_bf16_replay import Bf16Replay, export_bf16_weights

                replay = Bf16Replay(
                    export_bf16_weights(layer.experts),
                    d.__class__._per_rank_worst_case_token_count,
                    d.ep_group,
                    knobs=json.loads(os.environ.get("NT4_BF16_KNOBS", "{}")),
                )
            else:
                replay = Replay(
                    export_weights(layer.experts, dist.get_rank()),
                    d.__class__._per_rank_worst_case_token_count,
                    d.ep_group,
                    knobs=KNOBS,
                )
        resources = GraphResources(replay)
        self.layers[id(layer)] = (layer.layer_number, resources)
        original_dispatch = layer.dispatch
        original_compute = layer.routed_experts_compute
        original_combine = layer.combine
        adapter = self
        checked = False

        def dispatch(x, probs):
            if not layer._nt4_e2e_active:
                return original_dispatch(x, probs)
            # Apply the live engine padding mask before staging into graph inputs.
            # Do not capture a pointer to engine metadata which may change on reset.
            from megatron.core.transformer.moe.token_dispatcher_inference import (
                mask_routing_padding,
            )

            if d.__class__._real_token_count_tensor is not None:
                mask_routing_padding(
                    d.routing_map, d.__class__._real_token_count_tensor, d.sp_rank
                )
            return x, probs

        def compute(x, probs):
            nonlocal checked
            host_trace = adapter.host_timeline_enabled and not adapter.allow_setup
            entered_ns = time.monotonic_ns() if host_trace else None
            if not layer._nt4_e2e_active:
                return original_compute(x, probs)
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing and not adapter.model_graphs:
                raise RuntimeError("nested full-model capture requires --model-graphs")
            ids = d.routing_map
            n = x.shape[0]
            key = (
                tuple(x.shape),
                x.dtype,
                tuple(ids.shape),
                ids.dtype,
                tuple(probs.shape),
                probs.dtype,
            )
            bucket = resources.buckets.get(key)
            if bucket is None:
                if capturing:
                    raise RuntimeError(
                        "routed bucket must be prepared before model capture"
                    )
                if not adapter.allow_setup:
                    raise RuntimeError(
                        f"uncaptured decode bucket {key} at layer {layer.layer_number}; timed capture forbidden"
                    )
                # Capture must be collective. Refuse different rank-local buckets.
                keys = [None] * dist.get_world_size(d.ep_group)
                dist.all_gather_object(keys, key, group=d.ep_group)
                if any(k != key for k in keys):
                    raise ValueError(
                        "initial routed graphs require matching shapes across EP ranks"
                    )
                if n > d.__class__._per_rank_worst_case_token_count:
                    raise ValueError("decode bucket exceeds production capacity")
                sx, si, sp = x.clone(), ids.clone(), probs.clone()

                hybrid = None
                if os.environ.get("NT4_MXFP8_NVLS_HYBRID", "0") == "1":
                    from types import SimpleNamespace
                    from nt4_nvls_hybrid import NvlsHybrid

                    if replay is None or adapter.precision != "mxfp8":
                        raise RuntimeError("hybrid requires MXFP8 direct adapter")
                    # Gather only the current padded bucket, not the model's
                    # maximum capacity. These are the same live bucket inputs.
                    hybrid_class = NvlsHybrid
                    hybrid_kwargs = {}
                    clamped_trt = os.environ.get("NT4_TRTLLM_CLAMPED") == "1"
                    if (
                        os.environ.get("NT4_TRTLLM_RELU2_SURROGATE") == "1"
                        and layer.layer_number in (7, 32, 68)
                        or clamped_trt
                    ):
                        from nt4_trtllm_surrogate import TrtllmNvlsSurrogate

                        hybrid_class = TrtllmNvlsSurrogate
                        if clamped_trt:
                            hybrid_kwargs["clamp_scale"] = 16.0
                    hybrid = hybrid_class(
                        SimpleNamespace(
                            x=sx,
                            ids=si,
                            probs=sp,
                            capacity=n,
                            rank=replay.rank,
                            group=replay.group,
                            weights=replay.weights,
                            knobs=dict(replay.knobs),
                        ),
                        os.environ["NT4_HYBRID_MCORE_ROOT"],
                        fused=True,
                        packed_weights=resources.hybrid_weights,
                        **hybrid_kwargs,
                    )

                if hybrid is not None:
                    resources.hybrid_weights = hybrid.weights

                def production():
                    d.routing_map = si
                    hx, hp = original_dispatch(sx, sp)
                    out, bias = original_compute(hx, hp)
                    if bias is not None:
                        raise ValueError("routed expert bias unsupported")
                    return original_combine(out)

                def direct():
                    replay.x.zero_()
                    replay.ids.fill_(-1)
                    replay.probs.zero_()
                    replay.x[:n].copy_(sx)
                    replay.ids[:n].copy_(si)
                    replay.probs[:n].copy_(sp)
                    return replay.run()[:n]

                fn = (
                    hybrid.run
                    if hybrid is not None
                    else (direct if replay is not None else production)
                )
                metadata = d.__class__._real_token_count_tensor
                # si is already masked using the current engine metadata.
                d.__class__._real_token_count_tensor = None
                try:
                    if replay is not None and not checked:
                        expected = production().detach().clone()
                        replay.check(
                            dict(
                                x=sx.cpu(),
                                ids=si.long().cpu(),
                                probs=torch.where(si < 0, 0, sp).float().cpu(),
                                baseline=expected.cpu(),
                            ),
                            graph=False,
                        )
                        checked = True
                    if hybrid is not None:
                        native = production().detach().clone()
                        eager_hybrid = hybrid.run().clone()
                        hybrid.check_fused_helpers()
                        # The surrogate tunes its local tactics during validation.
                        eager_hybrid = hybrid.run().clone()
                        if not getattr(hybrid, "surrogate", False):
                            compare(eager_hybrid, native, "nvls_hybrid_vs_live_cutlass")
                            compare(
                                eager_hybrid, direct().clone(), "nvls_hybrid_vs_direct"
                            )
                        # Invalid slots must remain unread after a route change.
                        saved_i, saved_p = si.clone(), sp.clone()
                        if hasattr(hybrid, "poison"):
                            hybrid.poison()
                        else:
                            hybrid.workspace.local_contribution_view().fill_(
                                float("nan")
                            )
                        si.fill_(-1)
                        sp.zero_()
                        torch.testing.assert_close(
                            hybrid.run(), torch.zeros_like(eager_hybrid), rtol=0, atol=0
                        )
                        si.copy_(saved_i)
                        sp.copy_(saved_p)
                        torch.testing.assert_close(
                            hybrid.run(), eager_hybrid, rtol=0, atol=0
                        )
                    # Compile/initialize on a side stream before entering capture.
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            eager = fn()
                        expected_graph = eager.clone()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    dist.barrier(group=d.ep_group)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = fn()
                    graph.replay()
                    compare(output, expected_graph, "routed_graph_vs_eager")
                    # Reuse with different graph inputs, including a different
                    # route mask, to detect stale or baked-in input references.
                    saved_ids, saved_probs = si.clone(), sp.clone()
                    if hybrid is not None:
                        if hasattr(hybrid, "poison"):
                            hybrid.poison()
                        else:
                            hybrid.workspace.local_contribution_view().fill_(
                                float("nan")
                            )
                        si.fill_(-1)
                        sp.zero_()
                        graph.replay()
                        torch.testing.assert_close(
                            output, torch.zeros_like(output), rtol=0, atol=0
                        )
                        si.copy_(saved_ids)
                        sp.copy_(saved_probs)
                    si[:, 1::2] = -1
                    sp[:, 1::2] = 0
                    expected_masked = fn().clone()
                    graph.replay()
                    compare(
                        output, expected_masked, "routed_graph_changed_routes_vs_eager"
                    )
                    si.copy_(saved_ids)
                    sp.copy_(saved_probs)
                    graph.replay()
                    compare(
                        output, expected_graph, "routed_graph_restored_inputs_vs_eager"
                    )
                    if replay is not None and adapter.precision == "bf16":
                        print(
                            json.dumps(
                                dict(
                                    stage="bf16_effective_config",
                                    rank=dist.get_rank(),
                                    layer=layer.layer_number,
                                    rows=n,
                                    active_routed_rows=int(
                                        (si >= 0).any(dim=1).sum().item()
                                    ),
                                    **replay.audit_config(),
                                )
                            ),
                            flush=True,
                        )
                    if adapter.model_graphs:
                        graph.reset()  # validation-only subgraph; capture fn directly below
                        graph = None
                    bucket = dict(
                        graph=graph,
                        fn=fn,
                        x=sx,
                        ids=si,
                        probs=sp,
                        output=output,
                        hybrid=hybrid,
                    )
                    resources.buckets[key] = bucket
                    adapter.graph_count += 1
                    print(
                        json.dumps(
                            dict(
                                rank=dist.get_rank(),
                                stage="routed_graph_ready",
                                mode=(
                                    "trtllm_clamped"
                                    if getattr(hybrid, "activation_name", None)
                                    == "ClampedRelu2"
                                    else (
                                        "trtllm_relu2_surrogate"
                                        if getattr(hybrid, "surrogate", False)
                                        else (
                                            "nvls_hybrid"
                                            if hybrid is not None
                                            else adapter.mode
                                        )
                                    )
                                ),
                                layer=layer.layer_number,
                                rows=n,
                                capacity=d.__class__._per_rank_worst_case_token_count,
                                validation="eager_and_changed_routes",
                                graph_pool_policy="per_bucket_private",
                                layer_buckets=len(resources.buckets),
                                allocated_bytes=torch.cuda.memory_allocated(),
                                reserved_bytes=torch.cuda.memory_reserved(),
                                status="PASS",
                            )
                        ),
                        flush=True,
                    )
                finally:
                    d.__class__._real_token_count_tensor = metadata
                    d.routing_map = ids
            # These three live-input copies are identical on all arms and remain
            # timed. Internal candidate clearing/copying is captured in direct().
            copies_start_ns = time.monotonic_ns() if host_trace else None
            timing = adapter.begin_diagnostic_timing(layer.layer_number)
            bucket["x"].copy_(x)
            bucket["ids"].copy_(ids)
            bucket["probs"].copy_(probs)
            if timing is not None:
                timing[1].record()
            replay_start_ns = time.monotonic_ns() if host_trace else None
            if adapter.model_graphs:
                bucket["output"] = bucket["fn"]()
                if capturing:
                    adapter.model_capture_layers.setdefault(n, set()).add(
                        layer.layer_number
                    )
            else:
                bucket["graph"].replay()
            if host_trace:
                replay_end_ns = time.monotonic_ns()
                adapter.host_timeline.append(
                    dict(
                        layer=layer.layer_number,
                        rows=n,
                        entered_ns=entered_ns,
                        copies_start_ns=copies_start_ns,
                        replay_start_ns=replay_start_ns,
                        replay_end_ns=replay_end_ns,
                    )
                )
            if timing is not None:
                timing[2].record()
            adapter.graph_calls += 1
            if replay is not None:
                adapter.calls += 1
            if (
                adapter.allow_setup
                and adapter.precision == "bf16"
                and adapter.mode == "baseline"
            ):
                adapter.capture.save(
                    layer,
                    bucket["x"],
                    bucket["ids"],
                    bucket["probs"],
                    bucket["output"],
                    d.__class__._per_rank_worst_case_token_count,
                )
            return bucket["output"], None

        def combine(x):
            return x if layer._nt4_e2e_active else original_combine(x)

        # Host NVTX intervals correlate graph launches with layer/step/rank.
        # They are not GPU duration measurements.
        profiled_compute = compute
        if os.environ.get("NT4_E2E_PROFILE", "0") == "1":
            compute_call = 0

            def profiled_compute(x, probs):
                nonlocal compute_call
                if not layer._nt4_e2e_active:
                    return compute(x, probs)
                compute_call += 1
                with torch.cuda.nvtx.range(
                    f"nt4_routed_rank{dist.get_rank()}_layer{layer.layer_number}_call{compute_call}_rows{x.shape[0]}"
                ):
                    return compute(x, probs)

        layer.dispatch, layer.routed_experts_compute, layer.combine = (
            dispatch,
            profiled_compute,
            combine,
        )

        def restore():
            layer.dispatch, layer.routed_experts_compute, layer.combine = (
                original_dispatch,
                original_compute,
                original_combine,
            )

        self.restorers.append(restore)
        if self.shortcut_overlap:
            from nt4_shortcut_bridge import install_shortcut_bridge

            def execute_shortcut(x, probs, routes):
                if torch.cuda.is_current_stream_capturing():
                    self.shortcut_capture_layers.add(layer.layer_number)
                d.routing_map = routes
                return compute(x, probs)

            self.restorers.append(
                install_shortcut_bridge(
                    layer.experts,
                    enabled=lambda: layer._nt4_e2e_active,
                    execute=execute_shortcut,
                )
            )
            print(
                json.dumps(
                    dict(
                        stage="megamoe_shortcut_bridge_installed",
                        rank=dist.get_rank(),
                        layer=layer.layer_number,
                        branch=(
                            "NVLS gather/local MegaMoE/NVLS reduce"
                            if os.environ.get("NT4_MXFP8_NVLS_HYBRID", "0") == "1"
                            else "direct dispatch/expert/return"
                        ),
                        prefill=(
                            "native FlashInfer CUTLASS MXFP8+NVLS"
                            if self.precision == "mxfp8"
                            else "native vllm+NVLS"
                        ),
                        stream="caller shortcut stream",
                    )
                ),
                flush=True,
            )
