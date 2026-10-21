"""Experimental eager decode substitution; production prefill stays intact."""

import json
import os
from nt4_gc_policy import controlled_gc
import torch
import torch.distributed as dist

KNOBS = dict(
    mma_tiler_mnk=(128, 256, 128),
    cluster_shape_mnk=(1, 1, 1),
    use_2cta_instrs=False,
    group_hint=1024,
    flag_batch=1,
)

_stage_reduction = os.environ.get("NT4_MXFP8_AB_STAGE_REDUCTION")
if _stage_reduction:
    KNOBS["ab_stage_reduction"] = int(_stage_reduction)

_worker_cap = os.environ.get("NT4_MXFP8_WORKER_CLUSTERS")
if _worker_cap:
    KNOBS["max_active_clusters"] = int(_worker_cap)


def generation_rounds(rounds, *, model_graphs, replay_inputs):
    """Validate ordinary graphs first, then pair forced eager/graph generations."""
    for index in range(-1 if model_graphs else 0, rounds + 1):
        if replay_inputs and index > 0:
            yield index, True
        yield index, False


class DecodeAdapter:
    def __init__(self):
        self.layers = {}
        self.calls = 0
        self.allow_setup = True
        self.restorers = []
        self.precision = "mxfp8"
        self.observation_layer = None
        self.decode_occupancy = {}
        self.weight_formats = {}

    def weights_use_mxfp8(self, layer):
        # This harness loads one checkpoint before installing the adapter and
        # never refits weights. Validate all expert projections once per layer;
        # walking FC1/FC2 parameters on every token was a sampled host hotspot.
        key = id(layer)
        if key not in self.weight_formats:
            self.weight_formats[key] = layer.experts._expert_weights_use_mxfp8()
        return self.weight_formats[key]

    def install(self):
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from nt4_capture_state import snapshot

        adapter = self
        original = MoELayer.preprocess

        def preprocess(layer, *args, **kwargs):
            state = snapshot()
            if adapter.observation_layer is None:
                adapter.observation_layer = id(layer)
            if (
                not getattr(adapter, "model_graphs", False)
                and id(layer) == adapter.observation_layer
                and state is not None
                and state["decode_only"]
            ):
                count = state["active_requests"]
                adapter.decode_occupancy[count] = (
                    adapter.decode_occupancy.get(count, 0) + 1
                )
            eligible = (
                state is not None
                and state["decode_only"]
                and state["prefill_requests"] == 0
                and (adapter.weights_use_mxfp8(layer) == (adapter.precision == "mxfp8"))
            )
            if eligible and id(layer) not in adapter.layers:
                if not adapter.allow_setup:
                    raise RuntimeError(
                        "new eligible MoE layer encountered after warmup"
                    )
                adapter.bind(layer)
            layer._nt4_e2e_active = eligible
            return original(layer, *args, **kwargs)

        MoELayer.preprocess = preprocess
        self.restorers.append(lambda: setattr(MoELayer, "preprocess", original))

    def bind(self, layer):
        from nt4_checkpoint_capture import validate_layer, export_weights
        from nt4_checkpoint_replay import Replay

        validate_layer(layer)
        d = layer.token_dispatcher
        replay = Replay(
            export_weights(layer.experts, dist.get_rank()),
            d.__class__._per_rank_worst_case_token_count,
            d.ep_group,
            knobs=KNOBS,
        )
        original_dispatch = layer.dispatch
        original_compute = layer.routed_experts_compute
        original_combine = layer.combine
        checked = False
        adapter = self

        def dispatch(x, probs):
            if not layer._nt4_e2e_active:
                return original_dispatch(x, probs)
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
            if not layer._nt4_e2e_active:
                return original_compute(x, probs)
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "initial NT4 E2E adapter requires model CUDA graphs disabled"
                )
            ids = d.routing_map
            n = x.shape[0]
            if n > replay.capacity:
                raise ValueError("decode exceeds allocated production capacity")
            if not checked:
                # Untimed first decode call: compare full production routed path
                # with the same input and independently validated MegaMoE output.
                hx, hp = original_dispatch(x, probs)
                out, bias = original_compute(hx, hp)
                expected = original_combine(out).detach().clone()
                d.routing_map = ids
                sample = dict(
                    x=x.detach().cpu(),
                    ids=ids.long().cpu(),
                    probs=torch.where(ids < 0, 0, probs).float().cpu(),
                    baseline=expected.cpu(),
                )
                replay.check(sample, graph=False)
                checked = True
            replay.n = n
            replay.x.zero_()
            replay.ids.fill_(-1)
            replay.probs.zero_()
            replay.x[:n].copy_(x)
            replay.ids[:n].copy_(ids)
            replay.probs[:n].copy_(probs)
            adapter.calls += 1
            return replay.run()[:n], None

        def combine(x):
            return x if layer._nt4_e2e_active else original_combine(x)

        layer.dispatch, layer.routed_experts_compute, layer.combine = (
            dispatch,
            compute,
            combine,
        )

        def restore():
            layer.dispatch = original_dispatch
            layer.routed_experts_compute = original_compute
            layer.combine = original_combine

        self.restorers.append(restore)
        self.layers[id(layer)] = (layer.layer_number, replay)
        print(
            json.dumps(
                dict(
                    rank=dist.get_rank(),
                    stage="e2e_layer_installed",
                    layer=layer.layer_number,
                    capacity=replay.capacity,
                    knobs=KNOBS,
                )
            ),
            flush=True,
        )

    def close(self):
        # All PEs must free symmetric workspaces in the same order while the
        # distributed group and NVSHMEM runtime are still alive.
        torch.cuda.synchronize()
        dist.barrier()
        for restore in reversed(self.restorers):
            restore()
        self.restorers.clear()
        for number, replay in sorted(self.layers.values(), key=lambda entry: entry[0]):
            print(
                json.dumps(
                    dict(rank=dist.get_rank(), stage="e2e_layer_cleanup", layer=number)
                ),
                flush=True,
            )
            replay.close()
        self.layers.clear()
        self.weight_formats.clear()
        torch.cuda.synchronize()
        dist.barrier()


def run(engine, args, tokenizer, sampling, build_requests, run_inference):
    import os
    import time
    from pathlib import Path
    from nt4_capture_state import set_context

    mode = os.environ["NT4_E2E_MODE"]
    if mode not in ("baseline", "direct", "vllm"):
        raise ValueError("invalid E2E mode")
    model_graphs = os.environ.get("NT4_E2E_MODEL_GRAPHS", "0") == "1"
    if not model_graphs and args.cuda_graph_impl != "none":
        raise ValueError("initial E2E test requires --cuda-graph-impl none")
    routed_graphs = os.environ.get("NT4_E2E_ROUTED_GRAPHS", "0") == "1"
    if routed_graphs or model_graphs:
        from nt4_e2e_graph import RoutedGraphAdapter

        adapter = RoutedGraphAdapter(mode)
    else:
        adapter = DecodeAdapter()
    adapter.precision = os.environ.get("NT4_E2E_PRECISION", "mxfp8")
    if (
        mode == "direct"
        and adapter.precision == "bf16"
        and not (routed_graphs or model_graphs)
    ):
        raise ValueError("initial BF16 NT4 E2E path requires --routed-graphs")
    if mode == "direct" or routed_graphs or model_graphs:
        adapter.install()
    destination = Path(os.environ["NT4_E2E_OUTPUT"])
    destination.mkdir(parents=True, exist_ok=True)
    rounds = int(os.environ.get("NT4_E2E_ROUNDS", "3"))
    # Rank 0 controls the profiling session with a single NVTX range.
    # Keep every GPU quiescent while the capture range closes.
    # This group is host-only and is never used inside generation.
    profile_control = None
    if os.environ.get("NT4_E2E_PROFILE", "0") == "1":
        from datetime import timedelta

        profile_control = dist.new_group(backend="gloo", timeout=timedelta(seconds=300))
    if os.environ.get("NT4_TAIL_FIRST", "0") == "1":
        from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock

        blocks = [
            m
            for m in engine.controller._language_model.modules()
            if isinstance(m, ShortcutMoEBlock)
        ]
        if not blocks or adapter.mode != "direct" or not adapter.shortcut_overlap:
            raise RuntimeError("tail-first requires direct MegaMoE shortcut blocks")
        for block in blocks:
            block._nt4_tail_first = True
        print(
            json.dumps(
                dict(
                    stage="tail_first_enabled",
                    rank=dist.get_rank(),
                    blocks=len(blocks),
                    prefill="unchanged",
                )
            ),
            flush=True,
        )
    model_session = None
    if model_graphs:
        from nt4_model_graph import ModelGraphSession

        model_session = ModelGraphSession(engine, adapter)
    variance_audit = None
    if os.environ.get("NT4_VARIANCE_AUDIT", "0") == "1":
        from nt4_variance_audit import VarianceAudit

        variance_audit = VarianceAudit(
            engine,
            destination,
            events=os.environ.get("NT4_AUDIT_MODEL_EVENTS", "0") == "1",
        )
    branch_audit = None
    if os.environ.get("NT4_BRANCH_CONTENTION"):
        from nt4_branch_probe import BranchAudit

        branch_audit = BranchAudit(
            engine, destination, os.environ["NT4_BRANCH_CONTENTION"]
        )
    eager_outputs = None
    forced_eager_outputs = {}
    replay_inputs = branch_audit is not None and branch_audit.input_mode == "replay"
    reports = []
    for index, forced_eager in generation_rounds(
        rounds, model_graphs=model_graphs, replay_inputs=replay_inputs
    ):
        if branch_audit is not None:
            branch_audit.begin_round(index, collect=not forced_eager)
        if model_session is not None:
            if index == 0:
                model_session.build()
            if replay_inputs and index > 0:
                model_session.force_eager = forced_eager
            model_session.begin_round(index >= 0 and not forced_eager)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        engine.reset()
        requests = build_requests(args, tokenizer, sampling)
        adapter.decode_occupancy = {}
        adapter.calls = 0
        if routed_graphs or model_graphs:
            adapter.graph_calls = 0
            adapter.reset_diagnostic_timing()
            adapter.host_timeline = []
        set_context(engine.context)
        if variance_audit is not None:
            variance_audit.begin(index)
        with controlled_gc(
            os.environ.get("NT4_CONTROLLED_GC", "0") == "1" and index > 0
        ) as gc_policy:
            dist.barrier()
            torch.cuda.synchronize()
            profile_round = os.environ.get("NT4_E2E_PROFILE", "0") == "1" and index == 1
            if profile_round:
                import faulthandler

                stack_path = destination / f"profile-stacks-rank{dist.get_rank()}.log"
                stack_file = stack_path.open("w")
                faulthandler.dump_traceback_later(120, repeat=True, file=stack_file)
                print(
                    json.dumps(
                        dict(
                            stage="e2e_profile_context",
                            stack_dump_path=str(stack_path),
                            rank=dist.get_rank(),
                            pid=os.getpid(),
                            cpu_affinity=(
                                sorted(os.sched_getaffinity(0))
                                if hasattr(os, "sched_getaffinity")
                                else None
                            ),
                            torch_threads=torch.get_num_threads(),
                            torch_interop_threads=torch.get_num_interop_threads(),
                        )
                    ),
                    flush=True,
                )
                if dist.get_rank() == 0:
                    torch.cuda.nvtx.range_push("nt4_profile_generation")
                dist.monitored_barrier(group=profile_control, wait_all_ranks=True)
                print(
                    json.dumps(
                        dict(stage="e2e_profiler_started", rank=dist.get_rank())
                    ),
                    flush=True,
                )
                torch.cuda.nvtx.range_push(
                    f"nt4_generation_rank{dist.get_rank()}_round{index}"
                )
            stack_sampler = None
            if os.environ.get("NT4_HOST_STACK_SAMPLING", "0") == "1" and index > 0:
                from nt4_host_stacks import HostStackSampler

                stack_sampler = HostStackSampler(adapter.host_timeline)
                stack_sampler.start()
            if variance_audit is not None:
                variance_audit.mark_generation_start()
            start = time.perf_counter()
            try:
                result = run_inference(requests, engine)
                if profile_round:
                    print(
                        json.dumps(
                            dict(
                                stage="e2e_profile_inference_returned",
                                rank=dist.get_rank(),
                            )
                        ),
                        flush=True,
                    )
            except BaseException:
                if variance_audit is not None:
                    variance_audit.close()
                raise
            finally:
                if stack_sampler is not None:
                    stack_sampler.stop()
                set_context(None)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        if variance_audit is not None:
            variance_audit.end()
        if stack_sampler is not None:
            stack_sampler.save(
                destination / f"host-stacks-round{index}-rank{dist.get_rank()}.json",
                rank=dist.get_rank(),
                round=index,
                mode=mode,
                torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),
            )
        if profile_round:
            torch.cuda.nvtx.range_pop()
            dist.monitored_barrier(group=profile_control, wait_all_ranks=True)
            print(
                json.dumps(dict(stage="e2e_profiler_stop_enter", rank=dist.get_rank())),
                flush=True,
            )
            if dist.get_rank() == 0:
                torch.cuda.nvtx.range_pop()  # Close the session-wide capture range.
            faulthandler.cancel_dump_traceback_later()
            stack_file.close()
            print(
                json.dumps(dict(stage="e2e_profiler_stopped", rank=dist.get_rank())),
                flush=True,
            )
        if profile_round:
            # No CUDA allocation/collective/cleanup before every stop returned.
            dist.monitored_barrier(group=profile_control, wait_all_ranks=True)
            print(
                json.dumps(
                    dict(stage="e2e_profiler_all_stopped", rank=dist.get_rank())
                ),
                flush=True,
            )
        time_tensor = torch.tensor(elapsed, device="cuda", dtype=torch.float64)
        dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
        elapsed = time_tensor.item()
        if mode == "direct" and not model_graphs and not adapter.calls:
            raise RuntimeError("no MegaMoE decode calls executed")
        if routed_graphs and not adapter.graph_calls:
            raise RuntimeError("no routed graph replays executed")
        if routed_graphs and index == 0:
            adapter.capture.finish()
        adapter.allow_setup = False
        report = dict(
            mode=mode,
            precision=adapter.precision,
            decode_active_requests_histogram=adapter.decode_occupancy,
            peak_decode_active_requests=max(adapter.decode_occupancy, default=0),
            requested_batch_reached=(
                (
                    max(adapter.decode_occupancy, default=0)
                    >= int(os.environ.get("NT4_E2E_REQUEST_COUNT", "0"))
                )
                if routed_graphs or model_graphs or mode == "direct"
                else None
            ),
            round=index,
            warmup=index <= 0,
            profiled=profile_round,
            host_stack_sampling=stack_sampler is not None,
            performance_use=(
                "diagnostic_only"
                if stack_sampler is not None
                or variance_audit is not None
                or branch_audit is not None
                else "timing"
            ),
            routed_diagnostic_timing=(
                adapter.diagnostic_timing() if routed_graphs else None
            ),
            wall_seconds=elapsed,
            total_output_tokens=result["total_output_tokens"],
            output_tokens_per_second=result["total_output_tokens"] / elapsed,
            step_times=(
                result["step_times"]
                if any(
                    v > 0 for values in result["step_times"].values() for v in values
                )
                else None
            ),
            step_timing_note="Zero engine timings mean instrumentation unavailable; use synchronized wall time",
            grouped_gemm_backend=str(args.inference_grouped_gemm_backend),
            backend_note=(
                "FlashInfer CUTLASS decode experts + native NVLS; vLLM prefill"
                if os.environ.get("NT4_CUTLASS_OVERLAP", "0") == "1"
                else (
                    "vllm selection: BF16 vLLM; MXFP8 MCore scaled grouped GEMM"
                    if mode == "vllm"
                    else mode
                )
            ),
            megamoe_calls=None if model_graphs else adapter.calls,
            megamoe_layers=(
                sorted(n for n, _ in adapter.layers.values())
                if mode == "direct"
                else []
            ),
            routed_graphs=routed_graphs,
            routed_graph_calls=adapter.graph_calls if routed_graphs else 0,
            routed_graph_count=adapter.graph_count if routed_graphs else 0,
            graph_scope=(
                "whole language-model decode forward; eager prefill and request loop"
                if model_graphs
                else (
                    "decode routed dispatch/expert/combine; live input copies timed outside"
                    if routed_graphs
                    else None
                )
            ),
            model_graphs=model_graphs,
            cutlass_max_blocks=os.environ.get("NT4_CUTLASS_MAX_BLOCKS"),
            gc_policy=gc_policy,
            shortcut_overlap=os.environ.get("NT4_SHORTCUT_OVERLAP", "0") == "1",
            substitution=(
                "BF16 CUTLASS experts during decode only; native NVLS dispatch/combine"
                if os.environ.get("NT4_CUTLASS_OVERLAP", "0") == "1"
                else (
                    f"{adapter.precision} routed experts during decode only"
                    if mode == "direct"
                    else "none"
                )
            ),
        )
        if routed_graphs and adapter.host_timeline_enabled and index > 0:
            import socket

            timeline_path = (
                destination / f"host-timeline-round{index}-rank{dist.get_rank()}.json"
            )
            timeline_path.write_text(
                json.dumps(
                    dict(
                        schema=1,
                        hostname=socket.gethostname(),
                        pid=os.getpid(),
                        rank=dist.get_rank(),
                        round=index,
                        mode=mode,
                        clock="time.monotonic_ns; comparable on the same host only",
                        scope="CPU entry/copy submission/graph submission; not GPU execution duration",
                        events=adapter.host_timeline,
                    )
                )
            )
            report["host_timeline"] = str(timeline_path)
        outputs = []
        for request in requests:
            tokens = request.output_tokens
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.cpu().tolist()
            if tokens is None:
                raise RuntimeError("generation returned no token sequence")
            outputs.append(tokens)
        if forced_eager:
            forced_eager_outputs[index] = outputs
            (
                destination / f"forced-eager-round{index}-rank{dist.get_rank()}.json"
            ).write_text(json.dumps(outputs))
            print(
                json.dumps(
                    dict(
                        stage="forced_input_eager_reference_complete",
                        rank=dist.get_rank(),
                        round=index,
                    )
                ),
                flush=True,
            )
            continue  # Reference only: no timing rows, E2E reports or graph claims.
        if (
            os.environ.get("NT4_EAGER_RANK_AUDIT", "0") == "1"
            or branch_audit is not None
        ):
            (
                destination / f"outputs-round{index}-rank{dist.get_rank()}.json"
            ).write_text(json.dumps(outputs))
        if model_session is not None:
            if dist.get_rank() == 0:
                (destination / f"outputs-round{index}.json").write_text(
                    json.dumps(outputs)
                )
            report.update(model_session.report())
            if index == -1:
                eager_outputs = outputs
            else:
                reference_outputs = (
                    forced_eager_outputs[index]
                    if replay_inputs and index > 0
                    else eager_outputs
                )
                report["model_graph_reference"] = (
                    "recorded_inputs_eager"
                    if replay_inputs and index > 0
                    else "ordinary_eager"
                )
                mismatch = torch.tensor(
                    int(outputs != reference_outputs), device="cuda"
                )
                dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
                if mismatch.item():
                    # The gate covers every rank, while ordinary output files
                    # contain only rank 0. Preserve the evidence on every rank.
                    details = []
                    for request_id, (expected, actual) in enumerate(
                        zip(reference_outputs, outputs)
                    ):
                        if expected != actual:
                            first = next(
                                (
                                    i
                                    for i, (a, b) in enumerate(zip(expected, actual))
                                    if a != b
                                ),
                                min(len(expected), len(actual)),
                            )
                            details.append(
                                dict(request=request_id, first_mismatch=first)
                            )
                    diagnostic = dict(
                        rank=dist.get_rank(),
                        round=index,
                        eager_request_count=len(reference_outputs),
                        graph_request_count=len(outputs),
                        mismatches=details,
                        eager_outputs=reference_outputs,
                        reference_policy=report["model_graph_reference"],
                        graph_outputs=outputs,
                    )
                    path = (
                        destination
                        / f"graph-mismatch-round{index}-rank{dist.get_rank()}.json"
                    )
                    path.write_text(json.dumps(diagnostic))
                    print(
                        json.dumps(
                            dict(
                                stage="model_graph_mismatch",
                                rank=dist.get_rank(),
                                mismatched_requests=len(details),
                                diagnostic=str(path),
                            )
                        ),
                        flush=True,
                    )
                    raise RuntimeError(
                        "whole-model graph generation differs from eager reference; "
                        f"do not use timings; see {path}"
                    )
                if model_session.replays == 0:
                    raise RuntimeError("no whole-model graph replays observed")
                report["model_graph_vs_eager_outputs"] = "EXACT"
        reports.append(report)
        if dist.get_rank() == 0:
            (destination / f"outputs-round{index}.json").write_text(json.dumps(outputs))
            (destination / "results.json").write_text(
                json.dumps(reports, indent=2, default=str)
            )
            print(
                json.dumps(dict(stage="e2e_result", **report), default=str), flush=True
            )
    if variance_audit is not None:
        variance_audit.close()
    if model_session is not None:
        model_session.close()
    adapter.close()
    if profile_control is not None:
        dist.destroy_process_group(profile_control)
    dist.destroy_process_group()
    print(json.dumps(dict(stage="e2e_cleanup_complete", mode=mode)), flush=True)
