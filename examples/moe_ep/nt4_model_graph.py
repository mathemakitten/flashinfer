"""MCore whole-language-model decode graphs for the fixed-checkpoint NT4 harness."""

from contextlib import contextmanager
import json
import os


@contextmanager
def defer_model_graphs():
    # Construct normal graph-capable model/context, but install the experimental
    # backend and prepare it before MCore captures any forwards.
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine

    original = DynamicInferenceEngine.create_cuda_graphs
    DynamicInferenceEngine.create_cuda_graphs = lambda self, *a, **kw: None
    try:
        yield
    finally:
        DynamicInferenceEngine.create_cuda_graphs = original


@contextmanager
def synchronized_boundary(enabled, operation, **fields):
    """Diagnostic only: attribute asynchronous faults to a bounded operation."""
    if not enabled:
        yield
        return
    import torch

    def emit(event, **extra):
        print(json.dumps(dict(stage="model_cuda_boundary", rank=int(os.environ["RANK"]),
                              operation=operation, event=event, performance_use=False,
                              **fields, **extra)), flush=True)

    location = "before_sync"
    try:
        emit(location)
        torch.cuda.synchronize()
        emit("enter")
        location = "operation"
        yield
        location = "after_sync"
        emit(location)
        torch.cuda.synchronize()
        emit("complete")
    except BaseException as error:
        # No further CUDA calls after an error: retain the original traceback.
        emit("failed", location=location, error=str(error))
        raise


class ModelGraphSession:
    def __init__(self, engine, adapter):
        self.engine = engine
        self.adapter = adapter
        self.model = engine.controller._language_model
        self.manager = getattr(self.model, "cudagraph_manager", None)
        if self.manager is None:
            raise RuntimeError("language model has no MCore whole-model graph manager")
        if self.manager.cudagraph_runners:
            raise RuntimeError("model was captured before backend installation")
        self.force_eager = True
        self.measuring = False
        self.replays = 0
        self.decode_steps = 0
        self.wrapped = []
        self.building = False
        self.forward_index = 0
        self.sync_boundaries = os.environ.get("NT4_MODEL_SYNC_BOUNDARIES") == "1"
        self.private_capture_pools = os.environ.get("NT4_MODEL_PRIVATE_CAPTURE_POOLS") == "1"
        self.segment_spec = os.environ.get("NT4_MODEL_GRAPH_SPLIT_AFTER")
        self.segment_hook = None
        if self.segment_spec:
            from nt4_graph_bisect import install
            install(self, self.segment_spec)
        self.original_select = self.model._should_call_local_cudagraph
        self.original_get_runner = self.manager.get_cudagraph_runner
        self.original_forward = engine.controller._dynamic_step_forward_logits

        def select(*args, **kwargs):
            return False if self.force_eager else self.original_select(*args, **kwargs)

        def get_runner(*args, **kwargs):
            runner = self.original_get_runner(*args, **kwargs)
            if self.measuring and not runner.fwd_graph_recorded:
                raise RuntimeError("new model graph during measured generation")
            if not any(runner is r for r, _ in self.wrapped):
                if self.segment_spec:
                    from nt4_graph_bisect import SegmentedGraph
                    if runner.grad_enabled:
                        raise RuntimeError("graph segmentation is inference-only")
                    runner._diagnostic_graph_factory = lambda: SegmentedGraph(self)
                if self.private_capture_pools:
                    import torch

                    if runner.fwd_graph_recorded or runner.cudagraph_created:
                        raise RuntimeError("private capture pool requested after graph recording")
                    # Diagnostic: isolate captured temporary allocations across runners.
                    # MCore's explicit static-input allocations still use its global pool.
                    runner.mempool = torch.cuda.graph_pool_handle()
                original = runner.replay_graph_capture

                def replay(*a, **kw):
                    with synchronized_boundary(
                        self.sync_boundaries and not self.building,
                        "graph_replay", forward_index=self.forward_index,
                        runner_index=next(i for i, (r, _) in enumerate(self.wrapped) if r is runner),
                    ):
                        result = original(*a, **kw)
                    if self.measuring:
                        self.replays += 1
                    return result

                runner.replay_graph_capture = replay
                self.wrapped.append((runner, original))
            return runner

        def forward(*args, **kwargs):
            context = engine.context
            decode = context.is_decode_only()
            if decode:
                count = int(context.get_active_request_count())
                adapter.decode_occupancy[count] = (
                    adapter.decode_occupancy.get(count, 0) + 1
                )
            before = self.replays
            self.forward_index += 1
            with synchronized_boundary(
                self.sync_boundaries and not self.building,
                "model_forward", forward_index=self.forward_index,
                phase="decode" if decode else "prefill_or_mixed",
                force_eager=self.force_eager,
            ):
                result = self.original_forward(*args, **kwargs)
            if self.measuring and decode:
                self.decode_steps += 1
                if self.replays != before + 1:
                    raise RuntimeError(
                        "decode did not replay exactly one whole-model graph"
                    )
            return result

        self.model._should_call_local_cudagraph = select
        self.manager.get_cudagraph_runner = get_runner
        engine.controller._dynamic_step_forward_logits = forward

    def build(self):
        import torch
        import torch.distributed as dist
        from nt4_capture_state import set_context

        if self.manager.cudagraph_runners:
            raise RuntimeError("eager reference unexpectedly created model graphs")
        self.force_eager = False
        self.adapter.allow_setup = True
        set_context(self.engine.context)
        self.building = True
        try:
            self.engine.create_cuda_graphs()
        finally:
            self.building = False
            set_context(None)
        torch.cuda.synchronize()
        dist.barrier()
        if not self.manager.cudagraph_runners or not self.adapter.model_capture_layers:
            raise RuntimeError("model graph capture did not visit the routed adapter")
        expected = {number for number, _ in self.adapter.layers.values()}
        if self.adapter.shortcut_overlap and self.adapter.mode == "direct":
            expected_bridges = {
                number
                for number, resources in self.adapter.layers.values()
                if resources.replay is not None
            }
            if self.adapter.shortcut_capture_layers != expected_bridges:
                raise RuntimeError(
                    "model graphs did not capture the MegaMoE shortcut bridge in every adapted layer"
                )
        for key, layers in self.adapter.model_capture_layers.items():
            if layers != expected:
                raise RuntimeError(
                    f"model capture {key} missed layers: {expected - layers}"
                )
        self.adapter.allow_setup = False
        self.engine.reset()
        print(
            json.dumps(
                dict(
                    stage="model_graph_ready",
                    rank=dist.get_rank(),
                    backend=self.adapter.mode,
                    shortcut_overlap=self.adapter.shortcut_overlap,
                    shortcut_bridge_layers=sorted(self.adapter.shortcut_capture_layers),
                    graphs=len(self.manager.cudagraph_runners),
                    capture_pool_policy=(
                        "per_runner_private" if self.private_capture_pools else "mcore_default"
                    ),
                    static_input_pool_policy="mcore_default",
                    captured_layers={
                        str(k): sorted(v)
                        for k, v in self.adapter.model_capture_layers.items()
                    },
                    diagnostic_split_after_block=getattr(self, "segment_split", None),
                    performance_use=not bool(self.segment_spec),
                    scope="whole language-model decode forward; eager prefill and request loop",
                )
            ),
            flush=True,
        )

    def begin_round(self, measured):
        self.measuring = measured
        self.replays = 0
        self.decode_steps = 0

    def report(self):
        return dict(
            model_graph_replays=self.replays,
            model_decode_steps=self.decode_steps,
            model_graph_count=len(self.manager.cudagraph_runners),
        )

    def close(self):
        import torch
        from megatron.core.transformer.cuda_graphs import delete_cuda_graphs

        torch.cuda.synchronize()
        # Graphs reference adapter symmetric allocations: destroy them first.
        delete_cuda_graphs()
        if self.segment_hook is not None:
            self.segment_hook.remove()
        for runner, original in self.wrapped:
            runner.replay_graph_capture = original
            if self.segment_spec:
                del runner._diagnostic_graph_factory
        self.model._should_call_local_cudagraph = self.original_select
        self.manager.get_cudagraph_runner = self.original_get_runner
        self.engine.controller._dynamic_step_forward_logits = self.original_forward
