"""Diagnostic graph segmentation at an executed HybridStack boundary.

Preserves model inputs and computations, but changes graph allocation/capture and
replay synchronization. A disappearing failure is evidence of sensitivity to that
change, not a clean bill of health. Never use segmented timings for performance.
"""
import json
import os


class SegmentedGraph:
    def __init__(self, session):
        import torch
        self.session = session
        self.graphs = [torch.cuda.CUDAGraph()]
        self.generators = []
        self.retained = []
        self.capturing = False
        self.replays = 0

    def register_generator_state(self, generator):
        self.generators.append(generator)
        self.graphs[-1].register_generator_state(generator)

    def capture_begin(self, *args, **kwargs):
        self.begin_args = args
        self.begin_kwargs = dict(kwargs)
        self.graphs[-1].capture_begin(*args, **kwargs)
        self.capturing = True
        self.session.segment_active = self

    def split(self, output):
        import torch
        if len(self.graphs) != 1:
            raise RuntimeError("graph split boundary executed more than once")
        # Keep prefix outputs live through suffix capture and all later replay.
        self.retained.append(output)
        self.graphs[-1].capture_end()
        following = torch.cuda.CUDAGraph()
        for generator in self.generators:
            following.register_generator_state(generator)
        kwargs = dict(self.begin_kwargs)
        if kwargs.get("pool") is None:
            kwargs["pool"] = self.graphs[0].pool()
        self.graphs.append(following)
        following.capture_begin(*self.begin_args, **kwargs)

    def capture_end(self):
        self.graphs[-1].capture_end()
        self.capturing = False
        self.session.segment_active = None
        if len(self.graphs) != 2:
            raise RuntimeError("requested model graph split was not reached")

    def replay(self):
        import torch
        self.replays += 1
        for index, graph in enumerate(self.graphs):
            fields = dict(stage="model_graph_segment", rank=int(os.environ.get("RANK", -1)),
                          segment=index, split_after_block=self.session.segment_split,
                          forward_index=self.session.forward_index, replay=self.replays,
                          performance_use=False)
            location = "before_sync"
            try:
                torch.cuda.synchronize()
                location = "replay"
                if self.replays <= 2:
                    print(json.dumps(dict(fields, event="enter")), flush=True)
                graph.replay()
                location = "after_sync"
                torch.cuda.synchronize()
                if self.replays <= 2:
                    print(json.dumps(dict(fields, event="complete")), flush=True)
            except BaseException as error:
                print(json.dumps(dict(fields, event="failed", location=location,
                                      error=str(error))), flush=True)
                raise

    def reset(self):
        for graph in self.graphs:
            graph.reset()
        self.retained.clear()

    def __getattr__(self, name):
        return getattr(self.graphs[-1], name)


def find_stack(model, stack_type):
    stacks = [module for module in model.modules() if isinstance(module, stack_type)]
    if len(stacks) != 1:
        found = sorted({type(module).__name__ for module in model.modules()})
        raise RuntimeError(
            f"graph bisection needs one {stack_type.__name__}, found {len(stacks)}; "
            f"model module types: {found}"
        )
    return stacks[0]


def install(session, specification):
    from megatron.core.models.hybrid.hybrid_block import HybridStack

    layers = list(find_stack(session.model, HybridStack).layers)
    split = len(layers) // 2 if specification == "mid" else int(specification)
    if not 1 <= split < len(layers):
        raise ValueError(f"split must be an executed block ordinal in 1..{len(layers)-1}")
    session.segment_split = split
    session.segment_active = None

    def boundary(module, args, output):
        graph = session.segment_active
        if graph is not None and graph.capturing:
            graph.split(output)

    session.segment_hook = layers[split-1].register_forward_hook(boundary)
    print(json.dumps(dict(stage="model_graph_split_config", rank=int(os.environ.get("RANK", -1)),
                          split_after_block=split, executed_blocks=len(layers),
                          block_type=type(layers[split-1]).__name__, performance_use=False)), flush=True)
