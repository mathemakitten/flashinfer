"""Opt-in TE scratch isolation for the NT4 overlap correctness investigation.

The installed TE allocator caches by device/UB/grouped mode, omitting the caller
stream. Keep independent allocations for concurrent streams without serializing
their GEMMs. This is a candidate repair, not evidence that TE caused the failure.
"""

from collections import Counter
import importlib
import inspect
import json
import os


class StreamWorkspaces:
    def __init__(self, allocator, current_stream):
        self.allocate = allocator
        self.current_stream = current_stream
        self.buffers = {}
        self.calls = Counter()

    def __call__(self, device, ub, grouped_gemm):
        stream = self.current_stream(device)
        key = (device, stream.cuda_stream, ub, grouped_gemm)
        self.calls[key] += 1
        if key not in self.buffers:
            # Call the uncached allocator: wrapping TE's cached function would
            # still return the same tensor for different stream keys.
            self.buffers[key] = self.allocate(device, ub, grouped_gemm)
        return self.buffers[key]

    def report(self):
        return [
            dict(device=d, stream=s, ub=u, grouped_gemm=g, calls=count)
            for (d, s, u, g), count in sorted(self.calls.items())
        ]


def install():
    """Install before model construction; retain buffers for process lifetime."""
    import torch

    module = importlib.import_module("transformer_engine.pytorch.cpp_extensions.gemm")
    original = module.get_cublas_workspace
    if isinstance(original, StreamWorkspaces):
        return original
    allocator = getattr(original, "__wrapped__", None)
    if allocator is None or tuple(inspect.signature(allocator).parameters) != (
        "device",
        "ub",
        "grouped_gemm",
    ):
        raise RuntimeError(
            "Unrecognized TE workspace allocator; inspect before patching"
        )
    isolated = StreamWorkspaces(allocator, torch.cuda.current_stream)
    module.get_cublas_workspace = isolated
    return isolated


def emit_report(workspaces):
    print(
        json.dumps(
            dict(
                stage="te_stream_workspace_usage",
                rank=int(os.environ.get("RANK", "0")),
                exercised=bool(workspaces.calls),
                allocations=workspaces.report(),
                note="Zero calls rules out this workspace repair for this execution path",
            )
        ),
        flush=True,
    )
