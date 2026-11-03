"""Private eager-only workspace tap for NT4 numerical diagnosis, never benchmarking."""

import torch

_last = None
last_tactic = None
forced_tactic = None  # Set only by standalone numerical replay, never E2E.


def select_tactic(tactic):
    global last_tactic
    last_tactic = tactic if forced_tactic is None else forced_tactic
    return last_tactic


def record(buffers, tactic):
    global _last
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("NT4 intermediate diagnostics cannot run inside CUDA graphs")
    if len(buffers) != 5:
        raise RuntimeError("Sync the NT4 diagnostic launcher and Python hook together")
    # Materialize Torch ownership while the FFI result is still live. Do not
    # defer DLPack conversion across the decorated operation's return boundary.
    tensors = []
    names = ("fc1_output", "fc1_scales", "fc2_output", "routing_map", "expert_counts")
    for name, value in zip(names, buffers):
        # Counts are auxiliary metadata, never used by the stagewise analysis.
        # Some launcher paths return an undefined Tensor (Python None).
        if value is None and name == "expert_counts":
            tensors.append(None)
            continue
        if value is None:
            raise RuntimeError(f"NT4 required workspace is missing: {name}")
        try:
            tensors.append(
                value if isinstance(value, torch.Tensor) else torch.from_dlpack(value)
            )
        except Exception as error:
            _last = None
            raise RuntimeError(
                f"NT4 workspace conversion failed for {name}: {type(value)!r}"
            ) from error
    _last = (tensors, tactic)


def take():
    global _last
    if _last is None:
        raise RuntimeError(
            "No TRTLLM workspace captured; check diagnostic binding revision"
        )
    buffers, tactic = _last
    _last = None
    return buffers, tactic
