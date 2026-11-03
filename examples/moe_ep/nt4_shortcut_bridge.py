"""Adapt PR 8257's staged expert interface to the experimental MegaMoE branch.

The ShortcutMoEBlock owns stream fork/join. This bridge never changes streams.
Native prefill/mixed stages retain their original prepared/computed objects.
"""

from dataclasses import dataclass


def install_cutlass_shortcut_bridge(experts, *, enabled):
    """Replace decode expert compute only; caller retains native NVLS and streams."""
    # The PR selects vLLM at configuration time. Explicitly resolve the CUTLASS
    # enum for the experimental decode substitution; prefill remains native.
    previous = getattr(experts, "_flashinfer_activation_type", None)
    had_activation = hasattr(experts, "_flashinfer_activation_type")
    experts._flashinfer_activation_type = experts._resolve_flashinfer_activation_type()

    def execute(x, probs, routes):
        experts._ensure_concatenated_weights()
        if experts._uses_mxfp8_weights:
            raise RuntimeError("CUTLASS shortcut experiment requires BF16 weights")
        # BF16 output is copied into the FP32 symmetric reduction buffer by the
        # unchanged native token_combine path, as in serial CUTLASS execution.
        return experts._flashinfer_forward(x, routes, probs)

    restore_stages = install_shortcut_bridge(experts, enabled=enabled, execute=execute)

    def restore():
        restore_stages()
        if had_activation:
            experts._flashinfer_activation_type = previous
        else:
            del experts._flashinfer_activation_type

    return restore


@dataclass
class PreparedMegaMoE:
    x: object
    probs: object
    routes: object


@dataclass
class ComputedMegaMoE:
    output: object


def install_shortcut_bridge(experts, *, enabled, execute):
    original_prepare = experts.prepare_shortcut_overlap
    original_compute = experts.compute_shortcut_overlap
    original_finish = experts.finish_shortcut_overlap

    def prepare(x, probs, routing_map, *, max_blocks=0):
        if not enabled():
            return original_prepare(x, probs, routing_map, max_blocks=max_blocks)
        # MegaMoE owns dispatch/compute/return. max_blocks is a vLLM kernel knob;
        # it does not control MegaMoE's persistent grid.
        return PreparedMegaMoE(x, probs, routing_map)

    def compute(prepared):
        if isinstance(prepared, PreparedMegaMoE):
            return ComputedMegaMoE(execute(prepared.x, prepared.probs, prepared.routes))
        return original_compute(prepared)

    def finish(computed):
        if isinstance(computed, ComputedMegaMoE):
            return computed.output
        return original_finish(computed)

    experts.prepare_shortcut_overlap = prepare
    experts.compute_shortcut_overlap = compute
    experts.finish_shortcut_overlap = finish

    def restore():
        experts.prepare_shortcut_overlap = original_prepare
        experts.compute_shortcut_overlap = original_compute
        experts.finish_shortcut_overlap = original_finish

    return restore
