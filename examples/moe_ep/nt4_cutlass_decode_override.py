"""Instance-local experimental CUTLASS tactics; no global MCore/FI patches."""

from types import FunctionType, MethodType


def _with_globals(function, replacements):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    clone = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def install_decode_override(experts, profile_ids, *, enabled):
    """Select tactics at eager decode/capture time; prefill keeps native dispatch.

    Clone the two Python call sites with private globals so other expert instances
    and callers retain their original functions. Captured graphs retain the
    selected launches; the existing adapter supplies the decode-only predicate.
    """
    if len(profile_ids) != 2 or any(type(x) is not int or x < 0 for x in profile_ids):
        raise ValueError("expected two nonnegative CUTLASS tactic IDs")
    profile_ids = tuple(profile_ids)
    original = experts._flashinfer_forward
    method = original.__func__
    helper = method.__globals__["flashinfer_cutlass_mxfp8_moe"]
    cutlass = helper.__globals__["cutlass_fused_moe"]

    def selected_cutlass(*args, **kwargs):
        kwargs["profile_ids"] = list(profile_ids)
        return cutlass(*args, **kwargs)

    tuned_helper = _with_globals(helper, {"cutlass_fused_moe": selected_cutlass})

    def select_helper(*args, **kwargs):
        return (tuned_helper if enabled() else helper)(*args, **kwargs)

    replaced = _with_globals(method, {"flashinfer_cutlass_mxfp8_moe": select_helper})
    had_override = "_flashinfer_forward" in vars(experts)
    previous_override = vars(experts).get("_flashinfer_forward")
    experts._flashinfer_forward = MethodType(replaced, experts)

    def restore():
        if had_override:
            experts._flashinfer_forward = previous_override
        else:
            del experts._flashinfer_forward

    return restore
