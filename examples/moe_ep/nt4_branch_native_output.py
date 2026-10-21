"""Diagnostic continuation only: native clamped output after the timed join.

Installed on all sampled blocks in every surrogate comparison arm. Input copies
are common overhead inside route_dispatch. The native rerun occurs on the main
stream after both branches join, outside every reported branch interval.
"""


def install_native_output(block, probe):
    mlp = block.moe_layer.mlp
    # Installation precedes InferenceMode and MCore dispatcher selection.
    # Resolve the active dispatcher only after the live route/preprocess call.
    dispatcher = None
    original_route = block._moe_router_preprocess
    original_post = block._postprocess
    saved = None
    output = None
    # Resolve class implementations before installing wrappers. Instance methods
    # may be replaced by the direct adapter (identity dispatch/combine).
    dispatch = type(mlp).dispatch
    compute = type(mlp).routed_experts_compute
    combine = type(mlp).combine

    def route(*args, **kwargs):
        nonlocal saved, output, dispatcher
        saved = output = None
        x, probs = original_route(*args, **kwargs)
        if getattr(mlp, "_nt4_e2e_active", False):
            dispatcher = mlp.token_dispatcher
            if type(dispatcher).__name__ != "NVLSAllGatherVDispatcher":
                raise ValueError(
                    "native continuation requires active NVLS during decode; got "
                    + type(dispatcher).__name__
                )
            # NVLS dispatch replaces routing_map with its gathered buffer. Keep
            # local inputs independently owned, including padding mask inputs.
            saved = (x.clone(), probs.clone(), dispatcher.routing_map.clone())
        return x, probs

    def after_join():
        nonlocal output
        if saved is None:
            raise RuntimeError("native continuation join has no saved local inputs")
        if mlp.token_dispatcher is not dispatcher:
            raise RuntimeError(
                "dispatcher changed between route and native continuation"
            )
        import torch

        x, probs, routes = saved
        # Clones were produced on the shortcut stream and are consumed here on
        # the joined main stream. Keep their allocation alive through this use.
        for tensor in saved:
            tensor.record_stream(torch.cuda.current_stream())
        dispatcher.routing_map = routes
        hx, hp = dispatch(mlp, x, probs)
        out, bias = compute(mlp, hx, hp)
        if bias is not None:
            raise ValueError("native continuation does not support expert bias")
        output = combine(mlp, out)
        return output

    def post(*args, **kwargs):
        if saved is not None:
            if output is None:
                raise RuntimeError("native continuation must run after the timed join")
            # MCore uses keyword arguments at the instrumented overlap join.
            if args or "combined_output" not in kwargs:
                raise RuntimeError("unexpected shortcut postprocess call convention")
            kwargs["combined_output"] = output
        return original_post(*args, **kwargs)

    block._moe_router_preprocess = route
    block._postprocess = post
    probe.after_join = after_join
