"""Executable text-generation driver for the NT4 checkpoint capture wrapper.

Uses MCore's VLM/GPT checkpoint detection and engine builder, followed by its
offline dynamic request loop. Accepts the existing NT4 model/prompt arguments.
"""

import json
import os
from pathlib import Path
import sys

import torch


def vlm_language_impl(args):
    """Resolve the language selector for legacy VLM and current MIMO loaders."""
    if getattr(args, "mimo_checkpoint_prefix_map", None) is not None:
        # MIMO's model_provider calls resolve_hybrid_stack_spec with the
        # standard language config; vision modules use separate TE specs.
        return args.transformer_impl
    return getattr(args, "vlm_language_transformer_impl", None)


def main():
    import megatron.training

    root = Path(megatron.training.__file__).resolve().parents[2]
    for relative in ("examples/multimodal", "examples/inference/gpt"):
        sys.path.append(str(root / relative))
    from tools import run_dynamic_text_generation_server as setup
    from examples.inference.advanced.gpt_dynamic_inference import run_inference
    from examples.inference.utils import build_requests
    from megatron.core.inference.sampling_params import SamplingParams

    # Match the server's checkpoint-aware initialization without starting HTTP
    # or a coordinator. Explicit user arguments remain authoritative.
    explicit = {
        token[2:].split("=", 1)[0].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    defaults = []
    if "use_checkpoint_args" not in explicit:
        defaults.append("--use-checkpoint-args")
    if "inference_dynamic_batching" not in explicit:
        defaults.append("--inference-dynamic-batching")
    sys.argv[1:1] = defaults
    args, is_vlm = setup.parse_args_and_detect_vlm(
        setup.add_text_generation_server_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True},
        user_passed_attrs=explicit,
    )
    if args.input_image_path or args.input_prompts_json:
        raise ValueError(
            "checkpoint capture driver currently supports text prompts only"
        )
    if args.prompts and args.prompt_file:
        raise ValueError("use either --prompts or --prompt-file, not both")
    if not args.prompts and not args.prompt_file:
        raise ValueError(
            "supply --prompts or --prompt-file for the checkpoint capture run"
        )
    if is_vlm and vlm_language_impl(args) != "inference_optimized":
        raise RuntimeError(
            "VLM capture requires inference_optimized language layers. For "
            "MIMO checkpoints supported by #7839, set --transformer-impl "
            "inference_optimized. Other VLM loaders require their separate "
            "VLM language-model inference integration."
        )

    # The GPT engine builder already performs this standard inference storage
    # conversion. Apply the same step to the VLM language model before engine
    # construction. BF16 overrides remain untouched. Both compared backends
    # subsequently receive these same loaded inference bytes.
    if (
        is_vlm
        and vlm_language_impl(args) == "inference_optimized"
        and args.fp8_recipe == "mxfp8"
    ):
        original_loader = setup.get_vlm_model

        def load_model(*args_, **kwargs):
            from megatron.core.inference.quantization.utils import (
                quantize_model_to_mxfp8,
                resolve_mxfp8_backend,
            )
            from megatron.core.utils import unwrap_model

            model = original_loader(*args_, **kwargs)
            quantize_model_to_mxfp8(
                unwrap_model(model),
                backend=resolve_mxfp8_backend(args.inference_grouped_gemm_backend),
            )
            return model

        setup.get_vlm_model = load_model
    from megatron.core.inference import utils as inference_utils

    original_cache_check = inference_utils.check_flashinfer_jit_cache_installed
    source_jit = os.environ.get("NT4_ALLOW_SOURCE_JIT") == "1"
    if source_jit:
        # This check only enforces precompiled-cache installation to reduce
        # startup latency. FlashInfer itself still validates and compiles kernels.
        # Keep this development override scoped to model construction.
        def allow_source_jit(log_version=False):
            return None

        inference_utils.check_flashinfer_jit_cache_installed = allow_source_jit
        print(
            json.dumps(
                dict(stage="source_jit_enabled", precompiled_cache_required=False)
            ),
            flush=True,
        )
    try:
        engine = setup._build_engine_for_vlm_or_gpt(is_vlm=is_vlm)
    finally:
        inference_utils.check_flashinfer_jit_cache_installed = original_cache_check
        if is_vlm and "original_loader" in locals():
            setup.get_vlm_model = original_loader
    tokenizer = engine.controller.tokenizer
    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        skip_prompt_log_probs=args.skip_prompt_log_probs,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
        termination_id=(
            args.termination_id if args.termination_id is not None else tokenizer.eod
        ),
        top_n_logprobs=args.top_n_logprobs,
        stop_words=args.stop_words,
    )
    requests = build_requests(args, tokenizer, sampling)
    print(
        json.dumps(
            dict(stage="generation_start", requests=len(requests), is_vlm=is_vlm)
        ),
        flush=True,
    )
    engine.reset()
    result = run_inference(requests, engine)
    torch.cuda.synchronize()
    print(
        json.dumps(
            dict(
                stage="generation_complete",
                total_output_tokens=result["total_output_tokens"],
            )
        ),
        flush=True,
    )
    # Leave process-group lifetime to the capture wrapper / original launcher.


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    with torch.no_grad():
        main()
