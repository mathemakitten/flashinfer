"""Run eager full-generation baseline/direct trials from captured model arguments."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from nt4_three_way import build_command, check_source_headers, replace_model_options


def trial_modes(precision):
    modes = [("baseline-start", "baseline")]
    modes.append(("direct", "direct"))
    return modes + [("vllm-nvls", "vllm"), ("baseline-end", "baseline")]


def alternating_overlap_trials(pairs):
    return [
        (f"{index:02d}-{mode}", mode)
        for index in range(pairs)
        for mode in ("vllm", "direct")
    ]


def hybrid_overlap_trials(pairs):
    return [
        item
        for i in range(pairs)
        for item in (
            ("baseline-start" if i == 0 else f"{i:02d}-cutlass", "baseline"),
            (f"{i:02d}-hybrid", "direct"),
        )
    ]


def precision_options(model, precision):
    if precision == "bf16":
        return replace_model_options(
            model,
            {
                "--fp8-recipe": None,
                "--fp8-format": None,
                "--fp8-param-gather": None,
                "--fp8": None,
                "--fp8-amax-history-len": None,
                "--fp8-amax-compute-algo": None,
            },
        )
    return replace_model_options(
        model, {"--fp8-recipe": "mxfp8", "--fp8-format": "e4m3"}
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mxfp8-nvls-hybrid", action="store_true")
    p.add_argument("--trtllm-relu2-surrogate", action="store_true")
    p.add_argument(
        "--trtllm-clamped-compare",
        action="store_true",
        help="Compare actual NT4 TRTLLM + NVLS and MegaMoE hybrid with tuned CUTLASS controls",
    )
    p.add_argument("--branch-native-output", action="store_true")
    p.add_argument("--mxfp8-hybrid-pairs", type=int, default=0)
    p.add_argument("--source-artifact", type=Path, required=True)
    p.add_argument("--mcore-root", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--cutlass-decode-profile-ids",
        type=int,
        nargs=2,
        metavar=("FC1", "FC2"),
        help="Bracket a decode-only CUTLASS tactic pair with default controls (MXFP8 overlap)",
    )
    p.add_argument("--request-count", type=int, default=128)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument(
        "--controlled-gc",
        action="store_true",
        help="Collect cyclic garbage before each measured round barrier, disable automatic cyclic GC during generation, then restore the prior state; applies to all arms",
    )
    p.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=128,
        help="Generated tokens per request (EOS ignored), independent of request count and scheduler token budget",
    )
    p.add_argument(
        "--vllm-variance-audit",
        type=int,
        default=0,
        help="Number of fresh vLLM-only launches with configuration and GPU telemetry",
    )
    p.add_argument(
        "--audit-model-events",
        action="store_true",
        help="Add per-forward CUDA/host intervals to the variance audit; diagnostic only",
    )
    p.add_argument(
        "--alternating-overlap-pairs",
        type=int,
        default=0,
        help="Repeat vLLM/tuned MegaMoE pairs; fresh warmed process per arm, not co-resident",
    )
    p.add_argument("--timeout-seconds", type=int, default=7200)
    p.add_argument(
        "--routed-graphs",
        action="store_true",
        help="Graph each decode routed-MoE section on every backend; surrounding model stays eager",
    )
    p.add_argument(
        "--model-graphs",
        action="store_true",
        help="MCore whole-language-model decode graphs on every backend; eager prefill",
    )
    p.add_argument(
        "--shortcut-overlap",
        action="store_true",
        help="PR 8257 BF16 shortcut overlap: vLLM+NVLS versus MegaMoE decode on the shortcut stream",
    )
    p.add_argument(
        "--cutlass-overlap-pair",
        action="store_true",
        help="vLLM/CUTLASS/vLLM decode comparison with native NVLS and shortcut overlap",
    )
    p.add_argument(
        "--cutlass-max-blocks",
        type=int,
        default=None,
        help="Experimental SM103 CUTLASS FC1/FC2 grid cap; 0 uncapped, otherwise multiple of 16. Only with --cutlass-overlap-pair.",
    )
    p.add_argument(
        "--mxfp8-worker-clusters",
        type=int,
        help="NT4 MXFP8 matched-input direct worker cap",
    )
    p.add_argument("--mxfp8-ab-stage-reduction", type=int, choices=(0, 1, 2))
    p.add_argument("--mxfp8-worker-confirm", action="store_true")
    p.add_argument("--precision", choices=("mxfp8", "bf16"), default="mxfp8")
    p.add_argument("--model-token-budget", type=int, default=1024)
    p.add_argument(
        "--bf16-knobs-json",
        type=Path,
        help="Candidate knobs; adds an unchanged direct-control arm",
    )
    p.add_argument(
        "--capture-bf16-only",
        action="store_true",
        help="Run baseline only and save real tuning samples during warmup",
    )
    p.add_argument(
        "--diagnostic-timing-layers",
        default="",
        help="Comma-separated layers for in-context GPU timing; perturbs throughput, diagnostic only",
    )
    p.add_argument(
        "--profile-direct-pair",
        action="store_true",
        help="Nsight Systems traces of unchanged/tuned BF16 only; first post-warmup generation",
    )
    p.add_argument(
        "--profile-graph-trace",
        choices=("graph", "node"),
        default="graph",
        help="Graph-level tracing by default; node tracing is more intrusive",
    )
    p.add_argument(
        "--profile-trace",
        choices=("cuda", "host"),
        default="cuda",
        help="host records NVTX/OS runtime without CUDA collection; cannot measure GPU durations",
    )
    p.add_argument(
        "--host-timeline-pair",
        action="store_true",
        help="Unchanged/tuned BF16 CPU timestamp timelines; no Nsight or CUDA timing events",
    )
    p.add_argument(
        "--host-stack-sampling",
        action="store_true",
        help="Add bounded Python stack samples on all ranks to --host-timeline-pair; diagnostic only",
    )
    p.add_argument("--capture-layers", default="7,32,68")
    p.add_argument("--capture-samples", type=int, default=2)
    p.add_argument(
        "--eager-rank-audit",
        action="store_true",
        help="Run only eager BF16 vLLM overlap and save outputs on every rank",
    )
    p.add_argument(
        "--branch-contention",
        choices=("serial", "overlap"),
        help="Diagnostic phase timing of shortcut pairs",
    )
    p.add_argument(
        "--branch-backend", choices=("direct", "vllm", "baseline"), default="direct"
    )
    p.add_argument("--branch-inputs", type=Path)
    p.add_argument("--branch-input-mode", choices=("record", "replay"))
    p.add_argument(
        "--tail-first-pair",
        action="store_true",
        help="ABBA direct MegaMoE comparison: original vs tail-first launch order, identical knobs",
    )
    p.add_argument(
        "--overlap-headroom-sweep",
        action="store_true",
        help="BF16 group1024 cluster sweep with full-worker and vLLM overlap controls",
    )
    a = p.parse_args()
    if a.overlap_headroom_sweep:
        if not (
            a.precision == "bf16"
            and a.shortcut_overlap
            and a.model_graphs
            and a.controlled_gc
            and a.rounds > 0
        ):
            p.error(
                "headroom sweep requires BF16, shortcut overlap, model graphs and controlled GC"
            )
        if any(
            (
                a.bf16_knobs_json,
                a.tail_first_pair,
                a.branch_contention,
                a.alternating_overlap_pairs,
                a.cutlass_overlap_pair,
                a.vllm_variance_audit,
                a.eager_rank_audit,
                a.profile_direct_pair,
                a.host_timeline_pair,
                a.capture_bf16_only,
            )
        ):
            p.error(
                "headroom sweep supplies its own knobs and trials; omit other experiment flags"
            )
    if a.tail_first_pair:
        if not (
            a.precision == "bf16"
            and a.shortcut_overlap
            and a.model_graphs
            and a.bf16_knobs_json
            and a.rounds > 0
        ):
            p.error(
                "tail-first pair requires BF16, model graphs, shortcut overlap and knobs"
            )
        if any(
            (
                a.branch_contention,
                a.alternating_overlap_pairs,
                a.cutlass_overlap_pair,
                a.vllm_variance_audit,
                a.eager_rank_audit,
                a.profile_direct_pair,
                a.host_timeline_pair,
                a.capture_bf16_only,
            )
        ):
            p.error("tail-first pair must run without other comparison/profiling modes")
        source = (
            a.mcore_root / "megatron/core/models/hybrid/shortcut_block.py"
        ).read_text()
        if "_nt4_tail_first" not in source:
            p.error("sync the MCore tail-first schedule change before running")
    if (a.branch_inputs is None) != (a.branch_input_mode is None):
        p.error("--branch-inputs and --branch-input-mode must be specified together")
    if (a.branch_inputs or a.branch_backend != "direct") and not a.branch_contention:
        p.error("branch input/backend options require --branch-contention")
    if a.mxfp8_nvls_hybrid and not (
        a.precision == "mxfp8"
        and (
            (
                a.branch_contention
                and a.branch_backend == "direct"
                and a.branch_input_mode == "replay"
            )
            or (
                not a.branch_contention
                and a.controlled_gc
                and a.cutlass_decode_profile_ids is not None
                and not any(
                    (
                        a.mxfp8_worker_confirm,
                        a.profile_direct_pair,
                        a.host_timeline_pair,
                        a.capture_bf16_only,
                        a.diagnostic_timing_layers,
                        a.eager_rank_audit,
                    )
                )
            )
        )
        and a.model_graphs
        and a.shortcut_overlap
        and a.mxfp8_worker_clusters is None
        and a.mxfp8_ab_stage_reduction is None
    ):
        p.error(
            "NVLS hybrid requires MXFP8 model graphs/shortcut overlap, and either direct branch replay or controlled-GC confirmation with CUTLASS tactics; no resource overrides"
        )
    if a.branch_native_output and not (
        a.precision == "mxfp8"
        and a.model_graphs
        and a.shortcut_overlap
        and a.branch_contention
        and a.branch_input_mode == "replay"
        and a.branch_backend in ("baseline", "direct")
        and (a.branch_backend == "baseline" or a.mxfp8_nvls_hybrid)
    ):
        p.error(
            "native output isolation requires MXFP8 matched-input shortcut branch replay"
        )
    if a.branch_native_output and a.cutlass_decode_profile_ids not in (None, [39, 428]):
        p.error("native continuation protocol uses CUTLASS tactics [39,428]")
    if a.trtllm_clamped_compare and not (
        a.mxfp8_nvls_hybrid
        and not a.branch_contention
        and a.cutlass_decode_profile_ids == [39, 428]
        and not a.trtllm_relu2_surrogate
        and not a.mxfp8_hybrid_pairs
        and not a.alternating_overlap_pairs
        and os.environ.get("FLASHINFER_TRTLLM_NT4_EXPORT")
    ):
        p.error(
            "clamped comparison requires MXFP8 hybrid E2E and FLASHINFER_TRTLLM_NT4_EXPORT; no surrogate/branch/alternating modes"
        )
    if a.trtllm_relu2_surrogate and not (
        a.branch_native_output
        and a.mxfp8_nvls_hybrid
        and a.branch_contention
        and a.branch_input_mode == "replay"
        and a.branch_backend == "direct"
    ):
        p.error(
            "TRTLLM Relu2 surrogate requires --branch-native-output and MXFP8 hybrid matched-input branch replay"
        )
    if a.mxfp8_hybrid_pairs < 0 or (
        a.mxfp8_hybrid_pairs
        and not (
            a.mxfp8_nvls_hybrid
            and not a.branch_contention
            and not a.alternating_overlap_pairs
            and not a.trtllm_relu2_surrogate
        )
    ):
        p.error(
            "hybrid pairs require MXFP8 hybrid E2E confirmation without other alternating or surrogate modes"
        )
    native_branch_backend = "baseline" if a.precision == "mxfp8" else "vllm"
    if a.branch_contention and a.branch_backend not in (
        "direct",
        native_branch_backend,
    ):
        p.error(
            "branch backend must be direct or the native backend for this precision"
        )
    if a.branch_input_mode == "record" and a.branch_backend != native_branch_backend:
        p.error(
            "record the shared input sequence with the native backend for this precision"
        )
    if a.branch_contention and any(
        (
            a.profile_direct_pair,
            a.host_timeline_pair,
            a.capture_bf16_only,
            a.diagnostic_timing_layers,
        )
    ):
        p.error("branch timing cannot be combined with other diagnostics")
    if a.mxfp8_worker_confirm:
        if not (
            a.precision == "mxfp8"
            and a.model_graphs
            and a.shortcut_overlap
            and a.controlled_gc
            and a.mxfp8_worker_clusters is not None
            and a.cutlass_decode_profile_ids is not None
            and not a.branch_contention
            and a.mxfp8_ab_stage_reduction is None
        ):
            p.error(
                "worker confirmation requires MXFP8 overlap/full graphs/controlled GC, worker cap and CUTLASS IDs; no branch or pipeline diagnostic"
            )
    if a.mxfp8_ab_stage_reduction is not None:
        if not (
            a.precision == "mxfp8"
            and a.branch_contention
            and a.branch_backend == "direct"
            and a.model_graphs
            and a.mxfp8_worker_clusters == 152
        ):
            p.error(
                "pipeline experiment requires MXFP8 direct branch graphs and 152 worker clusters"
            )
    if a.mxfp8_worker_clusters is not None:
        if not (
            a.precision == "mxfp8"
            and (
                a.mxfp8_worker_confirm
                or (a.branch_contention and a.branch_backend == "direct")
            )
            and a.model_graphs
            and 1 <= a.mxfp8_worker_clusters <= 152
        ):
            p.error(
                "worker cap requires MXFP8 direct branch diagnostic with model graphs and 1..152 clusters"
            )
    if a.cutlass_decode_profile_ids is not None:
        if not (
            a.precision == "mxfp8"
            and a.shortcut_overlap
            and a.model_graphs
            and a.controlled_gc
        ):
            p.error(
                "CUTLASS decode tactics require MXFP8, shortcut overlap, model graphs and controlled GC"
            )
        if min(a.cutlass_decode_profile_ids) < 0:
            p.error("CUTLASS decode tactic IDs must be nonnegative")
        if a.branch_contention and a.branch_backend != "baseline":
            p.error("CUTLASS tactics require the baseline branch backend")
        if any(
            (
                a.tail_first_pair,
                a.overlap_headroom_sweep,
                a.vllm_variance_audit,
                a.eager_rank_audit,
                a.bf16_knobs_json,
                a.alternating_overlap_pairs,
                a.cutlass_overlap_pair,
                a.profile_direct_pair,
                a.host_timeline_pair,
                a.capture_bf16_only,
            )
        ):
            p.error(
                "CUTLASS decode tactic comparison cannot be combined with other experiments"
            )
    if a.shortcut_overlap and a.precision == "mxfp8":
        if not (a.model_graphs and a.controlled_gc):
            p.error("MXFP8 overlap comparison requires --model-graphs --controlled-gc")
        if any(
            (
                a.alternating_overlap_pairs,
                a.cutlass_overlap_pair,
                a.profile_direct_pair,
                a.host_timeline_pair,
                a.capture_bf16_only,
            )
        ):
            p.error("MXFP8 overlap uses its own CUTLASS/direct/CUTLASS comparison")
        source = (a.mcore_root / "megatron/core/transformer/moe/experts.py").read_text()
        if "class PreparedFlashInferMoE" not in source:
            p.error("sync the native FlashInfer shortcut stages to MCore first")
    if a.branch_contention and not (
        a.shortcut_overlap
        and a.model_graphs
        and (a.precision == "mxfp8" or a.bf16_knobs_json)
        and a.request_count == 128
        and not a.alternating_overlap_pairs
        and not a.cutlass_overlap_pair
        and not a.vllm_variance_audit
        and not a.eager_rank_audit
    ):
        p.error(
            "branch diagnostic requires 128 requests, model graphs, shortcut overlap and BF16 knobs when using BF16; no other comparisons"
        )
    if a.eager_rank_audit and (
        a.precision != "bf16"
        or not a.shortcut_overlap
        or a.model_graphs
        or a.routed_graphs
        or a.bf16_knobs_json
        or a.vllm_variance_audit
        or a.alternating_overlap_pairs
        or a.cutlass_overlap_pair
        or a.capture_bf16_only
        or a.profile_direct_pair
        or a.host_timeline_pair
    ):
        p.error(
            "eager rank audit requires BF16 + shortcut overlap without other comparison modes"
        )
    if a.vllm_variance_audit < 0:
        p.error("variance audit launch count must be nonnegative")
    if a.audit_model_events and not a.vllm_variance_audit:
        p.error("--audit-model-events requires --vllm-variance-audit")
    if a.vllm_variance_audit and not (
        a.shortcut_overlap
        and a.model_graphs
        and a.precision == "bf16"
        and not a.bf16_knobs_json
        and not a.alternating_overlap_pairs
        and not a.cutlass_overlap_pair
        and a.rounds > 0
    ):
        p.error(
            "variance audit requires BF16 model graphs + shortcut overlap without other comparison modes"
        )
    if a.cutlass_max_blocks is not None and (
        not a.cutlass_overlap_pair
        or a.cutlass_max_blocks < 0
        or a.cutlass_max_blocks > 4096
        or a.cutlass_max_blocks % 16
    ):
        p.error(
            "--cutlass-max-blocks requires --cutlass-overlap-pair and 0 or a multiple of 16 <= 4096"
        )
    if a.cutlass_overlap_pair and not (
        a.shortcut_overlap
        and a.model_graphs
        and a.precision == "bf16"
        and not a.bf16_knobs_json
        and not a.alternating_overlap_pairs
    ):
        p.error(
            "CUTLASS overlap requires BF16, model graphs and shortcut overlap; omit tuning knobs and alternating pairs"
        )
    if a.alternating_overlap_pairs < 0:
        p.error("alternating pair count must be nonnegative")
    if a.alternating_overlap_pairs and not (
        a.shortcut_overlap
        and a.model_graphs
        and a.precision == "bf16"
        and a.bf16_knobs_json
        and a.rounds > 0
    ):
        p.error(
            "alternating pairs require BF16, model graphs, shortcut overlap, candidate knobs and positive rounds"
        )
    if a.model_graphs:
        if (
            a.routed_graphs
            or a.profile_direct_pair
            or a.host_timeline_pair
            or a.capture_bf16_only
            or a.diagnostic_timing_layers
        ):
            p.error(
                "--model-graphs is a separate graph mode; do not combine with routed graphs or diagnostics"
            )
    if a.host_stack_sampling and not a.host_timeline_pair:
        p.error("--host-stack-sampling requires --host-timeline-pair")
    if a.host_timeline_pair:
        if not (a.precision == "bf16" and a.bf16_knobs_json and a.routed_graphs):
            p.error("host timeline requires BF16, candidate knobs and routed graphs")
        if a.profile_direct_pair or a.capture_bf16_only or a.diagnostic_timing_layers:
            p.error(
                "host timeline must run without profiler, event timing, or capture-only"
            )
    if a.profile_direct_pair:
        import shutil

        if not shutil.which("nsys"):
            p.error("--profile-direct-pair requires nsys on PATH")
        if not (a.precision == "bf16" and a.bf16_knobs_json and a.routed_graphs):
            p.error("profile pair requires BF16, candidate knobs, and routed graphs")
        if a.capture_bf16_only or a.diagnostic_timing_layers or a.rounds != 1:
            p.error("profile pair requires --rounds 1 and no capture-only/event timing")
    if a.diagnostic_timing_layers:
        try:
            if not all(int(x) >= 0 for x in a.diagnostic_timing_layers.split(",")):
                raise ValueError()
        except ValueError:
            p.error(
                "diagnostic timing layers must be comma-separated nonnegative integers"
            )
        if not a.routed_graphs:
            p.error("diagnostic timing requires --routed-graphs")
    if (a.bf16_knobs_json or a.capture_bf16_only) and a.precision != "bf16":
        p.error("BF16 capture/knobs require --precision bf16")
    from nt4_bf16_tuning_config import validate_knobs

    bf16_knobs = (
        validate_knobs(json.loads(a.bf16_knobs_json.read_text()))
        if a.bf16_knobs_json
        else {}
    )
    if a.capture_samples < 1:
        p.error("capture samples must be positive")
    if a.precision == "bf16" and not (
        a.routed_graphs or a.model_graphs or a.eager_rank_audit
    ):
        p.error("BF16 MegaMoE requires --routed-graphs or --model-graphs")
    if a.model_token_budget < a.request_count:
        p.error("token budget must accommodate the requested batch")
    if min(a.request_count, a.rounds, a.timeout_seconds, a.num_tokens_to_generate) < 1:
        p.error("counts and timeout must be positive")
    if a.model_graphs:
        argument_source = (a.mcore_root / "megatron/training/arguments.py").read_text()
        if (
            "CUDA graphs are not supported with --moe-shortcut-connection"
            in argument_source
        ):
            p.error(
                "MCore still has the old shortcut graph guard; sync the PR8257 integration checkout"
            )
        if a.shortcut_overlap:
            experts_source = (
                a.mcore_root / "megatron/core/transformer/moe/experts.py"
            ).read_text()
            if (
                "def prepare_shortcut_overlap(" not in experts_source
                or "--inference-shortcut-moe-overlap" not in argument_source
            ):
                p.error("shortcut overlap needs the complete MCore PR8257 checkout")
    manifest = json.loads((a.source_artifact / "manifest.json").read_text())
    if not manifest.get("capture_complete") or manifest.get("world_size") != 4:
        p.error("source must be a completed EP4 artifact")
    command = build_command(
        manifest,
        mcore_root=a.mcore_root,
        artifact=a.output,
        samples=1,
        counts="128",
        prompt_file=a.prompt_file,
        request_count=a.request_count,
        ignore_eos=True,
        model_token_budget=a.model_token_budget,
    )
    model = command[command.index("--") + 1 :]
    model = precision_options(model, a.precision)
    model = replace_model_options(
        model, {"--num-tokens-to-generate": a.num_tokens_to_generate}
    )
    if a.model_graphs:
        model = replace_model_options(
            model,
            {
                "--cuda-graph-impl": "local",
                "--inference-cuda-graph-scope": "block",
                "--cuda-graph-modules": None,
                "--cuda-graph-scope": None,
                "--cuda-graph-warmup-steps": 3,
                "--inference-dynamic-batching-num-cuda-graphs": 4,
                "--decode-only-cuda-graphs": None,
                "--inference-cuda-graph-all-prefills": None,
                "--inference-shortcut-moe-overlap": None,
                "--no-inference-shortcut-moe-overlap": None,
            },
        )
        model.append("--decode-only-cuda-graphs")
        if a.shortcut_overlap:
            model.append("--inference-shortcut-moe-overlap")
    if a.eager_rank_audit:
        model.append("--inference-shortcut-moe-overlap")
    trials = trial_modes(a.precision)
    if a.shortcut_overlap:
        if a.precision == "mxfp8":
            trials = [
                ("baseline-start", "baseline"),
                ("direct", "direct"),
                ("baseline-end", "baseline"),
            ]
        else:
            trials = [
                ("vllm-start", "vllm"),
                ("direct", "direct"),
                ("vllm-end", "vllm"),
            ]
    if a.bf16_knobs_json:
        trials.insert(1, ("direct-control", "direct"))
    if a.profile_direct_pair or a.host_timeline_pair:
        trials = [("direct-control", "direct"), ("direct", "direct")]
    if a.capture_bf16_only:
        trials = [("baseline-start", "baseline")]
    if a.alternating_overlap_pairs:
        trials = alternating_overlap_trials(a.alternating_overlap_pairs)
    if a.cutlass_overlap_pair:
        trials = [
            ("vllm-start", "vllm"),
            ("cutlass-overlap", "baseline"),
            ("vllm-end", "vllm"),
        ]
    if a.vllm_variance_audit:
        trials = [(f"audit-{i:02d}-vllm", "vllm") for i in range(a.vllm_variance_audit)]
    if a.eager_rank_audit:
        trials = [("eager-vllm", "vllm")]
    if a.branch_contention:
        trials = [(a.branch_backend, a.branch_backend)]
    if not a.model_graphs and (
        "--cuda-graph-impl" not in model
        or model[model.index("--cuda-graph-impl") + 1] != "none"
    ):
        p.error("source model must have CUDA graphs disabled")
    if a.branch_contention:
        shortcut_source = (
            a.mcore_root / "megatron/core/models/hybrid/shortcut_block.py"
        ).read_text()
        if 'probe.start("experts_combine")' not in shortcut_source:
            p.error("sync MCore shortcut_block.py with the branch-probe hooks first")
    checkout = Path(__file__).resolve().parents[2]
    check_source_headers(checkout)
    a.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["NT4_BRANCH_NATIVE_OUTPUT"] = "1" if a.branch_native_output else "0"
    env["NT4_BRANCH_CONTENTION"] = a.branch_contention or ""
    env["NT4_BRANCH_INPUT_MODE"] = a.branch_input_mode or ""
    env["NT4_BRANCH_INPUTS"] = str(a.branch_inputs.resolve()) if a.branch_inputs else ""
    env["NT4_EAGER_RANK_AUDIT"] = "1" if a.eager_rank_audit else "0"
    env["NT4_VARIANCE_AUDIT"] = "1" if a.vllm_variance_audit else "0"
    env["NT4_AUDIT_MODEL_EVENTS"] = "1" if a.audit_model_events else "0"
    if a.vllm_variance_audit:
        from nt4_variance_audit import sha256

        (a.output / "audit-inputs.json").write_text(
            json.dumps(
                dict(
                    prompt_file=str(a.prompt_file),
                    prompt_sha256=sha256(a.prompt_file),
                    source_manifest_sha256=sha256(a.source_artifact / "manifest.json"),
                    launches=a.vllm_variance_audit,
                    rounds=a.rounds,
                    model_events=a.audit_model_events,
                ),
                indent=2,
            )
        )
    env.pop("NT4_DEBUG_PREQUANT", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(checkout), str(a.mcore_root), env.get("PYTHONPATH", "")]
    )
    if a.profile_direct_pair:
        env["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"
    env["NT4_E2E_PROFILE"] = "1" if a.profile_direct_pair else "0"
    env["NT4_HOST_STACK_SAMPLING"] = "1" if a.host_stack_sampling else "0"
    env["NT4_HOST_TIMELINE"] = "1" if a.host_timeline_pair else "0"
    env["NT4_E2E_TIMING_LAYERS"] = a.diagnostic_timing_layers
    env["NT4_E2E_PRECISION"] = a.precision
    env["NT4_E2E_REQUEST_COUNT"] = str(a.request_count)
    env["NT4_E2E_ROUNDS"] = str(a.rounds)
    env["NT4_CONTROLLED_GC"] = "1" if a.controlled_gc else "0"
    env["NT4_SHORTCUT_OVERLAP"] = "1" if a.shortcut_overlap else "0"
    env["NT4_E2E_MODEL_GRAPHS"] = "1" if a.model_graphs else "0"
    env["NT4_E2E_ROUTED_GRAPHS"] = "1" if a.routed_graphs else "0"
    launch = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        *model,
    ]
    (a.output / "command.json").write_text(json.dumps(launch, indent=2))
    if a.tail_first_pair:
        trials = [
            ("original-start", "direct"),
            ("tail-first-0", "direct"),
            ("tail-first-1", "direct"),
            ("original-end", "direct"),
        ]
    if a.overlap_headroom_sweep:
        trials = [
            ("vllm-start", "vllm"),
            ("clusters76-start", "direct"),
            ("clusters74", "direct"),
            ("clusters72", "direct"),
            ("clusters68", "direct"),
            ("clusters76-end", "direct"),
            ("vllm-end", "vllm"),
        ]
    if a.cutlass_decode_profile_ids is not None and not a.branch_contention:
        trials = [
            ("baseline-start", "baseline"),
            ("cutlass-decode-tuned", "baseline"),
            ("baseline-end", "baseline"),
        ]
    if a.mxfp8_worker_confirm:
        trials = [
            ("baseline-start", "baseline"),
            ("direct", "direct"),
            ("baseline-end", "baseline"),
        ]
    if a.mxfp8_nvls_hybrid and not a.branch_contention:
        trials = [
            ("baseline-start", "baseline"),
            ("hybrid", "direct"),
            ("baseline-end", "baseline"),
        ]
    if a.mxfp8_hybrid_pairs:
        trials = hybrid_overlap_trials(a.mxfp8_hybrid_pairs)
    if a.trtllm_clamped_compare:
        trials = [
            ("baseline-start", "baseline"),
            ("hybrid", "direct"),
            ("trtllm-clamped", "direct"),
            ("baseline-end", "baseline"),
        ]
    # Repeat baseline at end to expose drift; each process warms before timing.
    for name, mode in trials:
        env["NT4_TRTLLM_CLAMPED"] = (
            "1" if a.trtllm_clamped_compare and name == "trtllm-clamped" else "0"
        )
        env["NT4_MXFP8_NVLS_HYBRID"] = (
            "1" if a.mxfp8_nvls_hybrid and mode == "direct" else "0"
        )
        env["NT4_TRTLLM_RELU2_SURROGATE"] = (
            "1" if a.trtllm_relu2_surrogate and mode == "direct" else "0"
        )
        env["NT4_HYBRID_MCORE_ROOT"] = str(a.mcore_root)
        env.pop("NT4_MXFP8_AB_STAGE_REDUCTION", None)
        if a.mxfp8_ab_stage_reduction is not None and mode == "direct":
            env["NT4_MXFP8_AB_STAGE_REDUCTION"] = str(a.mxfp8_ab_stage_reduction)
        env.pop("NT4_MXFP8_WORKER_CLUSTERS", None)
        if a.mxfp8_worker_clusters is not None and mode == "direct":
            env["NT4_MXFP8_WORKER_CLUSTERS"] = str(a.mxfp8_worker_clusters)
        env.pop("NT4_CUTLASS_DECODE_PROFILE_IDS", None)
        if a.cutlass_decode_profile_ids is not None and (
            name == "cutlass-decode-tuned"
            or a.branch_contention
            or ((a.mxfp8_worker_confirm or a.mxfp8_nvls_hybrid) and mode == "baseline")
        ):
            env["NT4_CUTLASS_DECODE_PROFILE_IDS"] = json.dumps(
                a.cutlass_decode_profile_ids
            )
        env["NT4_TAIL_FIRST"] = (
            "1" if a.tail_first_pair and name.startswith("tail-first-") else "0"
        )
        env["NT4_CUTLASS_OVERLAP"] = (
            "1" if a.cutlass_overlap_pair and mode == "baseline" else "0"
        )
        env.pop("NT4_CUTLASS_MAX_BLOCKS", None)
        if (
            a.cutlass_overlap_pair
            and mode == "baseline"
            and a.cutlass_max_blocks is not None
        ):
            env["NT4_CUTLASS_MAX_BLOCKS"] = str(a.cutlass_max_blocks)
        env["NT4_BF16_KNOBS"] = json.dumps(
            bf16_knobs
            if a.tail_first_pair
            or name == "direct"
            or (a.alternating_overlap_pairs and mode == "direct")
            else {}
        )
        if a.overlap_headroom_sweep:
            env["NT4_BF16_KNOBS"] = json.dumps(
                {"group_hint": 1024, "max_active_clusters": int(name[8:10])}
                if mode == "direct"
                else {}
            )
        env.pop("NT4_BF16_CAPTURE_ROOT", None)
        if a.capture_bf16_only:
            env["NT4_BF16_CAPTURE_ROOT"] = str(a.output.resolve() / "bf16-capture")
            env["NT4_BF16_CAPTURE_LAYERS"] = a.capture_layers
            env["NT4_BF16_CAPTURE_SAMPLES"] = str(a.capture_samples)
        env["NT4_E2E_MODE"] = mode
        env["NT4_E2E_OUTPUT"] = str(a.output / name)
        trial_model = replace_model_options(
            model,
            {
                "--inference-grouped-gemm-backend": (
                    "vllm" if mode == "vllm" or a.shortcut_overlap else "flashinfer"
                ),
                "--inference-moe-token-dispatcher-type": "nvls",
            },
        )
        if a.shortcut_overlap and a.precision == "mxfp8":
            # All arms share native CUTLASS+NVLS prefill/mixed execution. The
            # direct arm substitutes eligible decode experts only.
            trial_model = replace_model_options(
                trial_model,
                {
                    "--inference-grouped-gemm-backend": "flashinfer",
                    "--inference-shortcut-moe-expert-max-blocks": 0,
                },
            )
        trial_launch = launch[:5] + trial_model
        (a.output / f"{name}-command.json").write_text(
            json.dumps(trial_launch, indent=2)
        )
        if a.profile_direct_pair:
            trial_launch = [
                "nsys",
                "profile",
                (
                    "--trace=nvtx,osrt"
                    if a.profile_trace == "host"
                    else "--trace=cuda,nvtx,osrt"
                ),
                "--sample=none",
                "--cpuctxsw=none",
                "--trace-fork-before-exec=true",
                "--wait=all",
                "--capture-range=nvtx",
                "--nvtx-capture=nt4_profile_generation",
                "--capture-range-end=none",
                *(
                    [f"--cuda-graph-trace={a.profile_graph_trace}"]
                    if a.profile_trace == "cuda"
                    else []
                ),
                "--force-overwrite=false",
                "--output",
                str((a.output / name).resolve()),
                *trial_launch,
            ]
            (a.output / f"{name}-profile-command.json").write_text(
                json.dumps(trial_launch, indent=2)
            )
        log = a.output / f"{name}.log"
        print(json.dumps(dict(stage="e2e_start", mode=name, log=str(log))), flush=True)
        with log.open("w") as f:
            proc = subprocess.Popen(
                trial_launch,
                cwd=checkout,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = proc.wait(timeout=a.timeout_seconds)
            except subprocess.TimeoutExpired:
                import signal

                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise RuntimeError(f"E2E watchdog expired; see {log}")
        if code:
            raise RuntimeError(f"{name} failed ({code}); see {log}")
    if a.capture_bf16_only:
        capture_root = a.output.resolve() / "bf16-capture"
        manifest_path = capture_root / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Generation completed but BF16 capture manifest is missing: {manifest_path}. "
                "Do not start tuning. Check the child adapter revision and capture configuration."
            )
        captured = json.loads(manifest_path.read_text())
        layers = [int(x) for x in a.capture_layers.split(",")]
        if (
            not captured.get("complete")
            or captured.get("requests") != a.request_count
            or captured.get("layers") != layers
            or captured.get("samples") != a.capture_samples
        ):
            raise RuntimeError(
                f"BF16 capture manifest does not match this run: {manifest_path}"
            )
        for layer in layers:
            for rank in range(4):
                directory = capture_root / f"layer{layer}" / f"rank{rank}"
                required = [
                    "weights.pt",
                    "metadata.json",
                    *[f"sample{i}.pt" for i in range(a.capture_samples)],
                ]
                for filename in required:
                    if not (directory / filename).is_file():
                        raise RuntimeError(
                            f"Incomplete BF16 capture: {directory / filename}"
                        )
        print(
            json.dumps(
                dict(stage="bf16_capture_verified", manifest=str(manifest_path))
            ),
            flush=True,
        )
    report = {}
    for name, _ in trials:
        report[name] = json.loads((a.output / name / "results.json").read_text())
    comparison_reference = (
        "direct-control"
        if (a.profile_direct_pair or a.host_timeline_pair)
        else (
            "vllm-start"
            if a.shortcut_overlap and a.precision == "bf16"
            else "baseline-start"
        )
    )
    if a.alternating_overlap_pairs:
        comparison_reference = trials[0][0]
    if a.vllm_variance_audit:
        comparison_reference = trials[0][0]
    if a.tail_first_pair or a.branch_contention or a.eager_rank_audit:
        comparison_reference = trials[0][0]
    baseline = json.loads(
        (a.output / comparison_reference / "outputs-round1.json").read_text()
    )
    comparisons = []
    for name, _ in trials:
        for index in range(1, a.rounds + 1):
            other = json.loads(
                (a.output / name / f"outputs-round{index}.json").read_text()
            )
            assert len(other) == len(baseline), "request count changed"
            exact = sum(x == y for x, y in zip(baseline, other))
            mismatch = [
                next(
                    (i for i, (u, v) in enumerate(zip(x, y)) if u != v),
                    min(len(x), len(y)),
                )
                for x, y in zip(baseline, other)
                if x != y
            ]
            comparisons.append(
                dict(
                    mode=name,
                    round=index,
                    exact_requests=exact,
                    total_requests=len(other),
                    first_mismatch_token_positions=mismatch,
                )
            )
    report["host_stack_sampling"] = a.host_stack_sampling
    report["host_timeline_pair"] = a.host_timeline_pair
    report["comparison_reference"] = comparison_reference
    report["alternating_overlap_pairs"] = a.alternating_overlap_pairs
    report["cutlass_overlap_pair"] = a.cutlass_overlap_pair
    report["mxfp8_ab_stage_reduction"] = a.mxfp8_ab_stage_reduction
    report["mxfp8_worker_confirm"] = a.mxfp8_worker_confirm
    report["mxfp8_worker_clusters"] = a.mxfp8_worker_clusters
    report["cutlass_decode_profile_ids"] = a.cutlass_decode_profile_ids
    report["cutlass_max_blocks"] = a.cutlass_max_blocks
    report["vllm_variance_audit"] = a.vllm_variance_audit
    report["audit_model_events"] = a.audit_model_events
    report["profile_direct_pair"] = a.profile_direct_pair
    report["profile_trace"] = a.profile_trace if a.profile_direct_pair else None
    report["bf16_candidate_knobs"] = bf16_knobs
    report["branch_contention"] = a.branch_contention
    report["mxfp8_nvls_hybrid"] = a.mxfp8_nvls_hybrid
    report["mxfp8_hybrid_pairs"] = a.mxfp8_hybrid_pairs
    report["branch_native_output"] = a.branch_native_output
    if a.branch_native_output:
        report["native_continuation"] = {
            "policy": "cutlass_after_join_v1",
            "profile_ids": [39, 428],
            "measured_layers": [7, 32, 68],
            "other_layers": "native CUTLASS",
            "throughput_valid": False,
        }
    report["trtllm_clamped_compare"] = a.trtllm_clamped_compare
    if a.trtllm_clamped_compare:
        report["trial_backend_labels"] = {
            "baseline-start": "FlashInfer CUTLASS + NVLS [39,428]",
            "hybrid": "MegaMoE hybrid + NVLS",
            "trtllm-clamped": "TRTLLM-gen ClampedRelu2 + NVLS",
            "baseline-end": "FlashInfer CUTLASS + NVLS [39,428]",
        }
        report["trtllm_export"] = os.environ["FLASHINFER_TRTLLM_NT4_EXPORT"]
    report["trtllm_relu2_surrogate"] = a.trtllm_relu2_surrogate
    if a.trtllm_relu2_surrogate:
        report["surrogate_scope"] = (
            "Layers 7,32,68 only; unclamped Relu2; matched-input timing only; E2E throughput and quality invalid"
        )
    report["branch_backend"] = a.branch_backend if a.branch_contention else None
    report["branch_input_mode"] = a.branch_input_mode
    report["branch_inputs"] = str(a.branch_inputs) if a.branch_inputs else None
    report["controlled_gc"] = a.controlled_gc
    report["capture_only"] = a.capture_bf16_only
    report["num_tokens_to_generate"] = a.num_tokens_to_generate
    report["precision"] = a.precision
    report["request_count"] = a.request_count
    report["model_token_budget"] = a.model_token_budget
    report["backend_labels"] = dict(
        baseline="FlashInfer CUTLASS + NVLS",
        direct=(
            "TRTLLM-gen + NVLS (unclamped Relu2 surrogate, layers 7/32/68)"
            if a.trtllm_relu2_surrogate
            else (
                "MegaMoE local experts + NVLS (prototype)"
                if a.mxfp8_nvls_hybrid
                else "MegaMoE direct"
            )
        ),
        vllm=(
            "vLLM grouped GEMM + NVLS"
            if a.precision == "bf16"
            else "MCore scaled grouped GEMM + NVLS (vllm setting)"
        ),
    )
    report["output_comparisons"] = comparisons
    report["routed_graphs"] = a.routed_graphs
    report["model_graphs"] = a.model_graphs
    report["overlap_headroom_sweep"] = a.overlap_headroom_sweep
    report["tail_first_pair"] = a.tail_first_pair
    report["shortcut_overlap"] = a.shortcut_overlap
    if a.shortcut_overlap:
        report["overlap_contract"] = (
            "Native FlashInfer CUTLASS MXFP8+NVLS prefill/mixed and staged overlap; direct substitutes eligible decode routed branches; vLLM CTA cap disabled on both arms"
            if a.precision == "mxfp8"
            else (
                "PR8257 mixer/shared schedule; native vLLM+NVLS prefill; CUTLASS replaces decode expert compute only, preserving native NVLS dispatch/combine; vLLM expert cap does not apply to CUTLASS"
                if a.cutlass_overlap_pair
                else "PR8257 mixer/shared schedule; native vLLM+NVLS prefill; direct substitutes complete routed decode branch; vLLM expert cap does not apply to MegaMoE"
            )
        )
    report["scope"] = (
        "whole language-model decode graphs; eager prefill and request loop; eager/graph output gate"
        if a.model_graphs
        else "eager surrounding model; optional routed-section graphs; warmup excluded"
    )
    (a.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(dict(stage="e2e_complete", summary=str(a.output / "summary.json"))),
        flush=True,
    )


if __name__ == "__main__":
    main()
