"""Failure snapshots and single-GPU, no-checkpoint TRTLLM numerical replay."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import time

import torch


def tensor_hash(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def metrics(actual, reference):
    a, r = actual.double(), reference.double()
    d = a - r
    denom = r.norm(dim=-1).clamp_min(1e-8)
    rel = d.norm(dim=-1) / denom
    worst = int(rel.argmax()) if rel.numel() else 0
    return dict(
        global_rel_l2=float(d.norm() / r.norm().clamp_min(1e-8)),
        max_row_rel_l2=float(rel.max()) if rel.numel() else 0,
        max_abs=float(d.abs().max()) if d.numel() else 0,
        worst_row=worst,
        reference_norm=float(denom.flatten()[worst]) if rel.numel() else 0,
        finite=bool(torch.isfinite(a).all() and torch.isfinite(r).all()),
    )


def plain_tactic(tactic):
    """Persist FFI tactic arrays as Python integers, never runtime objects."""
    if tactic is None:
        return None
    if isinstance(tactic, int):
        return int(tactic)
    return [int(value) for value in tactic]


def load_snapshot(path, trusted_legacy=False):
    # Legacy snapshots from this diagnostic accidentally serialized an FFI
    # Array. Loading those executes pickle constructors; require explicit opt-in.
    saved = torch.load(path, map_location="cpu", weights_only=not trusted_legacy)
    saved["tactic"] = plain_tactic(saved.get("tactic"))
    return saved


def save_failure(h, actual, reference):
    tap = sys.modules.get("flashinfer.fused_moe.nt4_diagnostic")
    last_tactic = getattr(tap, "last_tactic", None)
    root = os.environ.get("NT4_TRT_FAILURE_ROOT") or os.environ.get("NT4_E2E_OUTPUT")
    if not root:
        print("TRT failure snapshot skipped: set NT4_TRT_FAILURE_ROOT", flush=True)
        return
    directory = Path(root) / "numerical-failures" / f"rank{h.rank}-{time.time_ns()}"
    directory.mkdir(parents=True)
    # One payload, local rank weights only. Preserve full routing geometry so
    # replay can reproduce tactic selection. No collectives or new GPU kernels.
    payload = dict(
        schema="nt4_trt_failure_v1",
        rank=h.rank,
        layer=getattr(h.direct, "layer_number", None),
        x=h.x.detach().cpu(),
        ids=h.ids.detach().cpu(),
        probs=h.probs.detach().cpu(),
        weights={k: v.detach().cpu() for k, v in h.direct.weights.items()},
        packed_weight_hashes={k: tensor_hash(v) for k, v in h.weights.items()},
        clamp_limit=None if h.clamp_limit is None else h.clamp_limit.detach().cpu(),
        actual=actual.detach().cpu(),
        reference=reference.detach().cpu(),
        export=os.environ.get("FLASHINFER_TRTLLM_NT4_EXPORT"),
        torch_version=str(torch.__version__),
        tactic=plain_tactic(last_tactic),
    )
    torch.save(payload, directory / "failure.pt")
    info = dict(
        complete=True,
        schema=payload["schema"],
        rank=h.rank,
        layer=payload["layer"],
        shape=list(h.x.shape),
        export=payload["export"],
        metrics=metrics(payload["actual"], payload["reference"]),
    )
    (directory / "manifest.json").write_text(json.dumps(info, indent=2))
    print(
        json.dumps(dict(stage="trtllm_failure_saved", artifact=str(directory), **info)),
        flush=True,
    )


def intermediate_sf_layout(export):
    """Require one consistent producer/consumer layout in this NT4 export."""
    path = Path(export) / "expanded-config.json"
    configs = json.loads(path.read_text())["configs"]
    producers = {
        c.get("sfLayoutC") for c in configs if c.get("eltwiseActType") == "clampedrelu2"
    }
    consumers = {
        c.get("sfLayoutB")
        for c in configs
        if c.get("dtypeA") == "mxe4m3"
        and c.get("dtypeB") == "mxe4m3"
        and c.get("dtypeC") == "bf16"
        and c.get("eltwiseActType") == "none"
    }
    if len(producers) != 1 or producers != consumers:
        raise ValueError(
            f"Ambiguous intermediate scale layouts: FC1={producers}, FC2={consumers}"
        )
    layout = next(iter(producers))
    if layout not in ("8x4", "128x4", "linear"):
        raise ValueError(f"Unsupported intermediate scale layout: {layout}")
    return layout


def sf_rows(raw, rows, width, layout="128x4"):
    """Decode scales using TRTLLM SfUtils::getSfOffset's selected layout."""
    from nt4_expert_diagnostics import decode_e8m0_exact

    k = torch.arange(width // 32, device=rows.device)[None, :]
    r = rows[:, None]
    if layout == "128x4":
        offsets = (
            ((r // 128) * ((width // 32 + 3) // 4) + k // 4) * 512
            + (r % 32) * 16
            + ((r % 128) // 32) * 4
            + k % 4
        )
    elif layout == "8x4":
        offsets = (
            ((r // 8) * ((width // 32 + 3) // 4) + k // 4) * 32 + (r % 8) * 4 + k % 4
        )
    elif layout == "linear":
        offsets = r * (width // 32) + k
    else:
        raise ValueError(f"Unsupported scale layout: {layout}")
    return decode_e8m0_exact(raw.view(torch.uint8).flatten()[offsets])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worst-tokens", type=int, default=4)
    p.add_argument("--qualify", action="store_true",
                   help="Check all rows and fail unless stage and overall gates pass; requires --graph")
    p.add_argument(
        "--graph",
        action="store_true",
        help="Also compare local CUDA graph replay; no NVLS or model graph",
    )
    p.add_argument(
        "--trusted-legacy-snapshot",
        action="store_true",
        help="Load an older snapshot containing FFI objects. Only use for a trusted capture you generated.",
    )
    a = p.parse_args()
    if a.qualify and not a.graph:
        p.error("--qualify requires --graph")
    if a.worst_tokens < 1:
        p.error("--worst-tokens must be positive")
    # Set before importing FlashInfer. This is a diagnostic binding rebuild,
    # not a rebuild of the exported TRTLLM-gen GEMM cubins.
    os.environ["NT4_TRT_DEBUG_INTERMEDIATES"] = "1"
    os.environ["FLASHINFER_DIST_AWARE_AUTOTUNE"] = "0"
    manifest = json.loads((a.artifact / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise RuntimeError("Incomplete capture")
    saved = load_snapshot(a.artifact / "failure.pt", a.trusted_legacy_snapshot)
    if saved["schema"] != "nt4_trt_failure_v1":
        raise ValueError("Unknown capture schema")
    if not os.environ.get("FLASHINFER_TRTLLM_NT4_EXPORT"):
        os.environ["FLASHINFER_TRTLLM_NT4_EXPORT"] = saved["export"]
    if os.environ["FLASHINFER_TRTLLM_NT4_EXPORT"] != saved["export"]:
        raise ValueError("Use the original TRTLLM export for this diagnostic")
    sf_layout = intermediate_sf_layout(saved["export"])
    a.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a, mxfp8_quantize
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe.nt4_diagnostic import take
    from flashinfer.fused_moe import nt4_diagnostic as tap
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        mxfp8_quantize_per_block_32,
    )
    from nt4_trtllm_surrogate import TrtllmNvlsSurrogate
    from nt4_expert_diagnostics import (
        decode_e8m0_exact,
        fc1_activation_interval,
        boundary_blocks,
    )

    h = object.__new__(TrtllmNvlsSurrogate)
    h.rank = saved["rank"]
    h.x = saved["x"].cuda()
    h.ids = saved["ids"].cuda()
    h.probs = saved["probs"].cuda()
    if saved["clamp_limit"] is None or not bool((saved["clamp_limit"] == 16).all()):
        raise ValueError("This boundary diagnostic requires NT4 clamp scale16")
    h.clamp_limit = saved["clamp_limit"].cuda()
    h.surrogate = False
    w = {k: v.cuda() for k, v in saved["weights"].items()}
    h.weights = {}
    for wk, sk in (("w1", "s1"), ("w2", "s2")):
        h.weights[wk] = torch.stack(
            [shuffle_matrix_a(t.view(torch.uint8), 128) for t in w[wk]]
        ).view(torch.float8_e4m3fn)
        h.weights[sk] = torch.stack(
            [shuffle_matrix_sf_a(t.view(torch.uint8), 128) for t in w[sk]]
        ).reshape(w[sk].shape)
    packed_hashes = {k: tensor_hash(v) for k, v in h.weights.items()}
    if packed_hashes != saved["packed_weight_hashes"]:
        raise RuntimeError("Repacked weights differ from the exact failing call")
    if saved.get("tactic") is not None:
        tap.forced_tactic = saved["tactic"]
    else:
        with autotune(tune_mode=True):
            h.local()
    out = h.local().clone()
    buffers, tactic = take()
    q1, s1, fc2, mapping, counts = buffers
    # Clone before another call can reuse temporary allocations.
    q1, s1, fc2, mapping = [v.clone() for v in (q1, s1, fc2, mapping)]
    counts = counts.clone() if counts is not None else None
    repeat = h.local().clone()
    take()
    report = dict(
        scope="local pre-NVLS; no throughput claims",
        tactic=str(tactic),
        snapshot_replay=metrics(out, saved["actual"].cuda()),
        repeat=metrics(repeat, out),
        saved_reference=metrics(out, saved["reference"].cuda()),
        raw_fc1_accumulator_available=False,
        prequant_activation_available=False,
        tensor_shapes=[list(v.shape) if v is not None else None for v in buffers],
        expert_counts_available=counts is not None,
        intermediate_scale_layout=sf_layout,
    )
    report["repeat_exact"] = bool(torch.equal(out, repeat))
    report["snapshot_exact"] = bool(torch.equal(out.cpu(), saved["actual"]))
    report["tactic_pinned"] = saved.get("tactic") is not None
    if a.graph:
        os.environ.pop("NT4_TRT_DEBUG_INTERMEDIATES")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = h.local()
        graph.replay()
        torch.cuda.synchronize()
        report["graph_vs_eager"] = metrics(graph_output, out)
        report["graph_exact"] = bool(torch.equal(graph_output, out))
        os.environ["NT4_TRT_DEBUG_INTERMEDIATES"] = "1"
    torch.save(
        dict(
            fc1_bytes=q1.cpu(),
            fc1_scales=s1.cpu(),
            fc2=fc2.cpu(),
            mapping=mapping.cpu(),
            counts=counts.cpu() if counts is not None else None,
            output=out.cpu(),
        ),
        a.output / "intermediates.pt",
    )
    ref = saved["reference"].cuda().double()
    report["total_rows"] = int(out.shape[0])
    report["expected_local_terms"] = int(
        ((h.ids >= h.rank * 128) & (h.ids < (h.rank + 1) * 128)).sum()
    )
    report["exact_zero_rows_preserved"] = bool((out[(ref == 0).all(-1)] == 0).all())
    rel = (out.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)
    tokens = rel.topk(len(rel) if a.qualify else min(a.worst_tokens, len(rel))).indices
    mapping = mapping.reshape(h.ids.shape).long()
    qx, sx = mxfp8_quantize(h.x, False, 32)
    rx, rs = mxfp8_quantize_per_block_32(h.x.float().contiguous(), torch.float8_e4m3fn)
    # mxfp8_quantize(...False) uses linear scales.
    dx = qx.float() * decode_e8m0_exact(
        sx.view(torch.uint8).reshape(rx.shape[0], -1)
    ).repeat_interleave(32, -1)
    reference_x = rx.float() * rs.float().repeat_interleave(32, -1)
    report["input_quantization"] = metrics(dx, reference_x)
    report["input_quantization"]["byte_mismatches"] = int(
        (qx.view(torch.uint8) != rx.view(torch.uint8)).sum()
    )
    report["input_quantization"]["scale_mismatches"] = int(
        (
            decode_e8m0_exact(sx.view(torch.uint8).reshape(rx.shape[0], -1))
            != rs.double()
        ).sum()
    )
    contributions = []
    records = []

    def dequant(q, s):
        return q.view(torch.float8_e4m3fn).double() * decode_e8m0_exact(
            s.view(torch.uint8)
        ).repeat_interleave(32, -1)

    def quant(v):
        q, s = mxfp8_quantize_per_block_32(v.float().contiguous(), torch.float8_e4m3fn)
        return q, s.double()

    for token in tokens.tolist():
        terms = []
        rounded_weight_terms = []
        staged = []
        original = []
        for slot, e in enumerate(h.ids[token].tolist()):
            if not h.rank * 128 <= e < (h.rank + 1) * 128:
                continue
            local = e - h.rank * 128
            perm = int(mapping[token, slot])
            if not 0 <= perm < min(q1.shape[0], fc2.shape[0]):
                raise RuntimeError("Invalid local row mapping")
            up = dequant(w["w1"][local], w["s1"][local])
            down = dequant(w["w2"][local], w["s2"][local])
            x = dx[token : token + 1].double()
            z = x @ up.T
            scale = h.clamp_limit[local].double()
            act = (scale * torch.tanh(z.clamp_min(0) / scale)).square()
            qr, sr = quant(act)
            ar = qr.double() * sr.repeat_interleave(32, -1)
            sf = sf_rows(
                s1, torch.tensor([perm], device="cuda"), q1.shape[-1], sf_layout
            )
            ak = q1[perm : perm + 1].view(
                torch.float8_e4m3fn
            ).double() * sf.repeat_interleave(32, -1)
            lo, hi = fc1_activation_interval(x, up, z)
            accepted = boundary_blocks(ak, ar, lo, hi, sf, quant)
            exact = (ak.reshape(-1, 32) == ar.reshape(-1, 32)).all(-1)
            predicted = ak @ down.T
            term = fc2[perm : perm + 1].double()
            weight = h.probs[token, slot].double()
            terms.append(term * weight)
            rounded_weight_terms.append(
                term.float() * weight.to(torch.bfloat16).float()
            )
            staged.append(predicted * weight)
            original.append((ar @ down.T) * weight)
            records.append(
                dict(
                    token=token,
                    slot=slot,
                    expert=e,
                    permuted_row=perm,
                    fc1_quantized=metrics(ak, ar),
                    fc1_unexplained_blocks=int((~(accepted | exact)).sum()),
                    fc2_using_actual_intermediate=metrics(term, predicted),
                    fc2_using_actual_intermediate_bf16=metrics(
                        term, predicted.to(torch.bfloat16)
                    ),
                    routing_weight=float(weight),
                )
            )
        summed = sum(
            terms, torch.zeros(1, h.x.shape[-1], device="cuda", dtype=torch.float64)
        )
        magnitude = sum((t.norm() for t in terms), torch.tensor(0.0, device="cuda"))
        contributions.append(
            dict(
                token=token,
                local_experts=len(terms),
                finalization=metrics(out[token : token + 1], summed),
                finalization_fp32_then_bf16=metrics(
                    out[token : token + 1],
                    sum(
                        [t.float() for t in terms], torch.zeros_like(summed).float()
                    ).to(torch.bfloat16),
                ),
                finalization_bf16_weights=metrics(
                    out[token : token + 1],
                    sum(rounded_weight_terms, torch.zeros_like(summed).float()).to(
                        torch.bfloat16
                    ),
                ),
                independent_fp64_sum=metrics(
                    out[token : token + 1], sum(original, torch.zeros_like(summed))
                ),
                fc2_fp64_sum_using_actual_intermediate=metrics(
                    out[token : token + 1], sum(staged, torch.zeros_like(summed))
                ),
                cancellation_ratio=float(magnitude / summed.norm().clamp_min(1e-30)),
            )
        )
    report["experts"] = records
    report["tokens"] = contributions
    report["interpretation"] = (
        "FC1 bound is provisional. Nonzero finalization error may indicate layout/weight-cast semantics; inspect raw tensors before attributing a defect. No numerical tolerance changed."
    )

    if a.qualify:
        from nt4_trtllm_qualification import qualify

        report["qualification"] = qualify(report)
    (a.output / "diagnostic.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if a.qualify and report["qualification"]["status"] != "PASS":
        raise AssertionError("TRTLLM qualification failed; see diagnostic.json qualification failures")


if __name__ == "__main__":
    main()
