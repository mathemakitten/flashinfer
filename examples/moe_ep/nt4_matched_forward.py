"""Fixed-workload decode timing; output replacement is diagnostic, not inference.

All shortcut inputs, routed inputs/probabilities, and block continuation tensors
are recorded. Replay restores block continuations so downstream model/cache
state follows the reference. Common copy/select work stays inside the timer;
CPU staging, verification, sampling, and file I/O stay outside it.
"""
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten
from nt4_capture_state import set_context
from nt4_gc_policy import controlled_gc


def select_blocks(model, block_type):
    """Select main-decoder MXFP8 blocks; exclude MTP and unchanged BF16 experts."""
    blocks = [m for m in model.decoder.modules() if isinstance(m, block_type)
              and m.moe_layer.mlp.experts._expert_weights_use_mxfp8()]
    layers = [m.moe_layer.layer_number for m in blocks]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError(f"Missing or duplicate main-decoder MXFP8 blocks: {layers}")
    return blocks


class FixedBlock:
    def __init__(self, block):
        self.layer = block.moe_layer.layer_number
        self.enabled = torch.zeros((), dtype=torch.bool, device="cuda")
        self.tag = torch.zeros((), dtype=torch.int64, device="cuda")
        self.buckets = {}
        self.key = None
        self.route = block._moe_router_preprocess
        self.post = block._postprocess
        block._nt4_branch_probe = self
        block._moe_router_preprocess = self.capture_route
        block._postprocess = self.capture_output
        self.block = block

    def begin(self, hidden):
        key = (tuple(hidden.shape), hidden.dtype)
        self.key = key
        if key not in self.buckets:
            self.buckets[key] = dict(tag=len(self.buckets),
                input=torch.empty_like(hidden), forced_input=torch.zeros_like(hidden))
        b = self.buckets[key]
        hidden.copy_(torch.where(self.enabled, b["forced_input"], hidden))
        b["input"].copy_(hidden)
        self.tag.fill_(b["tag"])

    def start(self, name):
        pass

    def end(self, name):
        pass

    def join(self):
        pass

    def snapshot(self, name, tensors):
        b = self.buckets[self.key]
        if name not in b:
            b[name] = [torch.empty_like(t) for t in tensors]
        if len(b[name]) != len(tensors):
            raise RuntimeError("Changed shortcut output structure")
        for dest, src in zip(b[name], tensors):
            dest.copy_(src)

    def capture_route(self, *args, **kwargs):
        result = self.route(*args, **kwargs)
        # These are the local routed input and probabilities before dispatch.
        if getattr(self.block.moe_layer.mlp, "_nt4_e2e_active", False):
            leaves, _ = tree_flatten(result)
            tensors = [t for t in leaves if isinstance(t, torch.Tensor)]
            tensors.append(self.block.moe_layer.mlp.token_dispatcher.routing_map)
            self.snapshot("route", tensors)
        return result

    def capture_output(self, *args, **kwargs):
        result = self.post(*args, **kwargs)
        if getattr(self.block.moe_layer.mlp, "_nt4_e2e_active", False):
            leaves, _ = tree_flatten(result)
            tensors = [t for t in leaves if isinstance(t, torch.Tensor)]
            b = self.buckets[self.key]
            if "forced_output" not in b:
                b["forced_output"] = [torch.zeros_like(t) for t in tensors]
            self.snapshot("raw_output", tensors)
            for tensor, fixed in zip(tensors, b["forced_output"]):
                tensor.copy_(torch.where(self.enabled, fixed, tensor))
            self.snapshot("output", tensors)
        return result

    def read(self):
        b = next(b for b in self.buckets.values() if b["tag"] == self.tag.item())
        return {"input": b["input"].cpu(),
                **{k: [t.cpu() for t in b[k]] for k in ("route", "output", "raw_output")}}

    def force(self, saved):
        t = saved["input"]
        b = self.buckets[(tuple(t.shape), t.dtype)]
        b["forced_input"].copy_(t)
        for dest, value in zip(b["forced_output"], saved["output"]):
            dest.copy_(value)
        self.enabled.fill_(True)


def check_block(actual, expected):
    if not torch.equal(actual["input"], expected["input"]):
        raise AssertionError("Shortcut input mismatch")
    for name in ("route", "output"):
        if len(actual[name]) != len(expected[name]) or any(
            not torch.equal(a, b) for a, b in zip(actual[name], expected[name])
        ):
            raise AssertionError(f"Shortcut {name} mismatch")


def run_probe(engine, args, tokenizer, sampling, build_requests, run_inference,
              adapter, model_session, destination):
    from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
    if model_session is None or engine.controller.num_speculative_tokens:
        raise ValueError("Matched forward requires full decode graphs without MTP")
    mode = os.environ["NT4_MATCHED_FORWARD_MODE"]
    if mode not in ("record", "replay"):
        raise ValueError("Invalid matched-forward mode")
    if adapter.mode != "direct" or not adapter.shortcut_overlap:
        raise ValueError("Matched forward requires hybrid overlap")
    # Explicitly exploratory: the raw candidate output is measured then replaced.
    # Never report this diagnostic as numerical qualification or generated tokens/s.
    if os.environ.get("NT4_HARDWARE_TANH") == "1":
        os.environ["NT4_TANH_FC1_ADVISORY"] = "1"
        os.environ["NT4_TANH_EXPERT_ADVISORY"] = "1"
        os.environ["NT4_TANH_OUTPUT_ADVISORY"] = "1"
    root = Path(os.environ["NT4_MATCHED_FORWARD_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    path = root / f"rank{rank}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True) if mode == "replay" else {}
    model = engine.controller._language_model
    blocks = select_blocks(model, ShortcutMoEBlock)
    print(json.dumps(dict(stage="matched_forward_discovery", rank=rank,
                         layers=[b.moe_layer.layer_number for b in blocks],
                         all_shortcut_layers=[m.moe_layer.layer_number for m in model.modules()
                                              if isinstance(m, ShortcutMoEBlock)],
                         scope="main decoder MXFP8 experts; MTP excluded")), flush=True)
    probes = [FixedBlock(m) for m in blocks]
    controller = engine.controller
    original = controller._dynamic_step_forward_logits
    rows = []
    limit = int(os.environ.get("NT4_MATCHED_FORWARD_SAMPLES", "8"))
    sampled = 0
    decode_index = 0
    measured = False
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def forward(*aa, **kw):
        nonlocal sampled, decode_index
        active = int(engine.context.get_active_request_count()) if engine.context.is_decode_only() else 0
        paired = measured and engine.context.is_decode_only() and sampled < limit
        selected = paired and active == 128
        if paired and mode == "replay":
            if decode_index not in saved:
                raise AssertionError("Missing recorded decode invocation")
            if saved[decode_index]["active"] != active:
                raise AssertionError("Recorded occupancy differs")
            for p in probes:
                p.force(saved[decode_index]["blocks"][p.layer])
        else:
            for p in probes:
                p.enabled.fill_(False)
        if selected:
            # All CPU staging/copies completed before common synchronized start.
            torch.cuda.synchronize()
            dist.barrier()
            start.record()
        result = original(*aa, **kw)
        if selected:
            end.record()
            end.synchronize()
            latency = start.elapsed_time(end) * 1000
        if paired:
            torch.cuda.synchronize()
            actual = {p.layer: p.read() for p in probes}
            logits = controller._all_logits_cuda[:, :active].detach().cpu().contiguous()
            if not torch.isfinite(logits).all():
                raise AssertionError("Nonfinite continuation logits")
            logit_hash = hashlib.sha256(logits.view(torch.uint8).numpy().tobytes()).hexdigest()
            for block in actual.values():
                if any(not torch.isfinite(t).all() for t in block["raw_output"]):
                    raise AssertionError("Nonfinite raw candidate block output")
            if mode == "record":
                saved[decode_index] = dict(blocks=actual, logits_sha256=logit_hash, active=active)
            else:
                for p in probes:
                    check_block(actual[p.layer], saved[decode_index]["blocks"][p.layer])
            if mode == "replay" and logit_hash != saved[decode_index]["logits_sha256"]:
                raise AssertionError("Continuation logits differ: workload not matched")
            decode_index += 1
        if selected:
            digest = hashlib.sha256()
            for layer in sorted(actual):
                for t in [actual[layer]["input"], *actual[layer]["route"]]:
                    digest.update(t.contiguous().view(torch.uint8).numpy().tobytes())
            row = dict(sample=sampled, rank=rank, forward_us=latency,
                       input_route_sha256=digest.hexdigest(), logits_sha256=logit_hash, layers=len(probes),
                       fixed_continuation=mode == "replay", active_requests=active)
            rows.append(row)
            print(json.dumps(dict(stage="matched_forward_sample", **row)), flush=True)
            sampled += 1
        return result

    controller._dynamic_step_forward_logits = forward
    try:
        # Eager preparation, graph warmup, then one sampled round. Fixed outputs
        # are needed only through the sampled prefix, which remains common.
        for round_id in (-1, 0, 1):
            if round_id == 0:
                model_session.build()
            model_session.begin_round(round_id >= 0)
            measured = round_id == 1
            sampled = 0
            decode_index = 0
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            engine.reset()
            requests = build_requests(args, tokenizer, sampling)
            set_context(engine.context)
            with controlled_gc(True):
                run_inference(requests, engine)
            set_context(None)
        covered = {id(p.block.moe_layer.mlp) for p in probes}
        if set(adapter.layers) != covered:
            raise AssertionError("Instrumented blocks do not cover exactly the executed MXFP8 layers")
        if sampled != limit:
            raise AssertionError(f"Only sampled {sampled}/{limit} full-occupancy forwards")
        if mode == "replay" and decode_index != len(saved):
            raise AssertionError("Recorded decode schedule differs")
        if mode == "record":
            torch.save(saved, path)
        report = dict(mode=mode, rows=rows, numerical_qualification="NOT_PERFORMED",
            scope="Full decode forward with all shortcut inputs and continuations fixed; copy/select overhead included; not E2E throughput",
            timing_includes="all layers and final logits; sampling excluded",
            matching="all executed MXFP8 shortcut inputs, routed inputs/probabilities/maps, and continuation tensors exact",
            layers=[p.layer for p in probes],
            hardware_tanh=os.environ.get("NT4_HARDWARE_TANH") == "1")
        (destination / f"matched-forward-rank{rank}.json").write_text(json.dumps(report, indent=2))
    finally:
        set_context(None)
        controller._dynamic_step_forward_logits = original
        model_session.close()
        adapter.close()
