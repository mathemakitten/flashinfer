"""Versioned, byte-preserving NT4 layer artifacts (CPU-safe, no MCore import)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

SCHEMA = "nt4_routed_checkpoint_v1"
WEIGHT_SHAPES = {
    "w1": (128, 4096, 768),
    "s1": (128, 4096, 24),
    "w2": (128, 768, 4096),
    "s2": (128, 768, 128),
}


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def canonical_scale(scale, rows, cols):
    """Invert the MCore 128x4 cuBLAS scale layout, checking the byte round trip."""
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = ((cols // 32 + 3) // 4) * 4
    raw = scale.contiguous().view(torch.uint8).reshape(-1)
    if raw.numel() != padded_rows * padded_cols:
        raise ValueError("unexpected MCore scale shape")
    logical = (
        raw.reshape(padded_rows // 128, padded_cols // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(padded_rows, padded_cols)
    )
    restored = (
        logical.reshape(padded_rows // 128, 4, 32, padded_cols // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(-1)
    )
    if not torch.equal(raw, restored):
        raise AssertionError("scale inverse transform did not preserve bytes")
    return logical[:rows, : cols // 32].contiguous()


def validate_weights(weights, rank):
    for key, shape in WEIGHT_SHAPES.items():
        value = weights[key]
        if tuple(value.shape) != shape or value.dtype != torch.uint8:
            raise ValueError(
                f"{key}: require raw uint8 {shape}, got {value.dtype} {value.shape}"
            )
        if key.startswith("s") and bool((value == 255).any()):
            raise ValueError(f"{key}: nonfinite E8M0 scales")
        if key.startswith("w") and bool(((value & 127) == 127).any()):
            raise ValueError(f"{key}: nonfinite E4M3 payload")
    if weights["expert_ids"].tolist() != list(range(rank * 128, (rank + 1) * 128)):
        raise ValueError("unsupported global expert mapping")


def validate_sample(sample):
    x, ids, probs, output = (sample[k] for k in ("x", "ids", "probs", "baseline"))
    n = x.shape[0]
    if x.shape != (n, 768) or x.dtype != torch.bfloat16:
        raise ValueError("latent input must be BF16 [T,768]")
    if ids.shape != (n, 10) or ids.dtype != torch.int64:
        raise ValueError("routes must be int64 [T,10]")
    if probs.shape != ids.shape or probs.dtype != torch.float32:
        raise ValueError("routing probabilities must be FP32 [T,10]")
    if output.shape != x.shape or output.dtype != torch.bfloat16:
        raise ValueError("baseline output must be BF16 [T,768]")
    if bool(((ids < -1) | (ids >= 512)).any()):
        raise ValueError("expert IDs outside [-1,512)")
    for value in (x, probs, output):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite sample")
    if bool((probs < 0).any()) or bool((probs[ids == -1] != 0).any()):
        raise ValueError("invalid routing probabilities")


def save_payload(path, payload):
    """Create once; never overwrite a previously captured sample."""
    path = Path(path)
    cpu = {k: v.detach().cpu().contiguous() for k, v in payload.items()}
    with path.open("xb") as stream:
        torch.save(cpu, stream)
    return {k: digest(v) for k, v in cpu.items()}


def load_payload(path, hashes):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != set(hashes):
        raise ValueError("artifact tensor keys differ from manifest")
    for key, value in payload.items():
        if digest(value) != hashes[key]:
            raise ValueError(f"artifact checksum mismatch: {key}")
    return payload


def save_manifest(path, manifest):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
