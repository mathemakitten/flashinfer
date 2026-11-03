"""Validate an opt-in, local NT4 TRTLLM-gen artifact (no network or GPU imports)."""

import functools
import hashlib
import json
from pathlib import Path


@functools.lru_cache(maxsize=4)
def inspect_nt4_export(directory: str, headers: tuple):
    """Return artifact root and content identity. Exports are immutable per process."""
    root = Path(directory).expanduser().resolve(strict=True)
    validation = json.loads((root / "validation.json").read_text())
    if (
        validation.get("status") != "PASS"
        or validation.get("consumer_compile") != "PASS"
    ):
        raise ValueError(
            "NT4 export must pass generator checks and consumer compilation"
        )
    artifact = root / "artifact"
    manifest = artifact / "flashinferMetaInfo.h"
    files = [manifest, *(artifact / h for h in headers)]
    if "ClampedRelu2" not in (artifact / "Enums.h").read_text():
        raise ValueError("NT4 export headers lack ClampedRelu2")
    # Initial bring-up must include FC2 from this same generator build. Never
    # combine headers from main with descriptors built against the shipped ABI.
    configs = json.loads((root / "expanded-config.json").read_text())
    configs = configs["configs"] if isinstance(configs, dict) else configs
    fc1 = any(c.get("eltwiseActType") == "clampedrelu2" for c in configs)
    fc2 = any(
        c.get("eltwiseActType") == "none"
        and c.get("dtypeC") == "bf16"
        and c.get("routeAct") is False
        for c in configs
    )
    if not (fc1 and fc2):
        raise ValueError(
            "Export requires both clamped FC1 and BF16-output FC2; smoke-only export is insufficient"
        )
    cubins = sorted((artifact / "cubins").glob("*.cubin"))
    if not cubins:
        raise ValueError("NT4 export contains no raw cubins")
    files.extend(cubins)
    digest = hashlib.sha256()
    # Path is part of identity because it is embedded in the compiled cubin loader.
    digest.update(str(artifact).encode())
    digest.update((root / "expanded-config.json").read_bytes())
    for path in files:
        content = path.read_bytes()
        if not content:
            raise ValueError(f"Empty NT4 export file: {path}")
        digest.update(str(path.relative_to(artifact)).encode())
        digest.update(hashlib.sha256(content).digest())
    return artifact, digest.hexdigest()[:20]


def nt4_export_tiles(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    configs = json.loads((root / "expanded-config.json").read_text())
    configs = configs["configs"] if isinstance(configs, dict) else configs
    fc1 = {c["tileN"] for c in configs if c.get("eltwiseActType") == "clampedrelu2"}
    fc2 = {
        c["tileN"]
        for c in configs
        if c.get("eltwiseActType") == "none"
        and c.get("dtypeC") == "bf16"
        and c.get("routeAct") is False
    }
    tiles = sorted(fc1 & fc2)
    if not tiles or any(
        type(t) is not int or t not in (8, 16, 32, 64, 128, 256) for t in tiles
    ):
        raise ValueError("NT4 export has no supported matching FC1/FC2 tile inventory")
    return tiles
