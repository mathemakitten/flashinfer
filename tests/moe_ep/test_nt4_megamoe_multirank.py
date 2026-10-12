"""Explicit GB300 qualification: torchrun --nproc-per-node=4 -m pytest this_file."""

import importlib.util
import os
from pathlib import Path
import sys

import pytest


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("mode", ["baseline", "nt4"])
@pytest.mark.parametrize("weights", ["sparse", "dense"])
def test_nt4_megamoe_oracle_and_replay(monkeypatch, graph, mode, weights):
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires torchrun with four ranks")
    path = (
        Path(__file__).resolve().parents[2] / "examples/moe_ep/nt4_megamoe_bringup.py"
    )
    spec = importlib.util.spec_from_file_location("nt4_bringup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [str(path), "--mode", mode, "--weights", weights, "--checks", "8"]
    if graph:
        argv.extend(["--graph", "--prequantized"])
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
