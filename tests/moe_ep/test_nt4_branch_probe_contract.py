"""CPU checks for diagnostic ordering; CUDA behavior requires the remote run."""

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples/moe_ep/nt4_branch_probe.py"


class Tensor:
    shape = (128, 768)
    dtype = "bf16"

    def detach(self):
        return self

    def copy_(self, other):
        pass

    def fill_(self, value):
        self.value = value


@pytest.mark.parametrize("clock", ["events", "device"])
@pytest.mark.parametrize("schedule,wait_count", [("serial", 5), ("overlap", 0)])
def test_phase_dependencies(schedule, wait_count, clock, monkeypatch):
    calls = []
    monkeypatch.setenv("NT4_BRANCH_CLOCK", clock)
    monkeypatch.setenv("NT4_BRANCH_TAIL", "0")
    monkeypatch.setitem(
        sys.modules,
        "nt4_branch_clock",
        SimpleNamespace(stamp=lambda output, slot: calls.append(("stamp", slot))),
    )

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == dict(
                enable_timing=clock == "events", external=clock == "events"
            )

        def record(self):
            calls.append(("record", self))

    stream = SimpleNamespace(
        cuda_stream=123, wait_event=lambda e: calls.append(("wait", e))
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, current_stream=lambda: stream),
        empty=lambda *a, **k: Tensor(),
        empty_like=lambda x: Tensor(),
        int64="int64",
    )
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.ClassDef))]
    ns = dict(torch=torch, os=os)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    probe = ns["BranchProbe"](7, schedule)
    for iteration in range(2):
        calls.clear()
        probe.begin(Tensor())
        for name in ns["PHASES"]:
            probe.start(name)
            probe.end(name)
        probe.join()
        waits = [e for op, e in calls if op == "wait"]
        assert len(waits) == wait_count
        if clock == "device":
            assert [slot for op, slot in calls if op == "stamp"] == list(range(14))
            assert len([op for op, _ in calls if op == "record"]) == (
                6 if schedule == "serial" else 0
            )
        if waits:
            assert waits == [probe.events[k][1] for k in ns["PHASES"][:-1]]
    assert len(probe.snapshots) == 1
    probe.begin(Tensor())
    with pytest.raises(RuntimeError, match="incomplete"):
        probe.join()


def test_tail_hooks_scope_and_return_values():
    tree = ast.parse(SOURCE.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "install_tail_hooks"
    )
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
    calls = []
    probe = SimpleNamespace(
        tail_active=False,
        tail_start=lambda name: calls.append(("start", name)),
        tail_end=lambda name: calls.append(("end", name)),
    )
    identity = lambda value: value
    fc1, fc2 = SimpleNamespace(forward=identity), SimpleNamespace(forward=identity)

    def shared(value):
        return fc2.forward(fc1.forward(value))

    mlp = SimpleNamespace(
        shared_experts_compute=shared,
        shared_experts=SimpleNamespace(linear_fc1=fc1, linear_fc2=fc2),
    )
    module = SimpleNamespace(
        compute_layer=SimpleNamespace(inference_post=identity),
        moe_layer=SimpleNamespace(
            _pre_mlp_layernorm_and_residual=identity,
            _maybe_unflatten_for_moe=identity,
            mlp=mlp,
        ),
    )
    ns["install_tail_hooks"](module, probe)
    sentinel = object()
    assert mlp.shared_experts_compute(sentinel) is sentinel
    assert not calls
    probe.tail_active = True
    assert mlp.shared_experts_compute(sentinel) is sentinel
    assert calls == [
        ("start", "shared_experts"),
        ("start", "shared_fc1"),
        ("end", "shared_fc1"),
        ("start", "shared_fc2"),
        ("end", "shared_fc2"),
        ("end", "shared_experts"),
    ]
