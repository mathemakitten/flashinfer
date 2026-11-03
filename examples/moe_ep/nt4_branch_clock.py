"""GPU timestamps for the NT4 branch diagnostic (no CUDA timing-event nodes)."""

import triton
import triton.language as tl


@triton.jit
def _stamp(output, slot: tl.constexpr):
    now = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )
    tl.store(output + slot, now)


def stamp(output, slot):
    _stamp[(1,)](output, slot, num_warps=1)


@triton.jit
def _busy(output, slot: tl.constexpr, duration_ns: tl.constexpr):
    start = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )
    now = start
    while now - start < duration_ns:
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;",
            constraints="=l",
            args=[],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )
    tl.store(output + slot, start)
    tl.store(output + slot + 1, now)


def busy(output, slot, duration_ns=2_000_000):
    # One CTA per branch, leaving ample room for concurrent execution.
    _busy[(1,)](output, slot, duration_ns, num_warps=1)
