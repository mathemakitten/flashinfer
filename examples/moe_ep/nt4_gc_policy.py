"""Opt-in benchmark policy; does not change application GC defaults."""

from contextlib import contextmanager
import gc
import time


@contextmanager
def controlled_gc(enabled):
    previously_enabled = gc.isenabled()
    info = dict(
        policy="collect_before_round_disable_during_round" if enabled else "unchanged",
        previously_enabled=previously_enabled,
        collected=None,
        collection_seconds=0.0,
    )
    try:
        if enabled:
            start = time.perf_counter()
            info["collected"] = gc.collect(2)
            info["collection_seconds"] = time.perf_counter() - start
            gc.disable()
        yield info
    finally:
        if enabled:
            if previously_enabled:
                gc.enable()
            else:
                gc.disable()
