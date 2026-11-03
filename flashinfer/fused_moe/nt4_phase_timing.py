"""Private diagnostic callback. Device buffers remain owned by the caller."""
from contextlib import contextmanager
from contextvars import ContextVar

_sink = ContextVar("nt4_phase_sink", default=None)

@contextmanager
def capture(sink):
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)

def record(ticks):
    sink = _sink.get()
    if sink is not None:
        sink(ticks)
