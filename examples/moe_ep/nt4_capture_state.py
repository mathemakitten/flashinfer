"""Untimed engine metadata shared by the generation driver and capture hooks."""

_context = None


def set_context(context):
    global _context
    _context = context


def snapshot():
    if _context is None:
        return None
    return dict(
        decode_only=bool(_context.is_decode_only()),
        active_requests=int(_context.get_active_request_count()),
        active_tokens=int(_context.active_token_count),
        padded_tokens=int(_context.padded_active_token_count),
        prefill_requests=int(_context.num_prefill_requests),
    )


def eligible(states, *, min_active_tokens):
    return all(
        state is not None
        and state["decode_only"]
        and state["prefill_requests"] == 0
        and state["active_tokens"] >= min_active_tokens
        for state in states
    )


def filter_reasons(
    counts,
    states,
    *,
    call,
    skip_calls,
    max_local_tokens,
    last_capture_call,
    capture_interval,
    decode_only,
    min_active_tokens,
):
    reasons = []
    if call <= skip_calls:
        reasons.append("skip_calls")
    if max(counts) <= 0:
        reasons.append("empty_step")
    if max(counts) > max_local_tokens:
        reasons.append("padded_rows_exceed_limit")
    if last_capture_call is not None and call - last_capture_call < capture_interval:
        reasons.append("capture_interval")
    if decode_only:
        if any(s is None for s in states):
            reasons.append("missing_engine_context")
        else:
            if any(not s["decode_only"] or s["prefill_requests"] != 0 for s in states):
                reasons.append("prefill_or_mixed_step")
            if any(s["active_tokens"] < min_active_tokens for s in states):
                reasons.append("active_tokens_below_minimum")
    return reasons


class FilterStats:
    def __init__(self):
        self.calls = 0
        self.rejections = {}
        self.decode_calls = 0
        self.best_decode = None
        self.best_decode_within_row_limit = None
        self.last = None

    def observe(self, call, counts, states, reasons):
        self.calls += 1
        for reason in reasons:
            self.rejections[reason] = self.rejections.get(reason, 0) + 1
        record = dict(
            call=call, tokens_per_rank=counts, engine_steps=states, reasons=reasons
        )
        self.last = record
        improved = False
        if all(
            s is not None and s["decode_only"] and s["prefill_requests"] == 0
            for s in states
        ):
            self.decode_calls += 1
            score = min(s["active_tokens"] for s in states)
            if (
                self.best_decode is None
                or score > self.best_decode["min_active_tokens"]
            ):
                self.best_decode = dict(record, min_active_tokens=score)
                improved = True
            if "padded_rows_exceed_limit" not in reasons and (
                self.best_decode_within_row_limit is None
                or score > self.best_decode_within_row_limit["min_active_tokens"]
            ):
                self.best_decode_within_row_limit = dict(
                    record, min_active_tokens=score
                )
        return improved

    def report(self):
        return dict(
            observed_calls=self.calls,
            decode_calls=self.decode_calls,
            rejected_by_reason=self.rejections,
            best_decode=self.best_decode,
            best_decode_within_row_limit=self.best_decode_within_row_limit,
            last_observed=self.last,
        )
