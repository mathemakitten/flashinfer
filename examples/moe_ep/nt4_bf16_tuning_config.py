"""Explicit NT4 BF16 scheduling candidates; no geometry or math changes."""

CANDIDATES = {
    "default": {},
    "group64": {"group_hint": 64},
    "group128": {"group_hint": 128},
    "group256": {"group_hint": 256},
    "group512": {"group_hint": 512},
    "group1024": {"group_hint": 1024},
    "workers76": {"group_hint": 1024, "max_active_clusters": 76},
    "workers74": {"group_hint": 1024, "max_active_clusters": 74},
    "workers72": {"group_hint": 1024, "max_active_clusters": 72},
    "workers68": {"group_hint": 1024, "max_active_clusters": 68},
    "workers56": {"group_hint": 1024, "max_active_clusters": 56},
    "workers38": {"group_hint": 1024, "max_active_clusters": 38},
    "atomic256": {"group_hint": 256, "load_balance_mode": "atomic_counter"},
    "group256_flag2": {"group_hint": 256, "flag_batch": 2},
    "group256_epi22": {"group_hint": 256, "epi_flag_batch": [2, 2]},
}


def validate_knobs(knobs):
    if not isinstance(knobs, dict):
        raise ValueError("BF16 knobs must be a JSON object")
    allowed = {
        "group_hint",
        "load_balance_mode",
        "flag_batch",
        "epi_flag_batch",
        "max_active_clusters",
    }
    if set(knobs) - allowed:
        raise ValueError(f"unsupported BF16 experiment knobs: {set(knobs) - allowed}")
    for key in ("group_hint", "flag_batch", "max_active_clusters"):
        if key in knobs and (type(knobs[key]) is not int or knobs[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    if knobs.get("load_balance_mode", "static") not in ("static", "atomic_counter"):
        raise ValueError("invalid load_balance_mode")
    if "epi_flag_batch" in knobs:
        value = knobs["epi_flag_batch"]
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(type(x) is not int or x < 1 for x in value)
        ):
            raise ValueError("epi_flag_batch must contain two positive integers")
    result = dict(knobs)
    if "epi_flag_batch" in result:
        result["epi_flag_batch"] = tuple(result["epi_flag_batch"])
    return result
