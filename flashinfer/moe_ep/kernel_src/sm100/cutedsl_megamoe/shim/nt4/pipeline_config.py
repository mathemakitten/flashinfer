"""Host-side NT4 pipeline experiment validation (no CUDA imports)."""


def reduce_ab_stages(a, b, reduction):
    if type(reduction) is not int or reduction not in (0, 1, 2):
        raise ValueError("AB stage reduction must be 0, 1 or 2")
    if reduction and min(a, b) - reduction < 2:
        raise ValueError(
            f"cannot remove {reduction} AB stages from ({a}, {b}); need at least two"
        )
    return a - reduction, b - reduction


def reduce_one_pipeline(a, b, side):
    """Change operand buffering for diagnostics, preserving at least two stages."""
    if side not in ("none", "a", "b", "a_plus_b_minus"):
        raise ValueError("NT4 pipeline side must be none, a, b or a_plus_b_minus")
    if side == "none":
        return a, b
    new_a = a + (side == "a_plus_b_minus") - (side == "a")
    new_b = b - (side in ("b", "a_plus_b_minus"))
    if min(new_a, new_b) < 2:
        raise ValueError(f"cannot reduce {side} pipeline from ({a}, {b}) below two")
    return new_a, new_b
