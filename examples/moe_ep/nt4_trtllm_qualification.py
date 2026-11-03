"""Fail-closed qualification of complete TRTLLM stagewise replay reports."""

import math


def qualify(report):
    """Keep the existing 1.5% output gate independent of FC1 boundary checks.

    Stage L2 checks are provisional; this is not a model-quality assessment or
    proof of the generator's tanh accuracy contract.
    """
    failures = []

    def require(condition, label):
        if not condition:
            failures.append(label)

    def l2(value, label):
        require(value.get("finite") is True, label + ": nonfinite")
        for name in ("global_rel_l2", "max_row_rel_l2"):
            x = value.get(name, float("nan"))
            require(math.isfinite(x) and 0 <= x <= 0.015, label + ": " + name)

    total = report.get("total_rows", 0)
    tokens = report.get("tokens", [])
    require(total > 0 and {t["token"] for t in tokens} == set(range(total))
            and len(tokens) == total, "incomplete row coverage")
    for name in ("snapshot_exact", "repeat_exact", "graph_exact", "tactic_pinned",
                 "exact_zero_rows_preserved"):
        require(report.get(name) is True, name)
    inp = report.get("input_quantization", {})
    require(inp.get("byte_mismatches") == 0 and inp.get("scale_mismatches") == 0,
            "input quantization mismatch")
    l2(inp, "input quantization")
    for term in report.get("experts", []):
        label = f"token {term['token']} slot {term['slot']}"
        require(term.get("fc1_unexplained_blocks") == 0, label + ": FC1 boundary")
        require(term.get("fc1_quantized", {}).get("finite") is True,
                label + ": FC1 nonfinite")
        l2(term.get("fc2_using_actual_intermediate_bf16", {}), label + ": FC2")
    require(len(report.get("experts", [])) == report.get("expected_local_terms"),
            "incomplete expert coverage")
    for token in tokens:
        l2(token.get("finalization_fp32_then_bf16", {}),
           f"token {token['token']}: recombination")
    stage_failures = list(failures)
    l2(report.get("saved_reference", {}), "overall independent output")
    return dict(validation="trtllm_boundary_stagewise_v1_provisional",
                stages_status="FAIL" if stage_failures else "PASS",
                status="FAIL" if failures else "PASS", failures=failures,
                overall_l2_limit=0.015)
